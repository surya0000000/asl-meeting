"""Training entrypoint for temporal ASL gesture classification."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
from typing import Sequence

import numpy as np

from ml import DEFAULT_VOCABULARY
from ml.gesture_model import ModelConfig, build_model
from ml.synthetic_dataset import (
    SyntheticDataConfig,
    SyntheticGestureDataset,
    generate_synthetic_sequences,
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(
        "PyTorch is required for training. Install ml/requirements.txt first.",
    ) from exc


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal ASL gesture model.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--samples-per-class", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--architecture", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--vocabulary", type=str, default="")
    parser.add_argument("--vocabulary-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("ml/models/gesture_model.pt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def resolve_vocabulary(args: argparse.Namespace) -> list[str]:
    if args.vocabulary_file is not None:
        loaded = json.loads(args.vocabulary_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, list) or not all(isinstance(v, str) for v in loaded):
            raise ValueError("Vocabulary file must contain a JSON list of strings.")
        return [v.strip().upper() for v in loaded if v.strip()]
    if args.vocabulary:
        return [item.strip().upper() for item in args.vocabulary.split(",") if item.strip()]
    return DEFAULT_VOCABULARY.copy()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_indices(length: int, val_ratio: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    indices = np.random.permutation(length)
    cut = int(length * (1.0 - val_ratio))
    return indices[:cut], indices[cut:]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * x.size(0)
    return running_loss / max(1, len(loader.dataset))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            total_loss += float(loss.item()) * x.size(0)
            total_correct += int((preds == y).sum().item())
            total_items += int(x.size(0))
    avg_loss = total_loss / max(1, total_items)
    accuracy = total_correct / max(1, total_items)
    return avg_loss, accuracy


def main() -> None:
    configure_logging()
    args = parse_args()
    set_seed(args.seed)
    vocabulary = resolve_vocabulary(args)

    LOGGER.info("Using vocabulary (%d classes): %s", len(vocabulary), ", ".join(vocabulary))
    synth_cfg = SyntheticDataConfig(
        vocabulary=vocabulary,
        samples_per_class=args.samples_per_class,
        sequence_length=args.sequence_length,
        feature_dim=63,
    )
    x, y = generate_synthetic_sequences(synth_cfg)
    train_idx, val_idx = split_indices(len(x))

    train_dataset = SyntheticGestureDataset(x[train_idx], y[train_idx])
    val_dataset = SyntheticGestureDataset(x[val_idx], y[val_idx])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model_cfg = ModelConfig(
        input_size=63,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=len(vocabulary),
        architecture=args.architecture,
    )
    model = build_model(model_cfg)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        LOGGER.info(
            "Epoch %d/%d - train_loss=%.4f val_loss=%.4f val_acc=%.3f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            val_acc,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "vocabulary": vocabulary,
        "input_size": 63,
        "sequence_length": args.sequence_length,
        "architecture": args.architecture,
    }
    torch.save(checkpoint, args.output)
    LOGGER.info("Saved model checkpoint to %s", args.output)


if __name__ == "__main__":
    main()

