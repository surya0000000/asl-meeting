"""Training entrypoint for temporal ASL gesture classification."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random

import numpy as np

from ml import DEFAULT_VOCABULARY
from ml.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LANDMARK_DIM,
    MAX_EPOCHS,
    SEQUENCE_LENGTH,
)
from ml.models.model_factory import get_model
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
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--samples-per-class", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--architecture", choices=["lstm", "transformer", "hybrid"], default="hybrid")
    parser.add_argument("--vocabulary", type=str, default="")
    parser.add_argument("--vocabulary-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("ml/models/gesture_model.pt"))
    parser.add_argument("--onnx-output", type=Path, default=Path("ml/models/gesture_model.onnx"))
    parser.add_argument("--label-map-output", type=Path, default=Path("ml/models/label_map.json"))
    parser.add_argument("--early-stopping-patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
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
    torch.cuda.manual_seed_all(seed)


def split_indices(length: int, val_ratio: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    indices = np.random.permutation(length)
    cut = int(length * (1.0 - val_ratio))
    return indices[:cut], indices[cut:]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    scaler: "torch.cuda.amp.GradScaler",
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += float(loss.item()) * x.size(0)
    return running_loss / max(1, len(loader.dataset))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
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
            all_true.append(y.detach().cpu().numpy())
            all_pred.append(preds.detach().cpu().numpy())
    avg_loss = total_loss / max(1, total_items)
    accuracy = total_correct / max(1, total_items)
    y_true = np.concatenate(all_true, axis=0) if all_true else np.empty((0,), dtype=np.int64)
    y_pred = np.concatenate(all_pred, axis=0) if all_pred else np.empty((0,), dtype=np.int64)
    return avg_loss, accuracy, y_true, y_pred


def _build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true.tolist(), y_pred.tolist()):
        matrix[int(truth), int(pred)] += 1
    return matrix


def _print_top_confusions(
    matrix: np.ndarray,
    vocabulary: list[str],
    limit: int = 20,
) -> None:
    entries: list[tuple[int, int, int]] = []
    for truth in range(matrix.shape[0]):
        for pred in range(matrix.shape[1]):
            if truth == pred:
                continue
            count = int(matrix[truth, pred])
            if count > 0:
                entries.append((count, truth, pred))

    entries.sort(key=lambda item: item[0], reverse=True)
    top = entries[:limit]
    if not top:
        LOGGER.info("No confusion pairs found in validation predictions.")
        return

    LOGGER.info("Top %d most-confused class pairs:", len(top))
    for rank, (count, truth, pred) in enumerate(top, start=1):
        LOGGER.info(
            "%2d) true=%s -> pred=%s : %d",
            rank,
            vocabulary[truth],
            vocabulary[pred],
            count,
        )


def export_onnx(model: nn.Module, output_path: Path, sequence_length: int) -> None:
    """Export trained model to ONNX with dynamic batch/sequence axes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, sequence_length, LANDMARK_DIM, dtype=torch.float32)

    torch.onnx.export(
        model_cpu,
        dummy,
        output_path,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={
            "landmarks": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    LOGGER.info("Exported ONNX model to %s", output_path)


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
        feature_dim=LANDMARK_DIM,
    )
    x, y = generate_synthetic_sequences(synth_cfg)
    train_idx, val_idx = split_indices(len(x))

    train_dataset = SyntheticGestureDataset(x[train_idx], y[train_idx])
    val_dataset = SyntheticGestureDataset(x[val_idx], y[val_idx])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = get_model(
        architecture=args.architecture,
        num_classes=len(vocabulary),
        sequence_length=args.sequence_length,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=args.epochs,
        steps_per_epoch=max(1, len(train_loader)),
        pct_start=0.1,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    best_val_true = np.empty((0,), dtype=np.int64)
    best_val_pred = np.empty((0,), dtype=np.int64)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        val_loss, val_acc, val_true, val_pred = evaluate(model, val_loader, criterion, device)
        LOGGER.info(
            "Epoch %d/%d - train_loss=%.4f val_loss=%.4f val_acc=%.3f lr=%.6f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            val_acc,
            optimizer.param_groups[0]["lr"],
        )
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_true = val_true
            best_val_pred = val_pred
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stopping_patience:
            LOGGER.info(
                "Early stopping triggered at epoch %d (best epoch=%d, best val_acc=%.3f).",
                epoch,
                best_epoch,
                best_val_accuracy,
            )
            break

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    label_map = {str(index): gloss for index, gloss in enumerate(vocabulary)}
    checkpoint = {
        "model_state_dict": best_state,
        "vocabulary": vocabulary,
        "label_map": label_map,
        "input_size": LANDMARK_DIM,
        "sequence_length": args.sequence_length,
        "architecture": args.architecture,
    }
    torch.save(checkpoint, args.output)
    LOGGER.info("Saved model checkpoint to %s", args.output)

    args.label_map_output.parent.mkdir(parents=True, exist_ok=True)
    args.label_map_output.write_text(json.dumps(label_map, ensure_ascii=True, indent=2), encoding="utf-8")
    LOGGER.info("Saved label map to %s", args.label_map_output)

    export_onnx(model, args.onnx_output, sequence_length=args.sequence_length)

    confusion = _build_confusion_matrix(
        y_true=best_val_true,
        y_pred=best_val_pred,
        num_classes=len(vocabulary),
    )
    _print_top_confusions(confusion, vocabulary=vocabulary, limit=20)


if __name__ == "__main__":
    main()

