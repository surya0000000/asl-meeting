# ASL Meeting Copilot

Production-quality research prototype for real-time ASL-to-subtitles-and-speech in virtual meetings (Zoom/Google Meet).

## Overview

ASL Meeting Copilot captures webcam video, extracts MediaPipe hand landmarks, runs temporal gesture recognition (PyTorch LSTM/Transformer), refines output text, and optionally generates speech audio.

It is built as a modular monorepo:

- `ml/` for camera + landmark + temporal model training/inference
- `backend/` for FastAPI REST/WebSocket streaming API
- `frontend/` for React real-time subtitle interface
- `docker/` + `infra/` for containerization and cloud deployment

## Architecture Diagram

```text
┌──────────────┐      ┌──────────────────────┐      ┌──────────────────────────┐
│ Webcam Input │ ───► │ MediaPipe Landmarks │ ───► │ Temporal Gesture Model   │
│ (30 FPS)     │      │ (21 x,y,z = 63 dims)│      │ (LSTM / Transformer)     │
└──────────────┘      └──────────────────────┘      └────────────┬─────────────┘
                                                                  │
                                                                  ▼
                                                      ┌──────────────────────────┐
                                                      │ FastAPI Stream Processor │
                                                      │ /ws/asl-stream           │
                                                      │ /predict                 │
                                                      └────────────┬─────────────┘
                                                                   │
                                            ┌──────────────────────┼──────────────────────┐
                                            ▼                      ▼                      ▼
                               ┌────────────────────┐   ┌───────────────────┐   ┌──────────────────┐
                               │ Text Refiner       │   │ Frontend Subtitles│   │ Speech Engine     │
                               │ (OpenAI optional)  │   │ (React + Vite)    │   │ (pyttsx3 .wav)    │
                               └────────────────────┘   └───────────────────┘   └──────────────────┘
```

## Project Structure

```text
.
├── backend
│   ├── app
│   │   ├── api
│   │   ├── core
│   │   ├── schemas
│   │   └── services
│   ├── tests
│   ├── Dockerfile
│   └── requirements.txt
├── ml
│   ├── camera_capture.py
│   ├── hand_landmarks.py
│   ├── gesture_model.py
│   ├── sequence_buffer.py
│   ├── inference_engine.py
│   ├── synthetic_dataset.py
│   ├── train.py
│   ├── infer.py
│   ├── requirements.txt
│   └── tests
├── frontend
│   ├── src
│   ├── Dockerfile
│   └── package.json
├── docker
│   └── nginx.conf
├── docs
│   └── virtual_microphone.md
├── infra
│   ├── render.yaml
│   ├── render_deploy.md
│   └── aws_ec2_deploy.md
├── docker-compose.yml
└── .env.example
```

## Supported Gesture Vocabulary

Default vocabulary (configurable via environment):

- HELLO
- YES
- NO
- WAIT
- QUESTION
- AGREE
- DISAGREE
- SLOW DOWN
- THANK YOU

## Real-Time Design Notes

- Target throughput: **30 FPS**
- Camera capture is threaded + async-consumed to avoid UI freezing.
- Inference is sequence-based (sliding temporal window), not static single-frame classification.
- Backend uses non-blocking `asyncio.to_thread` around inference/TTS work.
- CPU fallback works even without GPU.

## Quick Start (Local Development)

### 1) Prerequisites

- Python 3.11+
- Node.js 20+ (Node 22 recommended)
- npm
- Webcam access

### 2) Configure environment

```bash
cp .env.example .env
```

Edit `.env` as needed (`OPENAI_API_KEY` optional).

### 3) Install dependencies

Backend + ML:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt -r ml/requirements.txt
```

Frontend:

```bash
cd frontend
npm install
cd ..
```

### 4) Train temporal model (synthetic starter dataset)

```bash
python -m ml.train \
  --epochs 12 \
  --samples-per-class 256 \
  --sequence-length 30 \
  --architecture lstm \
  --output ml/models/gesture_model.pt
```

### 5) Run backend

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6) Run frontend

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Open: `http://localhost:5173`

### 7) Run real-time ASL stream client (camera + landmarks)

In a third terminal:

```bash
python -m ml.infer \
  --show-preview \
  --backend-ws-url ws://localhost:8000/ws/asl-stream
```

This performs:

`camera -> landmarks -> websocket -> backend inference -> refinement -> subtitles/speech`

## API

### GET `/health`

Service health and checkpoint status.

### POST `/predict`

Request body options:

- `landmarks: [63 floats]` for streaming frame mode
- `sequence: [[63 floats], ...]` for direct sequence prediction
- `refine_text: bool`
- `speak: bool`

### WebSocket `/ws/asl-stream`

Input payload:

```json
{
  "type": "landmarks",
  "landmarks": [0.01, 0.02, "...63 dims total..."],
  "refine_text": true,
  "speak": false
}
```

Output payload:

```json
{
  "type": "prediction",
  "raw_prediction": "QUESTION",
  "refined_text": "I have a question.",
  "confidence": 0.91,
  "audio_url": "/audio/tts-xxxx.wav",
  "frame_index": 45,
  "timestamp": "2026-02-23T12:34:56.000000+00:00"
}
```

## Testing

Run all tests:

```bash
pytest
```

Included tests:

- `ml/tests/test_sequence_buffer.py`
- `backend/tests/test_health.py`
- `backend/tests/test_websocket.py`
- `backend/tests/test_predict.py`

## Docker

Build images:

```bash
docker compose build
```

Run stack:

```bash
docker compose up -d
```

View logs:

```bash
docker compose logs -f backend
docker compose logs -f frontend
```

Stop:

```bash
docker compose down
```

## Cloud Deployment

### Render

- Blueprint: `infra/render.yaml`
- Guide: `infra/render_deploy.md`

### AWS EC2

- Guide: `infra/aws_ec2_deploy.md`

## Virtual Microphone Integration

See:

- `docs/virtual_microphone.md`

Covers:

- macOS + BlackHole
- Windows + VB-CABLE
- Linux PulseAudio/PipeWire

## Environment Variables

Key variables from `.env.example`:

- `HOST`, `PORT`, `LOG_LEVEL`
- `GESTURE_VOCAB`
- `MODEL_CHECKPOINT_PATH`
- `ENABLE_TEXT_REFINER`
- `ENABLE_TTS`
- `OPENAI_API_KEY`
- `VITE_API_BASE_URL`
- `VITE_WS_URL`

## Demo Runbook

1. Train model (`python -m ml.train ...`)
2. Start backend (`uvicorn ...`)
3. Start frontend (`npm run dev`)
4. Start webcam stream client (`python -m ml.infer --backend-ws-url ...`)
5. View subtitles live in browser
6. Enable speech playback toggle to hear generated audio

## Notes

- This is a research prototype intended to be extended with real labeled ASL sequence datasets and production observability.
- For low-latency production usage, keep model compact and run backend on CPU-optimized instances or GPU if available.