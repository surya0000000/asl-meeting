# Render Deployment Guide

This repository includes `infra/render.yaml` for blueprint-based deployment.

## Prerequisites

- GitHub repository connected to Render
- Render account with permission to create Web and Static services

## Steps

1. Push repository to GitHub.
2. In Render dashboard, choose **New > Blueprint**.
3. Select your repository.
4. Render detects `infra/render.yaml`.
5. Confirm services:
   - `asl-meeting-copilot-backend` (Docker web service)
   - `asl-meeting-copilot-frontend` (static site)
6. Set required secret:
   - `OPENAI_API_KEY` (optional, can be left empty to use rule-based refiner)
7. Deploy.

## Environment variables

Backend:

- `PORT=8000`
- `HOST=0.0.0.0`
- `LOG_LEVEL=INFO`
- `ENABLE_TEXT_REFINER=true`
- `ENABLE_TTS=false`
- `MODEL_CHECKPOINT_PATH=ml/models/gesture_model.pt`
- `GESTURE_VOCAB=HELLO,YES,NO,WAIT,QUESTION,AGREE,DISAGREE,SLOW DOWN,THANK YOU`

Frontend:

- `VITE_API_BASE_URL=https://<backend-service>.onrender.com`
- `VITE_WS_URL=wss://<backend-service>.onrender.com/ws/asl-stream`

## Post-deploy verification

- Health: `GET https://<backend-service>.onrender.com/health`
- Frontend: open static site URL and confirm connection indicator is `connected`.
- WebSocket: use `ml/infer.py --backend-ws-url wss://<backend-service>.onrender.com/ws/asl-stream`.

