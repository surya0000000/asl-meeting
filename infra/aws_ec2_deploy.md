# AWS EC2 Deployment Guide

This deployment uses Docker Compose to run both backend and frontend services.

## 1) Provision infrastructure

1. Launch an EC2 instance (Ubuntu 22.04 recommended).
2. Security Group inbound rules:
   - `22` (SSH) from your IP
   - `80` (HTTP) from `0.0.0.0/0`
   - `443` (HTTPS) from `0.0.0.0/0` (if TLS is enabled later)
   - `8000` optional (only for direct backend access/testing)

## 2) Install runtime dependencies

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin git
sudo usermod -aG docker $USER
```

Re-login to apply Docker group membership.

## 3) Clone repository and configure env

```bash
git clone <YOUR_REPOSITORY_URL> asl-meeting-copilot
cd asl-meeting-copilot
cp .env.example .env
```

Update `.env`:

- `OPENAI_API_KEY=...` (optional, for LLM correction)
- `ENABLE_TTS=true|false`
- `MODEL_CHECKPOINT_PATH=ml/models/gesture_model.pt`

## 4) Build and run

```bash
docker compose build
docker compose up -d
docker compose ps
```

## 5) Verify

```bash
curl http://<EC2_PUBLIC_IP>:8000/health
```

Frontend should be available at:

`http://<EC2_PUBLIC_IP>:5173`

## 6) Optional production hardening

- Add Nginx reverse proxy with TLS (Let’s Encrypt).
- Restrict direct access to port `8000` after proxying.
- Store secrets in AWS Systems Manager Parameter Store or AWS Secrets Manager.
- Add CloudWatch logging and alerts.

