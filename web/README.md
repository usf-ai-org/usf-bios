# USF BIOS Web UI

Enterprise-grade web interface for AI model fine-tuning.

## Features

- **Multi-step Wizard**: Guided fine-tuning workflow
- **Real-time Updates**: WebSocket-based live training progress
- **All Training Types**: SFT, LoRA, QLoRA, AdaLoRA, Full fine-tuning
- **All Modalities**: Text, Vision, Audio, Video
- **Cloud Storage Ready**: Works with mounted Azure/AWS/GCP storage
- **Docker Deployment**: UI-only access for security

## Quick Start

### Development Mode

```bash
# Start backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Start frontend (in another terminal)
cd frontend
npm install
npm run dev
```

### Production (Docker)

```bash
cd web
docker-compose up -d
```

Access at: http://localhost:3000

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                │
│                    Port 3000                         │
└─────────────────────────────────────────────────────┘
                           │
                           │ REST API / WebSocket
                           ▼
┌─────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                 │
│                    Port 8000                         │
└─────────────────────────────────────────────────────┘
                           │
                           │ CLI / Python API
                           ▼
┌─────────────────────────────────────────────────────┐
│                    USF BIOS Core                     │
│              Training Engine (PyTorch)               │
└─────────────────────────────────────────────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models/supported` | GET | List supported models |
| `/api/models/validate` | POST | Validate model path |
| `/api/datasets/validate` | POST | Validate dataset format |
| `/api/datasets/upload` | POST | Upload dataset file |
| `/api/jobs/create` | POST | Create training job |
| `/api/jobs/{id}/start` | POST | Start training |
| `/api/jobs/{id}/stop` | POST | Stop training |
| `/api/jobs/{id}` | GET | Get job status |
| `/ws/jobs/{id}` | WS | Real-time job updates |

## Docker Security

When deployed via Docker, users can ONLY access the UI - no CLI access:

```bash
# Build and run
docker build -t usf-bios-ui .
docker run -p 3000:3000 -p 8000:8000 usf-bios-ui

# Users only see the web interface, no shell access
```

## Support

- **Documentation**: https://github.com/us-inc/usf-bios/docs
- **Support**: support@us.inc

---

Copyright (c) 2024-2026 US Inc. All Rights Reserved.
