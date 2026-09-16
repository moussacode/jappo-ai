# JAPPO AI Orchestrator

FastAPI service for AI orchestration in the JAPPO platform.

## Architecture

```
Angular → Spring Boot → FastAPI → LLM
```

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the service:
```bash
uvicorn app.main:app --reload --port 8000
```

## Endpoints

- `GET /health` - Health check
- `POST /api/v1/chat` - Chat endpoint for AI responses

## Configuration

See `.env.example` for available configuration options.
