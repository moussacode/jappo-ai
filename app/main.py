from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.chat import router as chat_router
from app.core.config import settings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="JAPPO AI Orchestrator",
    description="FastAPI service for AI orchestration in the JAPPO platform",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, tags=["chat"])


@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("JAPPO AI Orchestrator starting...")
    logger.info(f"Environment: {settings.fastapi_env}")
    logger.info(f"LLM Model: {settings.llm_model}")
    logger.info(f"Max history messages: {settings.max_history_messages}")


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information."""
    logger.info("JAPPO AI Orchestrator shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=True
    )
