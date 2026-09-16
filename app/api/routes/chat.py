from fastapi import APIRouter, HTTPException, Header, status
from app.schemas.chat import ChatRequest, ChatResponse, HealthResponse
from app.services.orchestrator import OrchestratorService
from app.services.llm import OpenAIProvider
from app.services.llm.provider import (
    LLMTimeoutError,
    LLMUnavailableError,
    LLMRateLimitError,
    LLMBadResponseError,
)
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
llm_provider = OpenAIProvider()
orchestrator = OrchestratorService(llm_provider)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@router.post("/api/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    x_internal_api_key: str = Header(None, alias="X-Internal-Api-Key")
):
    """
    Chat endpoint for AI responses.

    This endpoint receives a message, business context, and conversation history,
    then generates an AI response using the configured LLM provider.
    """
    # Validate internal API key
    if x_internal_api_key != settings.ai_orchestrator_api_key:
        logger.warning("Invalid internal API key provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal API key"
        )

    # Validate message
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty"
        )

    try:
        logger.info(f"Processing chat request - message length: {len(request.message)}")
        logger.info(f"Context structureId: {request.context.get('structureId')}")
        logger.info(f"History size: {len(request.history)}")

        # Generate response
        response = await orchestrator.generate_response(
            message=request.message,
            context=request.context,
            history=request.history
        )

        logger.info(f"Response generated successfully - model: {response['model']}")
        return ChatResponse(**response)

    except LLMTimeoutError as e:
        logger.error(f"Timeout LLM : {e}")
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="LLM provider timeout")

    except LLMRateLimitError as e:
        logger.warning(f"Rate limit LLM : {e}")
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="LLM provider rate limited")

    except LLMUnavailableError as e:
        logger.error(f"LLM indisponible : {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM provider unavailable")

    except LLMBadResponseError as e:
        logger.error(f"Réponse LLM invalide : {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM provider returned an invalid response")

    except Exception as e:
        # Filet de sécurité pour tout ce qui n'est pas anticipé (bug applicatif, etc.).
        # Le détail complet est loggé côté serveur ; l'appelant (Spring) ne reçoit
        # qu'un message générique, jamais str(e) — celui-ci peut contenir des chemins
        # internes, des noms de variables, voire des fragments de payload.
        logger.exception("Erreur inattendue lors du traitement de la requête chat")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while processing the chat request"
        )