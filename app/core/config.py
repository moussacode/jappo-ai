from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # FastAPI Configuration
    fastapi_env: str = "development"
    fastapi_port: int = 8000
    fastapi_host: str = "0.0.0.0"

    # Security
    ai_orchestrator_api_key: str

    # LLM Provider Configuration
    llm_api_key: str
    llm_model: str = "openrouter/free"
    llm_base_url: str = "https://openrouter.ai/api/v1"

    # History Configuration
    max_history_messages: int = 20

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()