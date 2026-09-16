from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    content: str
    model: str
    success: bool = True
    sources: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str