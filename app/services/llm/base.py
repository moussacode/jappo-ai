from abc import ABC, abstractmethod
from typing import List


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        messages: List[dict]
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System prompt to guide the LLM behavior
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Generated response text
        """
        pass
