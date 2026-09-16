import logging
import httpx
from typing import List
from app.core.config import settings
from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """
    Erreur métier générique du provider LLM — jamais de détail d'infrastructure
    (URL, nom de fournisseur, clé) dans le message : celui-ci peut potentiellement
    remonter jusqu'à l'utilisateur final via Spring. Le détail technique réel est
    toujours loggé séparément côté serveur (voir chaque levée ci-dessous).
    """


class LLMTimeoutError(LLMProviderError):
    """Le fournisseur LLM n'a pas répondu dans le délai imparti."""


class LLMUnavailableError(LLMProviderError):
    """Le fournisseur LLM est injoignable (réseau, DNS, connexion refusée)."""


class LLMRateLimitError(LLMProviderError):
    """Le fournisseur LLM a renvoyé 429 (quota atteint)."""


class LLMBadResponseError(LLMProviderError):
    """Le fournisseur LLM a renvoyé une erreur HTTP (401, 404 modèle retiré, 500...) ou un payload inattendu."""


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible LLM provider."""

    def __init__(self):
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.base_url = settings.llm_base_url
        self.timeout = 60.0  # seconds

    async def generate(
        self,
        system_prompt: str,
        messages: List[dict]
    ) -> str:
        """
        Generate a response using OpenAI-compatible API.

        Args:
            system_prompt: System prompt to guide the LLM behavior
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Generated response text

        Raises:
            LLMTimeoutError, LLMUnavailableError, LLMRateLimitError, LLMBadResponseError:
            toujours avec un message sans détail d'infrastructure — voir logs pour la cause exacte.
        """
        if not self.api_key:
            # Erreur de configuration serveur, jamais censée arriver en usage normal :
            # on la garde distincte pour qu'elle ne soit jamais confondue avec une panne réseau.
            logger.error("LLM_API_KEY is not configured")
            raise LLMBadResponseError("Configuration du service IA incomplète")

        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": full_messages,
                        "temperature": 0.7,
                        "max_tokens": 2000
                    }
                )
                response.raise_for_status()
                data = response.json()

                # OpenRouter peut retourner HTTP 200 tout en encapsulant
                # une erreur du fournisseur dans le payload.
                if isinstance(data, dict) and data.get("error"):
                    error = data["error"]

                    logger.error(
                        "Erreur fournisseur LLM : %s",
                        error
                    )

                    error_type = error.get("metadata", {}).get("error_type")

                    if error_type == "provider_unavailable":
                        raise LLMUnavailableError(
                            "Le service IA est temporairement indisponible"
                        )

                    raise LLMBadResponseError(
                        "Le fournisseur de modèle a rencontré une erreur"
                    )

                choices = data.get("choices")

                if not choices:
                    logger.error(
                        "Payload LLM sans 'choices'. Payload reçu : %s",
                        str(data)[:1000]
                    )
                    raise LLMBadResponseError(
                        "Réponse du fournisseur de modèle illisible"
                    )

                first_choice = choices[0]

                if not isinstance(first_choice, dict):
                    raise LLMBadResponseError(
                        "Réponse du fournisseur de modèle illisible"
                    )

                message = first_choice.get("message")

                if not isinstance(message, dict):
                    raise LLMBadResponseError(
                        "Réponse du fournisseur de modèle illisible"
                    )

                content = message.get("content")

                if not content:
                    raise LLMBadResponseError(
                        "Réponse du fournisseur de modèle vide"
                    )

                return content

        except httpx.TimeoutException:
            logger.error(f"Timeout ({self.timeout}s) en appelant le fournisseur LLM (modèle={self.model})")
            raise LLMTimeoutError("Le fournisseur de modèle n'a pas répondu à temps") from None

        except httpx.ConnectError as e:
            # Pas de f"{e}" ici : le message d'httpx inclut parfois l'hôte cible.
            logger.error(f"Connexion au fournisseur LLM impossible : {e}")
            raise LLMUnavailableError("Le fournisseur de modèle est injoignable") from None

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            # Le corps de la réponse (souvent utile en dev) reste UNIQUEMENT dans le log serveur.
            logger.error(f"Le fournisseur LLM a répondu {status} pour le modèle '{self.model}' : {e.response.text[:500]}")
            if status == 429:
                raise LLMRateLimitError("Quota atteint chez le fournisseur de modèle") from None
            if status == 404:
                # Cas réel rencontré : identifiant de modèle retiré du catalogue gratuit.
                raise LLMBadResponseError(
                    f"Le modèle configuré ('{self.model}') n'est plus disponible chez le fournisseur"
                ) from None
            raise LLMBadResponseError("Le fournisseur de modèle a renvoyé une erreur") from None

        except (KeyError, IndexError, ValueError) as e:
            # Réponse 200 mais structure JSON inattendue (ex : le fournisseur change son format un jour).
            logger.error(f"Réponse du fournisseur LLM dans un format inattendu : {e}")
            raise LLMBadResponseError("Réponse du fournisseur de modèle illisible") from None