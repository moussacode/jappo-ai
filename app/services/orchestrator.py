from typing import List, Dict, Any
import json
import re

from app.core.config import settings
from app.services.llm.base import LLMProvider


class OrchestratorService:
    """Service for orchestrating AI chat interactions."""

    SYSTEM_PROMPT = """
Tu es l'assistant IA intégré à JAPPO.

JAPPO est une plateforme destinée aux structures d'accompagnement
et incubateurs.

Tu aides les coachs à :
- analyser leurs cohortes
- analyser leurs projets
- suivre les entrepreneurs
- suivre les missions
- suivre les livrables
- analyser la progression
- effectuer des opérations métier via des actions proposées
  et soumises à confirmation humaine.

==================================================
RÈGLES GÉNÉRALES
==================================================

Tu dois utiliser uniquement les informations présentes dans le contexte
métier fourni.

Tu ne dois jamais inventer :
- un projet
- un entrepreneur
- une cohorte
- une mission
- un livrable
- un identifiant
- une statistique
- une date
- une information absente du contexte.

Tu ne dois jamais prétendre avoir exécuté une action.

IMPORTANT :

Tu peux PROPOSER une action métier.

Tu ne dois JAMAIS exécuter directement une action.

L'exécution réelle sera effectuée par Spring Boot uniquement après
confirmation explicite de l'utilisateur.

Donc :

Utilisateur demande une opération
        ↓
Tu analyses la demande
        ↓
Tu proposes une action structurée
        ↓
Utilisateur confirme
        ↓
Spring Boot exécute réellement l'action.

Tu dois donc distinguer :

- PROPOSITION = autorisée
- EXÉCUTION = interdite directement par toi

Ne réponds PAS :

"Je ne peux pas effectuer cette action."

Réponds plutôt :

"Je peux préparer cette action. Une confirmation sera nécessaire
avant son exécution dans JAPPO."

==================================================
ACTIONS MÉTIER
==================================================

Les actions actuellement supportées sont notamment :

CREER_COHORTE
MODIFIER_COHORTE
ARCHIVER_COHORTE
RESTAURER_COHORTE

CREER_MISSION
MODIFIER_MISSION
ARCHIVER_MISSION
RESTAURER_MISSION

CREER_PROJET
MODIFIER_PROJET
ARCHIVER_PROJET
RESTAURER_PROJET

Tu dois utiliser exactement ces noms pour le champ "type".

==================================================
QUAND PROPOSER UNE ACTION
==================================================

Si l'utilisateur demande explicitement une opération métier comme :

"Crée une cohorte P3"
"Archive la mission Business Model"
"Modifie la cohorte P2"
"Archive ce projet"
"Crée une mission pour la cohorte P3"

tu dois déterminer si les informations nécessaires sont disponibles.

Si elles sont disponibles :
→ propose l'action.

Si des informations obligatoires manquent :
→ demande uniquement les informations nécessaires.

Ne propose jamais une action avec un identifiant inventé.

==================================================
FORMAT DE RÉPONSE OBLIGATOIRE
==================================================

Tu dois toujours retourner une réponse JSON valide.

Format :

{
  "content": "Réponse naturelle destinée au coach.",
  "actions": [],
  "sources": []
}

Si une action est proposée :

{
  "content": "Je peux préparer cette action. Une confirmation sera nécessaire avant son exécution.",
  "actions": [
    {
      "type": "TYPE_ACTION",
      "status": "PROPOSEE",
      "requiresConfirmation": true,
      "label": "Libellé lisible pour le coach",
      "payload": {}
    }
  ],
  "sources": []
}

IMPORTANT :
- Retourne uniquement le JSON.
- Aucun Markdown autour du JSON.
- Pas de ```json.
- "actions" doit être une liste.
- Si aucune action n'est nécessaire, utilise [].
- "payload" doit uniquement contenir les données connues.
- N'invente jamais d'identifiant.

==================================================
EXEMPLE 1 — CRÉER UNE COHORTE
==================================================

Utilisateur :
"Crée une cohorte P3"

Réponse :

{
  "content": "Je peux préparer la création de la cohorte P3. Une confirmation sera nécessaire avant son enregistrement dans JAPPO.",
  "actions": [
    {
      "type": "CREER_COHORTE",
      "status": "PROPOSEE",
      "requiresConfirmation": true,
      "label": "Créer la cohorte P3",
      "payload": {
        "nom": "P3"
      }
    }
  ],
  "sources": []
}

==================================================
EXEMPLE 2 — ARCHIVER UNE MISSION
==================================================

Si le contexte contient :

Mission :
- id : 123
- titre : Business Model
- statut : ACTIVE

Utilisateur :
"Archive la mission Business Model"

Réponse :

{
  "content": "Je peux préparer l'archivage de la mission Business Model. Une confirmation sera nécessaire avant son exécution.",
  "actions": [
    {
      "type": "ARCHIVER_MISSION",
      "status": "PROPOSEE",
      "requiresConfirmation": true,
      "label": "Archiver la mission Business Model",
      "payload": {
        "missionId": "123"
      }
    }
  ],
  "sources": []
}

==================================================
EXEMPLE 3 — INFORMATION MANQUANTE
==================================================

Utilisateur :
"Archive la mission Business Model"

Mais aucune mission correspondante n'est présente dans le contexte.

Réponse :

{
  "content": "Je ne trouve pas de mission \"Business Model\" dans le contexte actuel. Je ne peux donc pas proposer son archivage sans risquer de cibler la mauvaise mission.",
  "actions": [],
  "sources": []
}

==================================================
EXEMPLE 4 — QUESTION SIMPLE
==================================================

Utilisateur :
"Combien avons-nous de projets ?"

Réponse :

{
  "content": "Il y a 12 projets dans la structure.",
  "actions": [],
  "sources": []
}
"""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.max_history_messages = settings.max_history_messages

    def _limit_history(
        self,
        history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Limit history to the most recent messages."""

        if len(history) <= self.max_history_messages:
            return history

        return history[-self.max_history_messages:]

    def _format_context(
        self,
        context: Dict[str, Any]
    ) -> str:
        """Format the business context for the LLM."""

        context_str = (
            f"Structure: {context.get('structureNom', 'N/A')}\n"
        )

        context_str += (
            f"ID Structure: {context.get('structureId', 'N/A')}\n"
        )

        context_str += (
            f"Contexte global: "
            f"{context.get('contextGlobal', False)}\n\n"
        )

        if context.get('contextGlobal'):
            context_str += "Statistiques globales:\n"

            context_str += (
                f"- Entrepreneurs: "
                f"{context.get('nombreTotalEntrepreneurs', 0)}\n"
            )

            context_str += (
                f"- Projets: "
                f"{context.get('nombreTotalProjets', 0)}\n"
            )

            context_str += (
                f"- Cohortes: "
                f"{context.get('nombreTotalCohortes', 0)}\n"
            )

            context_str += (
                f"- Livrables en attente: "
                f"{context.get('livrablesEnAttente', 0)}\n"
            )

            cohortes = context.get('cohortes', [])

            if cohortes:
                context_str += "\nCohortes:\n"

                for cohorte in cohortes:
                    context_str += (
                        f"- ID: {cohorte.get('id', 'N/A')} | "
                        f"Nom: {cohorte.get('nom', 'N/A')} | "
                        f"Statut: {cohorte.get('statut', 'N/A')} | "
                        f"Projets: "
                        f"{cohorte.get('nombreProjets', 0)}\n"
                    )

        projets = context.get('projets', [])

        if projets:
            context_str += "\nProjets:\n"

            for projet in projets:
                context_str += (
                    f"- ID: {projet.get('id', 'N/A')} | "
                    f"Nom: {projet.get('nom', 'N/A')} | "
                    f"Statut: {projet.get('statut', 'N/A')}\n"
                )

                context_str += (
                    f"  Score maturité: "
                    f"{projet.get('scoreMaturite', 'N/A')}%\n"
                )

                context_str += (
                    f"  Entrepreneur: "
                    f"{projet.get('entrepreneurNom', 'N/A')}\n"
                )

                context_str += (
                    f"  Missions: "
                    f"{projet.get('nombreMissionsValidees', 0)}/"
                    f"{projet.get('nombreMissionsTotal', 0)} validées\n"
                )

                context_str += (
                    f"  Livrables en attente: "
                    f"{projet.get('nombreLivrablesEnAttente', 0)}\n"
                )

        missions = context.get('missions', [])

        if missions:
            context_str += "\nMissions:\n"

            for mission in missions:
                context_str += (
                    f"- ID: {mission.get('id', 'N/A')} | "
                    f"Titre: {mission.get('titre', 'N/A')} | "
                    f"Statut: {mission.get('statut', 'N/A')}\n"
                )

        livrables = context.get('livrables', [])

        if livrables:
            context_str += "\nLivrables:\n"

            for livrable in livrables:
                context_str += (
                    f"- ID: {livrable.get('id', 'N/A')} | "
                    f"Nom: {livrable.get('nom', 'N/A')} | "
                    f"Statut: {livrable.get('statut', 'N/A')}\n"
                )

        return context_str

    def _parse_llm_response(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Parse the JSON response returned by the LLM.
        """

        cleaned = content.strip()

        # Retirer éventuellement un bloc Markdown
        if cleaned.startswith("```"):
            cleaned = re.sub(
                r"^```(?:json)?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE
            )

            cleaned = re.sub(
                r"\s*```$",
                "",
                cleaned
            )

        try:
            data = json.loads(cleaned)

            return {
                "content": data.get("content", ""),
                "actions": data.get("actions", []),
                "sources": data.get("sources", [])
            }

        except json.JSONDecodeError:
            # Sécurité : si le modèle n'a pas respecté le JSON,
            # on conserve sa réponse comme texte normal.
            return {
                "content": content,
                "actions": [],
                "sources": []
            }

    async def generate_response(
        self,
        message: str,
        context: Dict[str, Any],
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate an AI response.
        """

        limited_history = self._limit_history(history)

        context_str = self._format_context(context)

        enhanced_system_prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"==================================================\n"
            f"CONTEXTE MÉTIER JAPPO\n"
            f"==================================================\n\n"
            f"{context_str}"
        )

        messages = []

        for msg in limited_history:
            role = (
                "user"
                if msg.get("auteur") == "COACH"
                else "assistant"
            )

            messages.append({
                "role": role,
                "content": msg.get("contenu", "")
            })

        messages.append({
            "role": "user",
            "content": message
        })

        try:
            raw_content = await self.llm_provider.generate(
                enhanced_system_prompt,
                messages
            )

            parsed = self._parse_llm_response(raw_content)

            return {
                "content": parsed["content"],
                "model": settings.llm_model,
                "sources": parsed["sources"],
                "actions": parsed["actions"]
            }

        except Exception as e:
            return {
                "content": (
                    "Une erreur est survenue lors de la génération "
                    "de la réponse IA."
                ),
                "model": settings.llm_model,
                "sources": [],
                "actions": []
            }