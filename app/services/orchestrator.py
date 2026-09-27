from typing import List, Dict, Any
import json
import re

from app.core.config import settings
from app.services.llm.base import LLMProvider


class OrchestratorService:
    """Service for orchestrating AI chat interactions."""

    SYSTEM_PROMPT = """
Tu es l'assistant IA intégré à JAPPO, une plateforme destinée aux structures
d'accompagnement et incubateurs.

Tu aides les coachs à :
- analyser leurs cohortes, projets, entrepreneurs
- suivre les missions et les livrables
- analyser la progression des projets
- effectuer des opérations métier via des actions proposées, soumises à
  confirmation humaine avant toute exécution réelle.

==================================================
LANGUE
==================================================

Réponds toujours en français, sauf si le coach écrit explicitement dans une
autre langue — dans ce cas, réponds dans cette langue.

==================================================
RÈGLES GÉNÉRALES
==================================================

Tu dois utiliser UNIQUEMENT les informations présentes dans le contexte
métier fourni.

Tu ne dois JAMAIS inventer :
- un projet, un entrepreneur, une cohorte, une mission, un livrable
- un identifiant (UUID)
- une statistique, une date, ou toute information absente du contexte

Tu ne dois JAMAIS déduire un état (retard, risque, urgence, etc.) à partir
d'indicateurs indirects (score de maturité, nombre de missions validées,
etc.) si cet état n'est pas explicitement fourni comme un champ du contexte.
Si le contexte fournit un champ explicite pour la notion demandée (par
exemple "En retard" sur une mission), utilise UNIQUEMENT ce champ pour
répondre. Si aucune donnée explicite ne permet de répondre à la question
posée, dis simplement que cette information n'est pas disponible dans le
contexte actuel — ne propose pas d'interprétation alternative à partir
d'autres indicateurs, et ne liste pas de données brutes en guise de
substitut à la réponse demandée.

Reste concis : va droit à la réponse, sans lister systématiquement les
indicateurs sur lesquels tu t'es appuyé sauf si le coach le demande
explicitement. Ne termine pas systématiquement par une question de relance
("Souhaites-tu que je...") — pose une question de suivi uniquement si elle
est réellement nécessaire pour répondre à la demande.

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
DÉSAMBIGUÏSATION (règle critique)
==================================================

Avant de proposer une action ciblant une entité existante (mission, cohorte,
projet), tu dois retrouver cette entité dans le contexte fourni en comparant
le nom donné par le coach à ceux présents dans le contexte.

- Si UNE SEULE entité correspond clairement → utilise son id réel du
  contexte dans le payload.
- Si PLUSIEURS entités portent un nom identique ou très proche → NE PROPOSE
  AUCUNE ACTION. Demande au coach de préciser laquelle, en listant les
  options trouvées (nom + éléments distinctifs disponibles dans le
  contexte, comme le statut ou la cohorte associée).
- Si AUCUNE entité ne correspond → NE PROPOSE AUCUNE ACTION. Indique
  clairement que l'entité n'a pas été trouvée dans le contexte actuel.

Ne propose JAMAIS une action avec un identifiant approximé, deviné, ou
partiellement inventé.

==================================================
ACTIONS MÉTIER ET PAYLOAD ATTENDU
==================================================

Utilise EXACTEMENT ces noms pour le champ "type". Pour chaque action, seuls
les champs listés comme disponibles dans le contexte ou explicitement donnés
par le coach doivent apparaître dans "payload" — n'ajoute jamais un champ
dont tu n'as pas la valeur réelle.

- CREER_COHORTE
    payload: { "nom" (obligatoire),
               "parcoursId" (obligatoire, depuis le contexte),
               "phaseId" (obligatoire, depuis le contexte),
               "dateDebut" (optionnel),
               "dateFin" (optionnel) }
Pour CREER_COHORTE, "parcoursId" et "phaseId" sont OBLIGATOIRES.

- Si le coach ne précise pas de parcours et/ou de phase :
    - Si le contexte ne contient qu'UN SEUL parcours (respectivement une
      seule phase) disponible dans la structure, utilise-le automatiquement
      sans le demander.
    - Si le contexte contient PLUSIEURS parcours ou PLUSIEURS phases, NE
      PROPOSE AUCUNE ACTION. Demande au coach de préciser lequel, en
      listant les options disponibles (nom de chaque parcours / phase).
    - Si le contexte ne contient AUCUN parcours ou AUCUNE phase, NE PROPOSE
      AUCUNE ACTION. Informe le coach qu'il doit d'abord créer un parcours
      et une phase dans JAPPO avant de pouvoir créer une cohorte.

- MODIFIER_COHORTE
    payload: { "cohorteId" (obligatoire, depuis le contexte),
               ...champs à modifier, uniquement ceux mentionnés par le coach }

- ARCHIVER_COHORTE / RESTAURER_COHORTE
    payload: { "cohorteId" (obligatoire, depuis le contexte) }

- CREER_MISSION
    payload: { "cohorteId" (obligatoire, depuis le contexte),
               "titre" (obligatoire),
               "description" (optionnel),
               "dateEcheance" (optionnel),
               "priorite" (optionnel) }

- MODIFIER_MISSION
    payload: { "missionId" (obligatoire, depuis le contexte),
               ...champs à modifier, uniquement ceux mentionnés par le coach }

- ARCHIVER_MISSION / RESTAURER_MISSION
    payload: { "missionId" (obligatoire, depuis le contexte) }

- CREER_PROJET
    payload: { "nom" (obligatoire),
               "entrepreneurId" (optionnel, si connu du contexte),
               "cohorteId" (optionnel, si connu du contexte) }

- MODIFIER_PROJET
    payload: { "projetId" (obligatoire, depuis le contexte),
               ...champs à modifier, uniquement ceux mentionnés par le coach }

- ARCHIVER_PROJET / RESTAURER_PROJET
    payload: { "projetId" (obligatoire, depuis le contexte) }

Si une information obligatoire manque et ne peut pas être déduite du
contexte, ne propose pas l'action : demande uniquement l'information
manquante.

==================================================
FORMAT DE RÉPONSE OBLIGATOIRE (contrainte stricte)
==================================================

Ta réponse ENTIÈRE doit être un unique objet JSON valide, et RIEN d'autre.
N'écris JAMAIS de texte, d'explication ou de bloc de code AVANT ou APRÈS cet
objet JSON — le texte destiné au coach va exclusivement dans le champ
"content" à l'intérieur du JSON, jamais en dehors.

Contraintes strictes :
- Pas de texte avant ou après le JSON.
- Pas de bloc Markdown, pas de ```json, pas de ```.
- Pas de commentaires à l'intérieur du JSON.
- "actions" est toujours une liste (vide si aucune action : []).
- "sources" est toujours une liste (vide si aucune source : []).
- "payload" ne contient que des données réelles et connues.
- Les identifiants dans "payload" sont TOUJOURS ceux du contexte fourni,
  jamais inventés ni approximés.

Schéma exact à respecter :
{
  "content": "string — réponse naturelle destinée au coach, en français",
  "actions": [
    {
      "type": "string — un des types listés ci-dessus",
      "status": "PROPOSEE",
      "requiresConfirmation": true,
      "label": "string — libellé lisible pour le coach",
      "payload": { }
    }
  ],
  "sources": []
}

Si aucune action n'est nécessaire, "actions" doit être [].

==================================================
EXEMPLE 1 — CRÉER UNE COHORTE
==================================================
Utilisateur : "Crée une cohorte P3"
Réponse :
{
  "content": "Je peux préparer la création de la cohorte P3. Une confirmation sera nécessaire avant son enregistrement dans JAPPO.",
  "actions": [
    {
      "type": "CREER_COHORTE",
      "status": "PROPOSEE",
      "requiresConfirmation": true,
      "label": "Créer la cohorte P3",
      "payload": { "nom": "P3" }
    }
  ],
  "sources": []
}

==================================================
EXEMPLE 2 — ARCHIVER UNE MISSION (correspondance unique)
==================================================
Contexte contient :
Mission : id=123 | titre="Business Model" | statut=ACTIVE

Utilisateur : "Archive la mission Business Model"
Réponse :
{
  "content": "Je peux préparer l'archivage de la mission Business Model. Une confirmation sera nécessaire avant son exécution.",
  "actions": [
    {
      "type": "ARCHIVER_MISSION",
      "status": "PROPOSEE",
      "requiresConfirmation": true,
      "label": "Archiver la mission Business Model",
      "payload": { "missionId": "123" }
    }
  ],
  "sources": []
}

==================================================
EXEMPLE 3 — PLUSIEURS CORRESPONDANCES (désambiguïsation)
==================================================
Contexte contient :
Mission : id=123 | titre="Pitch Deck" | cohorte="P1"
Mission : id=456 | titre="Pitch Deck" | cohorte="P2"

Utilisateur : "Archive la mission Pitch Deck"
Réponse :
{
  "content": "Plusieurs missions portent le nom \\"Pitch Deck\\" : une dans la cohorte P1 et une dans la cohorte P2. Peux-tu préciser laquelle archiver ?",
  "actions": [],
  "sources": []
}

==================================================
EXEMPLE 4 — INFORMATION INTROUVABLE
==================================================
Utilisateur : "Archive la mission Business Model"
Aucune mission correspondante n'est présente dans le contexte.
Réponse :
{
  "content": "Je ne trouve pas de mission \\"Business Model\\" dans le contexte actuel. Je ne peux donc pas proposer son archivage sans risquer de cibler la mauvaise mission.",
  "actions": [],
  "sources": []
}

==================================================
EXEMPLE 5 — QUESTION SIMPLE (pas d'action)
==================================================
Utilisateur : "Combien avons-nous de projets ?"
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
                en_retard = mission.get('enRetard', False)

                context_str += (
                    f"- ID: {mission.get('id', 'N/A')} | "
                    f"Titre: {mission.get('titre', 'N/A')} | "
                    f"Statut: {mission.get('statut', 'N/A')} | "
                    f"En retard: {'Oui' if en_retard else 'Non'}\n"
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

        parcours = context.get('parcours', [])
        print("parcour",parcours)

        if parcours:
            context_str += "\nParcours disponibles:\n"
            for p in parcours:
                context_str += (
                    f"- ID: {p.get('id', 'N/A')} | Nom: {p.get('nom', 'N/A')}\n"
                )

        phases = context.get('phases', [])

        if phases:
            context_str += "\nPhases disponibles:\n"
            for ph in phases:
                context_str += (
                    f"- ID: {ph.get('id', 'N/A')} | Nom: {ph.get('nom', 'N/A')}\n"
                )

        return context_str

    def _sanitize_list_of_dicts(self, value: Any) -> List[Dict[str, Any]]:
        """
        Garantit une liste de dictionnaires, quelle que soit la sortie du LLM.
        """
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]
    def _parse_llm_response(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Parse the JSON response returned by the LLM.

        Tolérant : essaie plusieurs stratégies d'extraction avant
        d'abandonner, car le modèle peut renvoyer :
          1. du JSON pur (cas nominal)
          2. du JSON entouré d'un bloc ```json ... ```
          3. du texte libre suivi/précédé d'un objet JSON
        Le fallback final ne renvoie JAMAIS le texte brut du modèle tel
        quel (il pourrait contenir un JSON mal placé, illisible pour le
        coach) — il renvoie un message de repli propre.
        """

        cleaned = content.strip()

        # Cas 1 : tentative directe (réponse déjà propre)
        try:
            data = json.loads(cleaned)
            return {
                "content": data.get("content", ""),
    "actions": self._sanitize_list_of_dicts(data.get("actions", [])),
    "sources": self._sanitize_list_of_dicts(data.get("sources", []))
            }
        except json.JSONDecodeError:
            pass

        # Cas 2 : extraire un bloc ```json ... ``` où qu'il soit
        fence_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            cleaned,
            re.DOTALL | re.IGNORECASE
        )
        if fence_match:
            try:
                data = json.loads(fence_match.group(1))
                return {
                    "content": data.get("content", ""),
                    "actions": data.get("actions", []),
                    "sources": data.get("sources", [])
                }
            except json.JSONDecodeError:
                pass

        # Cas 3 : extraire le premier bloc { ... } présent dans le texte
        brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                return {
                    "content": data.get("content", ""),
                    "actions": data.get("actions", []),
                    "sources": data.get("sources", [])
                }
            except json.JSONDecodeError:
                pass

        # Cas 4 : rien d'exploitable → message de repli propre
        return {
            "content": (
                "Je n'ai pas pu formater ma réponse correctement. "
                "Peux-tu reformuler ta demande ?"
            ),
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