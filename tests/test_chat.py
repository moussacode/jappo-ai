import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from app.main import app


client = TestClient(app)


def test_health_check():
    """Test 1: GET /health → 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_with_mock_llm():
    """Test 2: Mock du LLM - POST /api/v1/chat → réponse correcte."""
    # Mock the LLM provider
    with patch('app.services.orchestrator.OpenAIProvider') as mock_provider_class:
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value="Test response from mock LLM")
        mock_provider_class.return_value = mock_provider

        request_data = {
            "message": "Test question",
            "context": {
                "structureId": "test-structure-id",
                "structureNom": "Test Structure",
                "contextGlobal": True,
                "nombreTotalEntrepreneurs": 10,
                "nombreTotalProjets": 5,
                "nombreTotalCohortes": 2,
                "livrablesEnAttente": 3,
                "cohortes": [],
                "projets": []
            },
            "history": []
        }

        response = client.post(
            "/api/v1/chat",
            json=request_data,
            headers={"X-Internal-Api-Key": "default-secret-key"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert data["content"] == "Test response from mock LLM"
        assert data["model"] == "gpt-4o"
        assert data["sources"] == []
        assert data["actions"] == []


def test_chat_empty_message():
    """Test 3: Message vide → erreur 400."""
    request_data = {
        "message": "",
        "context": {},
        "history": []
    }

    response = client.post(
        "/api/v1/chat",
        json=request_data,
        headers={"X-Internal-Api-Key": "default-secret-key"}
    )

    assert response.status_code == 400
    assert "detail" in response.json()


def test_chat_invalid_api_key():
    """Test 5: Vérification de la clé interne - Sans clé → 401."""
    request_data = {
        "message": "Test question",
        "context": {},
        "history": []
    }

    response = client.post(
        "/api/v1/chat",
        json=request_data,
        headers={"X-Internal-Api-Key": "wrong-key"}
    )

    assert response.status_code == 401


def test_chat_no_api_key():
    """Test 5: Vérification de la clé interne - Sans clé → 401."""
    request_data = {
        "message": "Test question",
        "context": {},
        "history": []
    }

    response = client.post(
        "/api/v1/chat",
        json=request_data
    )

    assert response.status_code == 401


def test_chat_history_limit():
    """Test 6: Historique supérieur à la limite → seulement les N derniers messages."""
    with patch('app.services.orchestrator.OpenAIProvider') as mock_provider_class:
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value="Test response")
        mock_provider_class.return_value = mock_provider

        # Create history with 25 messages (more than MAX_HISTORY_MESSAGES=20)
        history = [
            {"auteur": "COACH", "contenu": f"Message {i}"}
            for i in range(25)
        ]

        request_data = {
            "message": "Current question",
            "context": {
                "structureId": "test-id",
                "structureNom": "Test",
                "contextGlobal": True,
                "nombreTotalEntrepreneurs": 10,
                "nombreTotalProjets": 5,
                "nombreTotalCohortes": 2,
                "livrablesEnAttente": 3,
                "cohortes": [],
                "projets": []
            },
            "history": history
        }

        response = client.post(
            "/api/v1/chat",
            json=request_data,
            headers={"X-Internal-Api-Key": "default-secret-key"}
        )

        assert response.status_code == 200

        # Verify that only the last 20 messages were used
        # This is verified by checking the mock was called with limited history
        mock_provider.generate.assert_called_once()
        call_args = mock_provider.generate.call_args
        messages_arg = call_args[0][1]  # Second argument is messages list

        # Should have system prompt + 20 most recent messages + current message
        assert len(messages_arg) == 21  # 20 history + 1 current message


def test_chat_llm_unavailable():
    """Test 4: LLM indisponible → erreur contrôlée."""
    with patch('app.services.orchestrator.OpenAIProvider') as mock_provider_class:
        mock_provider = MagicMock()
        # Simulate LLM error
        mock_provider.generate = AsyncMock(side_effect=Exception("LLM service unavailable"))
        mock_provider_class.return_value = mock_provider

        request_data = {
            "message": "Test question",
            "context": {
                "structureId": "test-id",
                "structureNom": "Test",
                "contextGlobal": True,
                "nombreTotalEntrepreneurs": 10,
                "nombreTotalProjets": 5,
                "nombreTotalCohortes": 2,
                "livrablesEnAttente": 3,
                "cohortes": [],
                "projets": []
            },
            "history": []
        }

        response = client.post(
            "/api/v1/chat",
            json=request_data,
            headers={"X-Internal-Api-Key": "default-secret-key"}
        )

        # Should return 500 with error message
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Error processing request" in data["detail"]
