"""
Tests for HuggingFace and MBPP web interface integration.

These tests verify the web API endpoints for loading HuggingFace models
and browsing MBPP benchmark tasks.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add webapp to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from webapp.app import create_app, Config


@pytest.fixture
def app():
    """Create test Flask application."""
    config = Config()
    config.UPLOAD_FOLDER = tempfile.mkdtemp()
    app = create_app(config)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestHuggingFaceLoader:
    """Tests for HuggingFaceLoader class."""

    def test_validate_model_id_valid(self, app):
        """Test validation accepts known model IDs."""
        with app.app_context():
            from webapp.app import HuggingFaceLoader
            loader = HuggingFaceLoader(app.config["UPLOAD_FOLDER"])

            with patch("huggingface_hub.model_info") as mock_info:
                mock_info.return_value = MagicMock()
                assert loader.validate_model_id("bert-base-uncased") is True

    def test_validate_model_id_invalid(self, app):
        """Test validation rejects invalid model IDs."""
        with app.app_context():
            from webapp.app import HuggingFaceLoader
            loader = HuggingFaceLoader(app.config["UPLOAD_FOLDER"])

            with patch("huggingface_hub.model_info") as mock_info:
                from huggingface_hub.utils import RepositoryNotFoundError
                mock_info.side_effect = RepositoryNotFoundError("Not found")
                assert loader.validate_model_id("nonexistent-model-xyz") is False

    def test_load_and_export_creates_json(self, app):
        """Test that load_and_export creates a valid JSON file."""
        with app.app_context():
            from webapp.app import HuggingFaceLoader
            loader = HuggingFaceLoader(app.config["UPLOAD_FOLDER"])

            # Mock the transformers import and model loading
            mock_model = MagicMock()
            mock_model.config = MagicMock()
            mock_model.config.hidden_size = 768
            mock_model.config.num_attention_heads = 12
            mock_model.config.num_hidden_layers = 12
            mock_model.config.vocab_size = 30522
            mock_model.config.max_position_embeddings = 512
            mock_model.parameters.return_value = []

            with patch("transformers.AutoModel.from_pretrained", return_value=mock_model):
                with patch("webapp.app.extract_transformer_model") as mock_extract:
                    mock_extract.return_value = {
                        "model_type": "transformer",
                        "name": "test_model",
                        "d_model": 768,
                        "num_heads": 12,
                        "num_layers": 12
                    }

                    result = loader.load_and_export("test-model")

                    assert result["success"] is True
                    assert "json_path" in result
                    assert Path(result["json_path"]).exists()


class TestHuggingFaceAPIEndpoint:
    """Tests for /api/huggingface/load endpoint."""

    def test_load_endpoint_missing_model_id(self, client):
        """Test endpoint returns error when model_id is missing."""
        response = client.post(
            "/api/huggingface/load",
            json={},
            content_type="application/json"
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_load_endpoint_invalid_model(self, client):
        """Test endpoint returns error for invalid model ID."""
        with patch("webapp.app.HuggingFaceLoader.validate_model_id", return_value=False):
            response = client.post(
                "/api/huggingface/load",
                json={"model_id": "nonexistent-model"},
                content_type="application/json"
            )

            assert response.status_code == 404
            data = json.loads(response.data)
            assert data["success"] is False

    def test_load_endpoint_success(self, client):
        """Test endpoint succeeds with valid model ID."""
        mock_result = {
            "success": True,
            "model_name": "bert-base-uncased",
            "model_type": "transformer",
            "parameters": 110000000,
            "json_path": "/tmp/bert-base-uncased.json"
        }

        with patch("webapp.app.HuggingFaceLoader.validate_model_id", return_value=True):
            with patch("webapp.app.HuggingFaceLoader.load_and_export", return_value=mock_result):
                response = client.post(
                    "/api/huggingface/load",
                    json={"model_id": "bert-base-uncased"},
                    content_type="application/json"
                )

                assert response.status_code == 200
                data = json.loads(response.data)
                assert data["success"] is True
                assert data["model_name"] == "bert-base-uncased"


class TestMBPPTaskEndpoints:
    """Tests for MBPP task browsing endpoints."""

    def test_list_tasks_empty(self, client):
        """Test listing tasks when none are cached."""
        with patch("webapp.app.list_available_tasks", return_value=[]):
            response = client.get("/api/mbpp/tasks")

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["tasks"] == []
            assert data["cached"] is False

    def test_list_tasks_with_data(self, client):
        """Test listing tasks when data exists."""
        mock_tasks = ["task_001", "task_002", "task_003"]

        with patch("webapp.app.list_available_tasks", return_value=mock_tasks):
            with patch("webapp.app.load_mbpp_task") as mock_load:
                # Return minimal task info for each
                mock_load.side_effect = [
                    MagicMock(task_id="task_001", description="Desc 1", source="mbpp"),
                    MagicMock(task_id="task_002", description="Desc 2", source="mbpp"),
                    MagicMock(task_id="task_003", description="Desc 3", source="mbpp"),
                ]

                response = client.get("/api/mbpp/tasks")

                assert response.status_code == 200
                data = json.loads(response.data)
                assert len(data["tasks"]) == 3
                assert data["cached"] is True

    def test_get_single_task(self, client):
        """Test getting a single task's details."""
        from benchmarks.verina import MBPPTask

        mock_task = MBPPTask(
            task_id="task_001",
            description="Calculate the average of a list",
            lean_code="def average (lst : List Nat) : Float := ...",
            lean_spec="theorem average_correct : ...",
            lean_proof="by simp [average]",
            test_cases=[{"input": [1, 2, 3], "output": 2.0}],
            source="mbpp"
        )

        with patch("webapp.app.load_mbpp_task", return_value=mock_task):
            response = client.get("/api/mbpp/tasks/task_001")

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["task_id"] == "task_001"
            assert "lean_code" in data
            assert "lean_spec" in data

    def test_get_nonexistent_task(self, client):
        """Test getting a task that doesn't exist."""
        with patch("webapp.app.load_mbpp_task") as mock_load:
            mock_load.side_effect = FileNotFoundError("Task not found")

            response = client.get("/api/mbpp/tasks/nonexistent")

            assert response.status_code == 404

    def test_fetch_dataset_endpoint(self, client):
        """Test the dataset fetch endpoint."""
        with patch("webapp.app.fetch_verina_dataset") as mock_fetch:
            mock_fetch.return_value = [MagicMock() for _ in range(49)]

            response = client.post(
                "/api/mbpp/fetch",
                json={"subset": "mbpp"},
                content_type="application/json"
            )

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["status"] == "success"
            assert data["task_count"] == 49


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_health_endpoint(self, client):
        """Test health check still works after modifications."""
        response = client.get("/health")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"

    def test_index_page_loads(self, client):
        """Test that the main page loads successfully."""
        response = client.get("/")
        assert response.status_code == 200
