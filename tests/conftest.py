import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path to ensure we can import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_sentence_transformer():
    with patch("main.SentenceTransformer") as mock:
        # Creating a mock instance that will be returned when initialized
        mock_instance = MagicMock()

        # Setup default behaviors for the mock instance
        # Mocking encode to return a dummy vector
        mock_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        # When SentenceTransformer(...) is called, return this mock_instance
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_trainer():
    with patch("main.SentenceTransformerTrainer") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance
