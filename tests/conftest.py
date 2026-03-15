import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path to ensure we can import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import engine  # noqa: E402
from main import app  # noqa: E402


@pytest.fixture
def mock_engine_load_unload():
    with (
        patch.object(engine, "load") as mock_load,
        patch.object(engine, "unload") as mock_unload,
    ):
        # We manually set model to indicate loaded so endpoints don't fail,
        # or endpoints use engine.model is not None to check
        def mock_load_side_effect():
            engine.model = MagicMock()

        def mock_unload_side_effect():
            engine.model = None

        mock_load.side_effect = mock_load_side_effect
        mock_unload.side_effect = mock_unload_side_effect
        yield mock_load, mock_unload
        engine.model = None


@pytest.fixture
def client(mock_engine_load_unload):
    with TestClient(app) as c:
        yield c
