"""Configuration pytest : rend api.py et le dashboard importables."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "dashboard"))


@pytest.fixture(scope="session")
def client():
    """Client HTTP branché sur l'application, avec le vrai modèle chargé."""
    import api

    return TestClient(api.app)
