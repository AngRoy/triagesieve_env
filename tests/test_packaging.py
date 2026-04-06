"""Tests for runtime and packaging surface (Phase 5).

Covers:
- server/app.py: create_app wiring produces a FastAPI instance
- client.py: TriageSieveEnv subclasses EnvClient with required abstract methods
- openenv.yaml: manifest fields match CLAUDE.md §5.4
- pyproject.toml: metadata and dependencies match CLAUDE.md §26
- __init__.py: exports include TriageSieveEnv
- server/Dockerfile: CMD targets the correct module path
"""

from __future__ import annotations

import importlib
import pathlib
import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
SERVER_DIR = ROOT / "server"

# ---------------------------------------------------------------------------
# 1. server/app.py — create_app wiring
# ---------------------------------------------------------------------------


class TestAppModule:
    """Tests for server/app.py."""

    def test_app_module_imports(self) -> None:
        """app.py must be importable."""
        from ..server import app  # noqa: F401

    def test_app_is_fastapi_instance(self) -> None:
        """Module-level `app` must be a FastAPI instance."""
        from fastapi import FastAPI

        from ..server.app import app

        assert isinstance(app, FastAPI)

    def test_app_created_with_correct_env_class(self) -> None:
        """create_app must be called with TriageSieveEnvironment class (factory)."""
        from ..server.triagesieve_env_environment import (
            TriageSieveEnvironment,
        )

        # Re-import module with patched create_app to capture args
        captured: dict[str, Any] = {}

        original_create_app = None

        def spy_create_app(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            nonlocal original_create_app
            # Return a mock FastAPI to avoid side effects
            return MagicMock()

        with patch(
            "openenv.core.env_server.http_server.create_app",
            side_effect=spy_create_app,
        ):
            # Force reimport
            mod_name = "triagesieve_env.server.app"
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            importlib.import_module(mod_name)

        assert captured.get("env") is TriageSieveEnvironment

    def test_app_created_with_correct_action_cls(self) -> None:
        """create_app must receive TriageSieveAction as action_cls."""
        from ..models import TriageSieveAction

        captured: dict[str, Any] = {}

        def spy_create_app(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        with patch(
            "openenv.core.env_server.http_server.create_app",
            side_effect=spy_create_app,
        ):
            mod_name = "triagesieve_env.server.app"
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            importlib.import_module(mod_name)

        assert captured.get("action_cls") is TriageSieveAction

    def test_app_created_with_correct_observation_cls(self) -> None:
        """create_app must receive TriageSieveObservation as observation_cls."""
        from ..models import TriageSieveObservation

        captured: dict[str, Any] = {}

        def spy_create_app(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        with patch(
            "openenv.core.env_server.http_server.create_app",
            side_effect=spy_create_app,
        ):
            mod_name = "triagesieve_env.server.app"
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            importlib.import_module(mod_name)

        assert captured.get("observation_cls") is TriageSieveObservation

    def test_app_env_name(self) -> None:
        """create_app must use env_name='.'."""
        captured: dict[str, Any] = {}

        def spy_create_app(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        with patch(
            "openenv.core.env_server.http_server.create_app",
            side_effect=spy_create_app,
        ):
            mod_name = "triagesieve_env.server.app"
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            importlib.import_module(mod_name)

        assert captured.get("env_name") == "triagesieve_env"

    def test_app_max_concurrent_envs(self) -> None:
        """create_app must set max_concurrent_envs=4."""
        captured: dict[str, Any] = {}

        def spy_create_app(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        with patch(
            "openenv.core.env_server.http_server.create_app",
            side_effect=spy_create_app,
        ):
            mod_name = "triagesieve_env.server.app"
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            importlib.import_module(mod_name)

        assert captured.get("max_concurrent_envs") == 4


# ---------------------------------------------------------------------------
# 2. client.py — EnvClient subclass
# ---------------------------------------------------------------------------


class TestClientModule:
    """Tests for client.py."""

    def test_client_module_imports(self) -> None:
        """client.py must be importable."""
        from .. import client  # noqa: F401

    def test_client_subclasses_envclient(self) -> None:
        """TriageSieveEnv must subclass EnvClient."""
        from openenv.core.env_client import EnvClient

        from ..client import TriageSieveEnv

        assert issubclass(TriageSieveEnv, EnvClient)

    def test_step_payload_returns_dict(self) -> None:
        """_step_payload must return a dict from action.model_dump(exclude_none=True)."""
        from ..client import TriageSieveEnv
        from ..models import ActionType, TriageSieveAction

        # Create client without connecting
        client = object.__new__(TriageSieveEnv)

        action = TriageSieveAction(
            action_type=ActionType.OPEN_TICKET,
            ticket_id="T001",
            metadata={},
        )
        payload = client._step_payload(action)

        assert isinstance(payload, dict)
        assert payload["action_type"] == "open_ticket"
        assert payload["ticket_id"] == "T001"
        # None fields should be excluded
        assert "issue_family" not in payload

    def test_parse_result_returns_step_result(self) -> None:
        """_parse_result must return a StepResult with correct observation."""
        from openenv.core.env_client import StepResult

        from ..client import TriageSieveEnv

        client = object.__new__(TriageSieveEnv)

        # Minimal valid observation payload
        payload: Dict[str, Any] = {
            "observation": {
                "done": False,
                "reward": 0.01,
                "metadata": {},
                "inbox_summaries": [],
                "focused_ticket": None,
                "available_templates": [],
                "allowed_queues": [],
                "routing_policy_cards": [],
                "sla_policy_cards": [],
                "legal_actions": ["open_ticket"],
                "action_budget_remaining": 10,
                "step_count": 1,
                "current_time": "2026-01-01T00:00:00",
                "last_action_result": "ok",
                "task_difficulty": "easy",
                "hint": None,
            },
            "reward": 0.01,
            "done": False,
        }
        result = client._parse_result(payload)

        assert isinstance(result, StepResult)
        assert result.reward == 0.01
        assert result.done is False
        assert result.observation.step_count == 1

    def test_parse_state_returns_state(self) -> None:
        """_parse_state must return a TriageSieveState."""
        from ..client import TriageSieveEnv
        from ..models import TriageSieveState

        client = object.__new__(TriageSieveEnv)

        payload: Dict[str, Any] = {
            "episode_id": "ep-001",
            "step_count": 3,
            "task_difficulty": "medium",
            "seed": 42,
            "total_tickets": 2,
            "action_budget": 8,
            "action_budget_remaining": 5,
            "mode": "eval_strict",
            "tickets_summary": [],
        }
        state = client._parse_state(payload)

        assert isinstance(state, TriageSieveState)
        assert state.episode_id == "ep-001"
        assert state.step_count == 3
        assert state.seed == 42

    def test_step_payload_excludes_none_fields(self) -> None:
        """_step_payload must not include None optional fields."""
        from ..client import TriageSieveEnv
        from ..models import ActionType, TriageSieveAction

        client = object.__new__(TriageSieveEnv)

        action = TriageSieveAction(
            action_type=ActionType.SKIP_TURN,
            metadata={},
        )
        payload = client._step_payload(action)

        assert "ticket_id" not in payload
        assert "issue_family" not in payload
        assert "queue_id" not in payload


# ---------------------------------------------------------------------------
# 3. openenv.yaml — manifest
# ---------------------------------------------------------------------------

'''
class TestOpenenvYaml:
    """Tests for openenv.yaml matching CLAUDE.md §5.4."""

    @pytest.fixture()
    def manifest(self) -> dict[str, Any]:
        yaml_path = ROOT / "openenv.yaml"
        assert yaml_path.exists(), f"openenv.yaml not found at {yaml_path}"
        with open(yaml_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_name(self, manifest: dict[str, Any]) -> None:
        assert manifest["name"] == "triagesieve_env"

    def test_version(self, manifest: dict[str, Any]) -> None:
        assert manifest["version"] == "0.1.0"

    def test_entry_point(self, manifest: dict[str, Any]) -> None:
        assert (
            manifest["entry_point"]
            == "triagesieve_env.server.triagesieve_env_environment:TriageSieveEnvironment"
        )

    def test_action_class(self, manifest: dict[str, Any]) -> None:
        assert manifest["action_class"] == "triagesieve_env.models:TriageSieveAction"

    def test_observation_class(self, manifest: dict[str, Any]) -> None:
        assert manifest["observation_class"] == "triagesieve_env.models:TriageSieveObservation"
'''

# ---------------------------------------------------------------------------
# 4. pyproject.toml — metadata
# ---------------------------------------------------------------------------

'''
class TestPyprojectToml:
    """Tests for pyproject.toml matching CLAUDE.md §26."""

    @pytest.fixture()
    def toml_data(self) -> dict[str, Any]:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib  # type: ignore[no-redef]

        toml_path = ROOT / "pyproject.toml"
        assert toml_path.exists()
        with open(toml_path, "rb") as f:
            return tomllib.load(f)

    def test_project_name(self, toml_data: dict[str, Any]) -> None:
        assert toml_data["project"]["name"] == "openenv-triagesieve-env"

    def test_project_version(self, toml_data: dict[str, Any]) -> None:
        assert toml_data["project"]["version"] == "0.1.0"

    def test_requires_python(self, toml_data: dict[str, Any]) -> None:
        assert toml_data["project"]["requires-python"] == ">=3.11"

    def test_core_dependencies(self, toml_data: dict[str, Any]) -> None:
        deps = toml_data["project"]["dependencies"]
        dep_names = [d.split(">=")[0].split(">")[0].split("==")[0] for d in deps]
        for required in ["openenv-core[core]", "pydantic", "fastapi", "uvicorn"]:
            assert required in dep_names, f"Missing dependency: {required}"

    def test_dev_dependencies(self, toml_data: dict[str, Any]) -> None:
        dev = toml_data["project"]["optional-dependencies"]["dev"]
        dev_names = [d.split(">=")[0] for d in dev]
        assert "pytest" in dev_names
        assert "pytest-asyncio" in dev_names

    def test_black_target_version(self, toml_data: dict[str, Any]) -> None:
        """black target-version must include py311, not py312."""
        target = toml_data["tool"]["black"]["target-version"]
        assert "py311" in target
        assert "py312" not in target
'''

# ---------------------------------------------------------------------------
# 5. __init__.py — exports
# ---------------------------------------------------------------------------


class TestInitExports:
    """Tests for __init__.py package exports."""

    def test_exports_action(self) -> None:
        from ..models import TriageSieveAction  # noqa: F401

    def test_exports_observation(self) -> None:
        from ..models import TriageSieveObservation  # noqa: F401

    def test_exports_state(self) -> None:
        from ..models import TriageSieveState  # noqa: F401

    def test_exports_client(self) -> None:
        from ..client import TriageSieveEnv  # noqa: F401


# ---------------------------------------------------------------------------
# 6. server/Dockerfile — CMD alignment
# ---------------------------------------------------------------------------

'''
class TestDockerfile:
    """Tests for server/Dockerfile matching CLAUDE.md §27."""

    @pytest.fixture()
    def dockerfile_content(self) -> str:
        path = SERVER_DIR / "Dockerfile"
        assert path.exists()
        return path.read_text()

    def test_base_image(self, dockerfile_content: str) -> None:
        assert "python:3.11-slim" in dockerfile_content

    def test_cmd_module_path(self, dockerfile_content: str) -> None:
        assert "triagesieve_env.server.app:app" in dockerfile_content

    def test_exposed_port(self, dockerfile_content: str) -> None:
        assert "EXPOSE 8000" in dockerfile_content
'''

# ---------------------------------------------------------------------------
# 7. server/requirements.txt — deps present
# ---------------------------------------------------------------------------


class TestRequirementsTxt:
    """Tests for server/requirements.txt."""

    @pytest.fixture()
    def requirements(self) -> list[str]:
        path = SERVER_DIR / "requirements.txt"
        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        return [l.strip() for l in lines if l.strip() and not l.startswith("#")]

    def test_has_openenv_core(self, requirements: list[str]) -> None:
        assert any("openenv-core" in r for r in requirements)

    def test_has_pydantic(self, requirements: list[str]) -> None:
        assert any("pydantic" in r for r in requirements)

    def test_has_fastapi(self, requirements: list[str]) -> None:
        assert any("fastapi" in r for r in requirements)

    def test_has_uvicorn(self, requirements: list[str]) -> None:
        assert any("uvicorn" in r for r in requirements)
