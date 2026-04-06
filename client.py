"""OpenEnv client wrapper for TriageSieve-OpenEnv.

Implements TriageSieveEnv(EnvClient) per CLAUDE.md §5.3.
Provides the thin client that connects to the TriageSieve environment server.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_client import EnvClient, StepResult

from .models import (
    TriageSieveAction,
    TriageSieveObservation,
    TriageSieveState,
)

__all__ = ["TriageSieveEnv"]


class TriageSieveEnv(
    EnvClient[TriageSieveAction, TriageSieveObservation, TriageSieveState]
):
    """Async client for the TriageSieve-OpenEnv environment server."""

    def _step_payload(self, action: TriageSieveAction) -> Dict[str, Any]:
        """Convert a TriageSieveAction to the JSON payload for the server.

        Uses exclude_unset (not exclude_none) to preserve explicitly-set None
        values while omitting fields the caller didn't provide.
        """
        return action.model_dump(exclude_unset=True, mode="json")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TriageSieveObservation]:
        """Parse a server response into a StepResult with TriageSieveObservation."""
        obs_data = payload["observation"]
        observation = TriageSieveObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TriageSieveState:
        """Parse a server state response into a TriageSieveState."""
        return TriageSieveState(**payload)
