"""FastAPI entrypoint via OpenEnv create_app().

Wires TriageSieveEnvironment, TriageSieveAction, and TriageSieveObservation
into the OpenEnv HTTP server per CLAUDE.md §5.2.
"""

from __future__ import annotations


from openenv.core.env_server.http_server import create_app

from ..models import TriageSieveAction, TriageSieveObservation
from ..server.triagesieve_env_environment import TriageSieveEnvironment

app = create_app(
    env=TriageSieveEnvironment,
    action_cls=TriageSieveAction,
    observation_cls=TriageSieveObservation,
    env_name="triagesieve_env",
    max_concurrent_envs=4,
)

__all__ = ["app", "main"]

def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="TriageSieve-OpenEnv server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
    

if __name__ == "__main__":
    main()