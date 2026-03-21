"""Backend package for the Adaptive Presentation Engine.

This package contains core modules: configuration, LLM helpers, Bayesian
engine, utilities, and the HTTP server runner. All internal logic is
organized here for clean separation from the launcher (app.py).
"""

from .server import run_server

__all__ = ["run_server"]
