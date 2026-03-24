"""Entry point for the Adaptive Presentation Engine.

Starts the HTTP server that serves the frontend (GET /) and API endpoints
(/api/chat, /api/chat_plain, /api/health, etc.). Run from the project root (vivek/).

Usage:
  pip install -r requirements.txt
  cp .env.example .env   # then add your API keys
  python app.py
  open http://localhost:5051   # or PORT from env (default 5051)

Project layout:
  vivek/
  ├── frontend/     # UI assets (index.html)
  ├── backend/      # Python API, LLM calls, engine
  └── app.py        # this file
"""

from backend.server import run_server


if __name__ == "__main__":
    run_server()
