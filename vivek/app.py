"""Entry point.

Run:
  pip install -r requirements.txt
  python app.py
  open http://localhost:5051
"""

from backend.server import run_server


if __name__ == "__main__":
    run_server()
