# Frontend

UI assets for the Adaptive Presentation Engine.

## Contents

- **index.html** — Single-page app with inline CSS and JS. Contains:
  - Chat UI (message input, send, history)
  - Strategy selector and posterior visualization (Thompson Sampling bars)
  - Widget renderer (iframes for HTML widgets, or JSON schema renderer)
  - Thumbs up/down feedback, session reset

## Layout

The backend serves `index.html` at `GET /`. All API calls go to `/api/*` endpoints.

## Adding assets

To add CSS, JS, or images:

1. Place files in `frontend/` (e.g. `frontend/js/app.js`, `frontend/css/main.css`).
2. Update `server.py` to serve static files under a path like `/static/` if needed.
3. Reference them in index.html (e.g. `<script src="/static/js/app.js">`).

Currently the app is self-contained in `index.html` with inline styles and CDN links.
