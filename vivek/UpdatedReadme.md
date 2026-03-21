# Vivek: Adaptive Presentation Engine — Complete Reference

A professional, modular backend demonstrating Bayesian strategy selection and real-time posterior updating. This app learns user preferences and adapts its response format (bullet points, prose, questions, etc.) based on observed rewards.

---

## Table of Contents
- [Project Structure](#project-structure)
- [File Manifest](#file-manifest)
- [System Architecture](#system-architecture)
- [How the App Works](#how-the-app-works)
- [Setup & Running](#setup--running)
- [Configuration](#configuration)
- [API Reference](#api-reference)

---

## Project Structure

```
backend/
├── __init__.py           # Package entrypoint
├── config.py             # Environment + constants
├── utils.py              # Math, heuristics, post-processing
├── llm.py                # LLM backends (Anthropic, OpenAI-compatible)
├── engine.py             # Bayesian learner
└── server.py             # HTTP handlers + runner

Root folder
├── app.py                # Launcher (imports backend.server.run_server)
├── index.html            # Frontend demo UI
├── Dockerfile            # Container image
├── requirements.txt      # Python dependencies
├── README.md             # Original readme (kept for reference)
└── UpdatedReadme.md      # This file
```

---

## File Manifest

### `backend/__init__.py`
**Purpose:** Package initialization and public API.

Exposes `run_server()` so callers only need:
```python
from backend import run_server
run_server()
```

---

### `backend/config.py`
**Purpose:** Centralized configuration and constants.

**Key variables:**
- `LLM_MODE` — selects backend: `"openai_compat"` or `"anthropic"`
- `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` — remote OpenAI-compatible API (Groq)
- `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` — Anthropic Claude API configuration
- `D=10` — feature vector dimensionality
- `LAMBDA`, `GAMMA`, `ALPHA_G`, `TS_TEMPERATURE` — Bayesian hyperparameters
- `STRATEGIES` — dict of 5 response primitives (bulleted, narrative, concise, Socratic, step-by-step)
- `HERE`, `INDEX_HTML` — paths for serving the frontend

**Usage:** Other modules import from this single source of truth.

---

### `backend/utils.py`
**Purpose:** Small, reusable utilities.

**Functions:**
- `sigmoid(x)` — numerically stable logistic function
- `mean_uncertainty(sigma_inv)` — summarize posterior variance from precision matrix
- `fast_valence(message, prev_response)` — lightweight regex-based sentiment heuristic; returns `{"pos", "neg", "reason"}`
- `enforce_response(strategy, text)` — post-process LLM output to match the chosen format (strips questions, forces bullets/numbers, caps sentence counts)

**Helpers (module-level):**
- `_POS`, `_NEG`, `_REPHRASE` — regex patterns for auto-reward detection

**Usage:** Called by server during chat turns and posterior updates.

---

### `backend/llm.py`
**Purpose:** LLM API wrappers for Anthropic and OpenAI-compatible endpoints.

**Public functions:**
- `call_openai_compat(prompt, system, timeout=120)` — POST to OpenAI-compatible endpoint; returns `(text, elapsed_sec, mode)`
- `call_anthropic(prompt, system, timeout=120)` — call Anthropic Messages API; returns `(text, elapsed_sec, mode)`
- `openai_health(timeout=10)` — check OpenAI-compatible endpoint and available models
 - `anthropic_health()` — lightweight Anthropic config health summary

**Internal helpers:**
- `_post_json_url()`, `_get_json_url()` — raw HTTP wrappers

**Usage:** Server calls these during `/api/chat` to fetch responses from the LLM.

---

### `backend/engine.py`
**Purpose:** The core Bayesian learner for strategy selection and posterior updating.

**Class: `BayesianEngine`**
- `__init__()` — initialize global and per-user posterior means (`mu`) and precision matrices (`sigma_inv`)
- `get_user(uid)` — fetch or create user state
- `featurize(message, user)` — convert message + history into a fixed-length feature vector (10-dim)
- `select(uid, message)` — use Thompson Sampling to pick a strategy; return `(strategy, scores, x)`
- `update(uid, strategy, x, reward)` — Bayesian update for user and global posterior
- `apply_preferences(uid, strategy_names)` — apply soft bias or hard-lock if user selects one strategy
- `posterior_summary()`, `user_posterior()`, `global_posterior()` — compute compact summaries of expected reward + uncertainty

**Singletons:**
- `engine` — global instance used by the server
- `USERB_ID` — reserved user ID for a secondary reference posterior (demo artifact)

**Bayesian model:** Online logistic regression with exponential decay (γ=0.99) and per-strategy Gaussian posteriors.

**Usage:** Server calls during `/api/chat`, `/api/reward`, `/api/preference` endpoints.

---

### `backend/server.py`
**Purpose:** HTTP server and request handlers.

**Class: `Handler(BaseHTTPRequestHandler)`**
Handles:
- `GET /` — serve `index.html` frontend
- `GET /api/health` — return LLM backend status
- `GET /api/state` — return user's current posterior and global stats
- `POST /api/chat` — accept user message, auto-reward previous turn, select strategy, call LLM, enforce format, persist state
- `POST /api/reward` — accept explicit user reward
- `POST /api/preference` — set user strategy preferences
- `POST /api/reset` — reset user state
- `OPTIONS *` — handle CORS preflight

**Private helpers:**
- `_json()` — JSON response with CORS headers
- `_html()` — serve frontend file
- `_body()` — parse JSON request body
- `_cors()` — set CORS headers
- `log_message()` — suppress access logs for cleanliness

**Function: `run_server()`**
- Prints startup banner
- Performs lightweight health checks on Anthropic/OpenAI endpoint
- Starts `ThreadedServer` on port 5051 (configurable via `PORT` env var)

**Class: `ThreadedServer(ThreadingMixIn, HTTPServer)`**
- Allows concurrent request handling

**Usage:** Imported and called by `app.py`.

---

### `app.py` (Root)
**Purpose:** Lightweight launcher script.

Simply imports and calls:
```python
from backend.server import run_server

if __name__ == "__main__":
    run_server()
```

This keeps a familiar entrypoint (`python app.py`) while the implementation lives in the package.

---

### `index.html`
**Purpose:** Frontend React/Preact demo.

Communicates with backend via:
- `GET /api/health` — check LLM status on load
- `GET /api/state` — fetch user posteriors
- `POST /api/chat` — send message, receive response + Bayesian state
- `POST /api/reward` — send user feedback
- `POST /api/preference` — set strategy lock
- `POST /api/reset` — reset session

Displays:
- Strategy label and instruction
- Expected reward scores per strategy
- Feature vector (x ∈ ℝ¹⁰)
- Posterior bar charts (mean + uncertainty)
- Auto-detected valence reason

---

### `Dockerfile`
**Purpose:** Container build for deployment.

Installs Python, dependencies, and runs `python app.py`.

---

### `requirements.txt`
**Purpose:** Python package dependencies.

Currently:
- `numpy` — linear algebra for Bayesian updates
- `python-dotenv` — load `.env` for API keys

---

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend (index.html)                       │
│         Browser UI → POST /api/chat, GET /api/state, etc.          │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  app.py (Launcher)                                  │
│              from backend.server import run_server()                  │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│          HTTP Server + Handler (backend/server.py)                    │
│  ├─ GET /api/health     → check LLM backend                        │
│  ├─ GET /api/state      → fetch posteriors                         │
│  ├─ POST /api/chat      → process user message                     │
│  ├─ POST /api/reward    → apply explicit reward                    │
│  ├─ POST /api/preference→ lock strategy                            │
│  └─ POST /api/reset     → clear user data                          │
└────────────────────────┬─────────┬──────────────┬──────────────────┘
                         │         │              │
         ┌───────────────┘         │              └──────────────┐
         │                         │                             │
         ▼                         ▼                             ▼
   ┌──────────────┐          ┌─────────────┐            ┌────────────────┐
  │ Bayesian     │          │ LLM Backends│            │ Utils (Heuristics
  │ Engine       │          │ (backend/llm) │            │ & Post-process)
  │ (backend/      │          │  ┌─ Anthropic │          │  ├─ sigmoid()
  │  engine.py)  │          │  └─ OpenAI   │            │  ├─ fast_valence()
  │              │          │             │            │  └─ enforce_response()
   │ • select()   │◄─────────┤ • call_*()  │            └────────────────┘
   │ • update()   │          │ • health()  │
   │ • featurize()│          │ • format_*()│
   │ • apply_pref│          │             │
   └──────────────┘          └─────────────┘
         ▲
         │ (reads hyperparams + strategy list)
         │
         ▼
   ┌──────────────────────────────────────────┐
  │ Config (backend/config.py)                 │
  │ ├─ LLM_MODE, OPENAI_*, ANTHROPIC_*      │
   │ ├─ D, LAMBDA, GAMMA, ALPHA_G, TS_TEMP   │
   │ ├─ STRATEGIES dict + STRATEGY_NAMES      │
   │ └─ HERE, INDEX_HTML paths               │
   └──────────────────────────────────────────┘
```

---

## How the App Works

### 1. **Initialization**
- User loads `http://localhost:5051` → frontend fetches `/api/health` and `/api/state`
- Server initializes or fetches user state (in-memory dict keyed by `uid`)
- User inherits global posterior (hierarchical Bayesian prior)

### 2. **User Sends a Message**
```
User message
    ↓
POST /api/chat {uid, message}
    ↓
[Server] fast_valence(message, prev_response)
    ├─ Auto-reward previous turn (if exists)
    └─ Call engine.update(uid, strategy, x, reward)
    ↓
[Server] engine.select(uid, message)
    ├─ featurize(message, user) → x ∈ ℝ¹⁰
    ├─ Thompson Sampling per strategy
    └─ Return best strategy + scores
    ↓
[Server] Build system prompt with FORMAT RULE
    ├─ Include selected strategy instruction
    ├─ Add recent conversation history
    └─ Send to LLM
    ↓
[LLM] Generate response (single-call)
    └─ Return text + optional widget HTML
    ↓
[Server] enforce_response(strategy, text)
    ├─ Strip unwanted questions
    ├─ Force format (bullets, numbers, etc.)
    └─ Return polished response
    ↓
[Server] Persist conversation + state
    └─ Update user["history"], ["last_response"], ["last_x"]
    ↓
[Server] Return JSON response
    ├─ response text + strategy label
    ├─ instruction (what the system told the LLM)
    ├─ scores (expected reward per strategy)
    ├─ x_vec (feature vector used)
    ├─ posteriors (updated beliefs)
    ├─ auto_detected + auto_r (heuristic reward)
    └─ auto_reason (why the heuristic fired)
    ↓
[Frontend] Display response
    ├─ Show strategy + instruction
    ├─ Show bar charts for expected reward
    ├─ Show feature vector
    └─ Display 👍/👎 buttons
```

### 3. **User Rates Response**
```
User clicks 👍 or 👎
    ↓
POST /api/reward {uid, strategy, x_vec, reward}
    ↓
[Server] engine.update(uid, strategy, x, reward)
    ├─ Update user["mu"][strategy] via logistic regression
    ├─ Update user["sigma_inv"][strategy] (precision matrix)
    ├─ Also update global posterior (with weight α=0.05)
    └─ Append to user["reward_log"]
    ↓
[Server] Recompute posteriors
    └─ Return new bar chart data
    ↓
[Frontend] Animate bar chart updates
    └─ Show how beliefs changed
```

### 4. **Posterior Update (Bayesian Mechanics)**
The engine maintains per-user **Gaussian posteriors** β ~ N(μ, Σ) for each strategy.

**Model:** Logistic regression where reward r̂ = sigmoid(x^T β)

**Update rule:**
- r̂_old = sigmoid(x^T μ_old)
- Σ_new = Σ_old^{-1} + x x^T · w + λ I  (w = r̂(1 - r̂))
- μ_new = μ_old + Σ_new^{-1} x (r - r̂_old)

**Global posterior** gets a small, weighted update (α=0.05) so all users benefit from collective learning.

**Exponential decay** (γ=0.99) biases the engine toward recent history.

---

## Setup & Running

### 1. Install Python 3.8+
```bash
python3 --version
```

### 2. Clone/download the repo
```bash
cd backend
```

### 3. Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. LLM backend
This repository supports OpenAI-compatible providers and Anthropic Claude.

### 6. Run the app
```bash
python app.py
```

You should see a startup banner and then the server will be live at `http://localhost:5051`.

### 7. Open browser
```
http://localhost:5051
```

---

## Configuration

### Environment Variables

Create a `.env` file in the root folder:

```env
# Backend selection
LLM_MODE=openai_compat  # or "anthropic"

# OpenAI-compatible (Groq, etc.)
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=your_groq_api_key_here
OPENAI_MODEL=llama-3.1-8b-instant

# Anthropic (Claude)
ANTHROPIC_API_KEY=your_anthropic_key_here
ANTHROPIC_MODEL=claude-opus-4-6

# Port
PORT=5051
```

### Bayesian Hyperparameters

Edit `backend/config.py`:

```python
D = 10              # Feature vector dimension
LAMBDA = 0.01       # L2 regularization on posteriors
GAMMA = 0.99        # Exponential decay (prefer recent history)
ALPHA_G = 0.05      # Global posterior update weight
TS_TEMPERATURE = 2.0  # Thompson Sampling variance scale (exploration)
```

---

## API Reference

### `GET /api/health`
**Returns:** LLM backend status.

**Response:**
```json
{
  "server": "ok",
  "mode": "openai_compat",
  "openai_base_url": "https://api.groq.com/openai/v1",
  "model": "llama-3.1-8b-instant",
  "ok": true,
  "reachable": true,
  "models": ["llama-3.1-8b-instant", "mixtral-8x7b-32768"]
}
```

---

### `GET /api/state?uid=demo`
**Returns:** Current user state and posteriors.

**Response:**
```json
{
  "posterior": {
    "structured_bullets": {"r": 0.52, "u": 0.34},
    "narrative_prose": {"r": 0.48, "u": 0.35},
    ...
  },
  "global": {...},
  "userb": {...},
  "global_n": 42,
  "n_users": 3,
  "msg_count": 5
}
```

---

### `POST /api/chat`
**Body:**
```json
{
  "uid": "demo",
  "message": "How do I make pasta?"
}
```

**Response:**
```json
{
  "response": "- Cook 1 liter of water\n- Add salt\n- ...",
  "strategy": "step_by_step",
  "instruction": "Numbered list of 3-6 steps only.",
  "elapsed": 2.3,
  "llm_mode": "anthropic",
  "scores": {
    "structured_bullets": 0.54,
    "narrative_prose": 0.48,
    ...
  },
  "x_vec": [0.12, 0.34, ...],
  "posterior": {...},
  "global": {...},
  "auto_detected": true,
  "auto_r": 0.75,
  "auto_reason": "positive signal(s)"
}
```

---

### `POST /api/reward`
**Body:**
```json
{
  "uid": "demo",
  "strategy": "step_by_step",
  "x_vec": [0.12, 0.34, ...],
  "reward": 0.9
}
```

**Response:**
```json
{
  "posterior": {...},
  "global": {...},
  "global_n": 43
}
```

---

### `POST /api/preference`
**Body:**
```json
{
  "uid": "demo",
  "strategies": ["structured_bullets"]
}
```

**Response:**
```json
{
  "posterior": {...}
}
```

---

### `POST /api/reset`
**Body:**
```json
{
  "uid": "demo"
}
```

**Response:**
```json
{
  "ok": true
}
```

---

## Troubleshooting

### Server won't start
- Ensure port 5051 is available: `lsof -i :5051`
- Check Python version: `python --version` (need 3.8+)
- Verify dependencies: `pip install -r requirements.txt`

### LLM returns empty responses
- If using OpenAI-compatible: verify API key in `.env`
- If using Anthropic: verify `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL`
- Check model name matches `OPENAI_MODEL` / `ANTHROPIC_MODEL`

### Feature vector or posterior looks weird
- This is expected! Bayesian posteriors start with high uncertainty.
- Send a few more messages and rate them — posteriors will stabilize.

### Valence heuristic seems wrong
- The regex patterns in `utils.py` are intentionally simple.
- For production, replace with a small classifier model.

---

## License & Attribution

Built as a demonstration of hierarchical Bayesian architecture. Feel free to adapt for your use case.

For questions or contributions, reach out to the team.
