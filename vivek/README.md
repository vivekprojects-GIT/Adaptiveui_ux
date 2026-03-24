---
sdk: docker
app_port: 7860
---
# Adaptive Presentation Engine — Demo

A web demo of the Contextual Hierarchical Bayesian Architecture pipeline.

## Project structure (modular layout)

```
vivek/
├── frontend/           # UI assets
│   ├── index.html      # Single-page app (chat, widgets, posterior viz)
│   └── README.md       # Frontend documentation
├── backend/            # Python API and logic
│   ├── config.py       # Env vars, LLM modes, Thompson Sampling params
│   ├── server.py       # HTTP server, API routes, serves frontend at GET /
│   ├── llm.py          # Anthropic and OpenAI-compatible LLM calls
│   ├── engine.py       # Bayesian engine (Thompson Sampling)
│   ├── widget_prompt.py
│   ├── combined_prompt.py
│   └── primitives.json # Strategy definitions
├── app.py              # Entry point
├── requirements.txt
├── .env.example        # Template for API keys (copy to .env)
└── .env                # Your keys (gitignored)
```

## What it shows
- **Live strategy selection** via Thompson Sampling over the Bayesian posterior
- **Posterior updating in real-time** as you rate responses (👍 / 👎)
- **Feature vector** used for each inference
- **Per-strategy expected reward** estimates that evolve with each interaction

---

## Setup (5 minutes)

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
# (numpy, python-dotenv, anthropic — server is built-in, no Flask)
```

### 2. Configure environment
Copy the example env file and add your API keys:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and/or ANTHROPIC_API_KEY
```

This app supports:
- OpenAI-compatible providers (via `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`)
- Anthropic Claude (via `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`)

### 3. Run the demo server
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5051   # or PORT from env (Docker uses 7860)
```

---

## How to demo it

1. **Send a few messages** — watch the strategy get selected in the sidebar
2. **Rate the responses** with 👍 or 👎 — watch the posterior bars update live
3. **Send different message types** (short vs long, questions vs statements) — the feature vector changes
4. **After 5-10 interactions**, the engine starts preferring strategies that got positive rewards
5. **Reset session** to show the system starting fresh from the global prior

---

## Pipeline stages shown
| Stage | What the demo shows |
|---|---|
| Feature extraction | Feature vector panel (x ∈ ℝ⁸) |
| Thompson Sampling | Expected reward % per strategy |
| LLM rendering | Live response with strategy label |
| Reward observation | 👍/👎 buttons |
| Posterior update | Bar charts animate in real-time |

---

## Architecture notes (for Q&A)
- **No JSON files** — posterior stored in-memory (Redis in production)
- **Hierarchical prior** — new users inherit global posterior
- **Exponential decay** — old observations lose weight over time (γ=0.99)
- **Global update** — each interaction slightly updates the shared prior (α=0.05)
- **Circuit breaker** — LLM timeouts fail fast gracefully
