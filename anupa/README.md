---
sdk: docker
app_port: 7860
---
# Adaptive Presentation Engine — Demo

A web demo of the Contextual Hierarchical Bayesian Architecture pipeline.

## What it shows
- **Live strategy selection** via Thompson Sampling over the Bayesian posterior
- **Posterior updating in real-time** as you rate responses (👍 / 👎)
- **Feature vector** used for each inference
- **Per-strategy expected reward** estimates that evolve with each interaction

---

## Setup (5 minutes)

### 1. Install Python dependencies
```bash
pip install numpy
# (flask is not needed — the server is built-in)
```

### 2. Configure model (optional)
This app supports:
- OpenAI-compatible providers (via `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`)
- Anthropic Claude (via `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`)

### 3. Run the demo server
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
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
