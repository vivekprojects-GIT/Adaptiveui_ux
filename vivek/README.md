---
sdk: docker
app_port: 7860
---
# Adaptive Presentation Engine — Demo

A chat app where Claude answers a question **and** generates an interactive widget for it.
Widgets are built **only from the app's own UI components** (a registry), as JSON — **no
LLM-generated HTML, no iframe, no external chart libraries**. Charts render with bundled
ECharts. A per-user Thompson-Sampling bandit adapts the prose-answer style independently.

> **Architecture:** see [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full registry →
> prompt-menu → synthesizer → validate → components pipeline (Part B / visualization).

## Project structure

```
vivek/
├── frontend-vue/                 # Vue 3 + Vite + TypeScript SPA (the live UI)
│   ├── src/
│   │   ├── widget-registry.json  # SINGLE SOURCE: block types + 33 chart kinds + data shapes
│   │   ├── lib/widgetRegistry.ts # type → Vue component map (RENDER)
│   │   ├── lib/echartsOption.ts  # shared ECharts option builder (live render + HTML export)
│   │   ├── lib/exportWidgetHtml.ts # deterministic widget → standalone interactive HTML
│   │   ├── components/WidgetRegistryRenderer.vue  # parses widget JSON → <component :is>
│   │   ├── components/WidgetSchemaChart.vue       # ECharts chart component
│   │   └── components/widgets/*.vue               # TextBlock, KpiRow, ChartBlock, ...
│   └── dist/                     # built SPA (served by the backend in prod)
├── backend/
│   ├── config.py                 # env, LLM modes, bandit params, strategy loading
│   ├── server.py                 # FastAPI: auth, /api/chat[_stream], /api/rate, serves SPA
│   ├── llm.py                    # Anthropic + OpenAI-compatible calls
│   ├── engine.py                 # Thompson-Sampling bandit (text strategy)
│   ├── combined_prompt.py        # builds the combined prompt; registry → menu; validation
│   ├── db.py / auth.py / utils.py
├── strategies.json               # bandit text strategies (admin-manageable)
├── app.py                        # entry point
├── requirements.txt
├── .env.example                  # template (copy to .env; .env is gitignored)
└── ARCHITECTURE.md
```

## What it shows
- **Components-only widgets** — 9 block types + 33 chart kinds, all from the registry
- **Live strategy selection** via Thompson Sampling (text style only)
- **Posterior updating in real-time** as you rate responses (👍 / 👎)
- **Feature vector** (`x ∈ ℝ¹⁰`) used for each inference
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
| Feature extraction | Feature vector panel (x ∈ ℝ¹⁰) |
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
