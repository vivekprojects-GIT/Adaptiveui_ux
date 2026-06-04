---
title: AdaptiveUI (vivek)
emoji: "🧠"
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# AdaptiveUI (vivek)

Claude-style single-call response **+ an interactive widget** for each answer. Widgets are
built **only from the app's own UI components** (a registry), expressed as JSON and rendered
by Vue components / ECharts — **no LLM-generated HTML, no iframe, no external chart libraries**.
A Thompson-Sampling bandit adapts the prose-answer style; the widget is independent of it.

See `vivek/ARCHITECTURE.md` for the full pipeline (registry → prompt menu → synthesizer →
validate → components).

This Space runs the `vivek` server on port 7860.
