"""Single source of truth loader — reads frontend-vue/src/widget-registry.json so the
synthesizer prompt and the validator both derive the allowed block types + chart kinds
from the SAME file the renderer uses. Define a component once → all three stay in sync.
"""

from __future__ import annotations

import json
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "frontend-vue" / "src" / "widget-registry.json"


def _load() -> dict:
    # Read fresh each call so edits to the JSON propagate without a code change.
    try:
        return json.loads(_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def allowed_types() -> list[str]:
    blocks = _load().get("blocks") or []
    types = [str(b.get("type")) for b in blocks if isinstance(b, dict) and b.get("type")]
    return types or ["text", "kpi_row", "chart"]


def allowed_chart_kinds() -> list[str]:
    for b in _load().get("blocks") or []:
        if isinstance(b, dict) and str(b.get("type")).lower() == "chart" and isinstance(b.get("kinds"), list):
            return [str(k).lower() for k in b["kinds"]]
    return ["line", "bar", "pie", "donut"]
