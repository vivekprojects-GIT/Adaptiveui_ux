"""HTTP server and request handlers for the Adaptive Presentation Engine backend."""

import json
import os
import re
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import numpy as np

from . import config, llm
from .engine import engine, USERB_ID
from .widget_prompt import (
    estimate_widget_height,
    inject_design_system,
)
from .combined_prompt import (
    build_combined_system_prompt,
    build_combined_user_prompt,
    parse_combined_output,
)
from .utils import (
    fast_valence,
    enforce_response,
    detect_format_override,
    detect_explore_trigger,
    negative_strength,
)

def _looks_truncated_widget_html(html: str) -> bool:
    """Heuristic check for obviously cut-off widget HTML."""
    if not html:
        return True
    lower = html.lower()
    # Missing critical closures often indicates token truncation.
    if lower.count("<style") > lower.count("</style>"):
        return True
    if lower.count("<script") > lower.count("</script>"):
        return True
    if lower.count("<body") > lower.count("</body>"):
        return True
    if lower.count("<html") > lower.count("</html>"):
        return True
    # Rarely, response ends mid-token; catch abrupt ending.
    if re.search(r"[<{(]$", html.strip()):
        return True
    return False


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _html(self):
        try:
            html = config.INDEX_HTML.read_bytes()
        except FileNotFoundError:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        raw = self.rfile.read(n)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Don't crash the handler thread; return a sentinel the caller can handle.
            try:
                preview = raw[:500].decode("utf-8", errors="replace")
            except Exception:
                preview = repr(raw[:200])
            return {"__invalid_json__": True, "__raw_preview__": preview}

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            self._html(); return

        if p == "/api/health":
            if config.LLM_MODE == "openai_compat":
                h = llm.openai_health()
                self._json({"server": "ok", "mode": config.LLM_MODE, "openai_base_url": config.OPENAI_BASE_URL, "model": config.OPENAI_MODEL, **h})
            elif config.LLM_MODE == "anthropic":
                h = llm.anthropic_health()
                self._json({"server": "ok", "mode": config.LLM_MODE, **h})
            else:
                self._json({"server": "ok", "mode": config.LLM_MODE, "ok": False, "reachable": False, "error": "Unsupported LLM_MODE (expected openai_compat or anthropic)"})
            return

        if p == "/api/state":
            uid  = self.path.split("uid=")[-1] if "uid=" in self.path else "demo"
            user = engine.get_user(uid)
            x    = np.ones(config.D) * 0.5
            ub   = engine.get_user(USERB_ID)
            self._json({
                "posterior": engine.user_posterior(uid, x),
                "global":    engine.global_posterior(x),
                "userb":     engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
                "global_n":  engine.global_n,
                "n_users":   len(engine.users),
                "msg_count": user["msg_count"],
            }); return

        self.send_response(404); self.end_headers()

    def do_POST(self):
        p    = self.path.split("?")[0]
        body = self._body()
        if isinstance(body, dict) and body.get("__invalid_json__"):
            self._json({"error": "invalid_json", "preview": body.get("__raw_preview__", "")}, 400)
            return

        if p == "/api/chat_plain":
            uid = body.get("uid", "demo") + "_plain"
            msg = body.get("message", "").strip()
            if not msg:
                self._json({"error": "empty message"}, 400); return

            user = engine.get_user(uid)

            # Build a simple conversation prompt (no bandit, no enforced format).
            ctx = []
            for t in user["history"][-6:]:
                ctx += [f"User: {t['user']}", f"Assistant: {t['assistant']}"]
            ctx.append(f"User: {msg}")
            prompt = "\n".join(ctx)

            system = "You are a helpful AI assistant."

            try:
                base_mode = (config.BASELINE_LLM_MODE or config.LLM_MODE).lower()
                if base_mode == "openai_compat":
                    response, elapsed, mode = llm.call_openai_compat(prompt, system, timeout=120)
                elif base_mode == "anthropic":
                    response, elapsed, mode = llm.call_anthropic(prompt, system, timeout=120)
                else:
                    raise RuntimeError("Unsupported BASELINE_LLM_MODE (expected openai_compat or anthropic)")
            except Exception as e:
                self._json({"error": f"LLM error: {str(e)}"}, 500)
                return

            if not response:
                self._json({"error": "LLM returned empty response. Check model/service."}, 500)
                return

            user["history"].append({"user": msg, "assistant": response})
            user["history"] = user["history"][-20:]

            self._json({
                "response": response,
                "elapsed":  elapsed,
                "llm_mode": mode,
            }); return

        if p == "/api/chat":
            uid = body.get("uid", "demo")
            msg = body.get("message", "").strip()
            if not msg:
                self._json({"error": "empty message"}, 400); return

            user = engine.get_user(uid)

            # Auto-reward previous turn using valence heuristic.
            ev            = fast_valence(msg, user["last_response"])
            auto_detected = False
            auto_r        = None

            if user["last_response"] and user["last_x"] is not None and user["last_strategy"]:
                reward = float(np.clip(0.5 + 0.45*ev["pos"] - 0.45*ev["neg"], 0.05, 0.95))
                engine.update(uid, user["last_strategy"], np.array(user["last_x"]), reward)
                auto_detected = True
                auto_r        = reward

            # --- NEW: explicit overrides + corrective exploration ---
            explicit = detect_format_override(msg, config.STRATEGY_NAMES)
            force_explore = bool(detect_explore_trigger(msg) or (ev.get("neg", 0.0) >= config.NEG_EXPLORE_THRESHOLD))
            neg_s = negative_strength(ev)

            strat, scores, x, prev = engine.select(
                uid, msg,
                force_explore=force_explore,
                neg_strength=neg_s,
                explicit_strategy=explicit,
            )

            format_rule = config.STRATEGIES.get(strat, "Be helpful and clear.")

            # ── Single-call combined prompt (Claude-style) ─────────────────
            combined_max_tokens = getattr(config, "COMBINED_MAX_TOKENS", 2800)
            # Keep the UI responsive by failing fast by default.
            combined_timeout = getattr(config, "COMBINED_TIMEOUT_SECONDS", 30)

            combined_system = build_combined_system_prompt(
                strategy_id=strat,
                format_rule=format_rule,
                primitive_extra_context="",
                user_message=msg,
                forbidden_components=None,
                required_components=None,
            )
            combined_prompt = build_combined_user_prompt(
                user_message=msg,
                history=user["history"],
            )

            try:
                adapt_mode = (config.ADAPTIVE_LLM_MODE or config.LLM_MODE).lower()
                if adapt_mode == "openai_compat":
                    raw_combined, elapsed, mode = llm.call_openai_compat(
                        combined_prompt,
                        combined_system,
                        timeout=combined_timeout,
                        max_tokens=combined_max_tokens,
                        temperature=0.2,
                    )
                elif adapt_mode == "anthropic":
                    raw_combined, elapsed, mode = llm.call_anthropic(
                        combined_prompt,
                        combined_system,
                        timeout=combined_timeout,
                        max_tokens=combined_max_tokens,
                        temperature=0.2,
                    )
                else:
                    raise RuntimeError("Unsupported ADAPTIVE_LLM_MODE (expected openai_compat or anthropic)")
            except Exception as e:
                self._json({"error": f"LLM error: {str(e)}"}, 500)
                return

            if not raw_combined:
                self._json({"error": "LLM returned empty response. Check model/service."}, 500)
                return

            # Parse combined output into text response + widget payload (HTML or JSON schema).
            response, widget_payload_raw = parse_combined_output(raw_combined)

            if not response:
                response = raw_combined.strip()

            if config.STRICT_PRIMITIVES:
                response = enforce_response(strat, response)

            # ── Validate widget from single call ───────────────────────────
            widget_html = ""
            widget_schema = ""
            widget_height = 0
            widget_debug = ""
            widget_mode = getattr(config, "WIDGET_MODE", "json").strip().lower()

            if widget_payload_raw:
                if widget_mode == "json":
                    widget_schema = widget_payload_raw
                    widget_debug = widget_debug or "combined_schema_ok"
                else:
                    if _looks_truncated_widget_html(widget_payload_raw):
                        widget_debug = "combined_widget_truncated"
                    else:
                        widget_html = widget_payload_raw
                        widget_height = estimate_widget_height(widget_payload_raw)
                        widget_debug = widget_debug or "combined_widget_ok"
            else:
                widget_debug = widget_debug or ("combined_no_schema" if widget_mode == "json" else "combined_no_widget_tag")
                raw_preview = (raw_combined or "")[:800]
                if widget_mode != "json":
                    # Never return a blank iframe: serve a safe interactive placeholder mini-app.
                    placeholder = f"""
<html><head></head><body>
  <div class="widget-root card">
    <div class="card-title">Interactive widget</div>
    <div style="color:var(--text2);font-size:13px;line-height:1.6">
      I couldn't generate a full widget for this turn. This placeholder stays interactive and can request the missing data.
    </div>
    <div class="ctrl-row" style="margin-top:12px">
      <div class="ctrl-lbl">Assumption</div>
      <input id="assump" type="range" min="0" max="100" step="1" value="50" style="flex:1" />
      <div class="ctrl-val" id="assumpv">50</div>
    </div>
    <div id="chart" class="raised" style="height:260px;margin-top:10px"></div>
    <div class="btn-group" style="margin-top:10px">
      <button class="btn" id="ask_range">Ask for date range</button>
      <button class="btn" id="ask_source">Ask for data source</button>
    </div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/echarts/dist/echarts.min.js"></script>
  <script>
  const data = [
    {{name:'Series A', base: 100}},
    {{name:'Series B', base: 100}}
  ];
  const state = {{assumption: 50}};
  function compute(state, data) {{
    const k = (state.assumption/50);
    return {{
      labels: data.map(d=>d.name),
      values: data.map((d,i)=>Math.round(d.base*(i? (1.15*k):(1.05*k))))
    }};
  }}
  let chart;
  function render() {{
    const c = compute(state, data);
    document.getElementById('assumpv').textContent = String(state.assumption);
    if (!chart) chart = echarts.init(document.getElementById('chart'));
    chart.setOption({{
      backgroundColor:'transparent',
      tooltip:{{trigger:'axis'}},
      xAxis:{{type:'category',data:c.labels,axisLabel:{{color:'#5a5f72'}}}},
      yAxis:{{type:'value',axisLabel:{{color:'#5a5f72'}}}},
      series:[{{type:'bar',data:c.values}}]
    }});
  }}
  document.getElementById('assump').addEventListener('input', (e)=>{{ state.assumption = +e.target.value; render(); }});
  document.getElementById('ask_range').onclick = ()=> sendPrompt('Provide a date range to plot (e.g., 2020-01-01 to 2024-12-31).');
  document.getElementById('ask_source').onclick = ()=> sendPrompt('Which data source should we use for SPY and QQQ prices (yfinance, stooq, other)?');
  render();
  </script>
</body></html>
""".strip()
                    widget_html = inject_design_system(placeholder)
                    widget_height = estimate_widget_height(widget_html)
                    widget_debug = "fallback_widget_generated"

            # No primitive fallback. If widget is missing/invalid, return text-only.

            user["history"].append({"user": msg, "assistant": response})
            user["history"]       = user["history"][-20:]
            user["last_message"]  = msg
            user["last_response"] = response
            user["last_strategy"] = strat
            user["last_x"]        = x.tolist()
            user["msg_count"]    += 1

            ub = engine.get_user(USERB_ID)
            self._json({
                "response":      response,
                "strategy":      strat,
                "prev_strategy": prev,
                "explicit":      explicit,
                "force_explore": force_explore and (explicit is None),
                "instruction":   config.STRATEGIES[strat],
                "elapsed":       elapsed,
                "llm_mode":      mode,
                "scores":        {k: round(v, 4) for k, v in scores.items()},
                "x_vec":         x.tolist(),
                "posterior":     engine.user_posterior(uid, x),
                "global":        engine.global_posterior(x),
                "userb":         engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
                "global_n":      engine.global_n,
                "auto_detected": auto_detected,
                "auto_r":        auto_r,
                "auto_reason":   ev["reason"],
                "widget_html":   widget_html,
                "widget_schema": widget_schema,
                "widget_height": widget_height,
                "widget_debug":  widget_debug,
                "widget_raw_preview": raw_preview if not (widget_html or widget_schema) else "",
            }); return

        if p == "/api/reward":
            uid      = body.get("uid", "demo")
            strategy = body.get("strategy")
            x_vec    = body.get("x_vec")
            reward   = float(body.get("reward", 0.5))
            if strategy not in config.STRATEGY_NAMES or x_vec is None:
                self._json({"error": "bad request"}, 400); return
            x = np.array(x_vec, dtype=float)
            engine.update(uid, strategy, x, reward)
            ub = engine.get_user(USERB_ID)
            self._json({
                "posterior": engine.user_posterior(uid, x),
                "global":    engine.global_posterior(x),
                "userb":     engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
                "global_n":  engine.global_n,
            }); return

        if p == "/api/preference":
            uid = body.get("uid", "demo")
            strategies = body.get("strategies", [])
            lock = bool(body.get("lock", False))
            engine.apply_preferences(uid, strategies, lock=lock)
            self._json({"posterior": engine.user_posterior(uid)}); return

        if p == "/api/reset":
            engine.reset_user(body.get("uid", "demo"))
            self._json({"ok": True}); return

        # ── NEW: adaptive streaming endpoint (Claude-like) ────────────────
        if p == "/api/chat_stream":
            uid = body.get("uid", "demo")
            msg = body.get("message", "").strip()
            if not msg:
                self._json({"error": "empty message"}, 400); return

            user = engine.get_user(uid)

            # Auto-reward previous turn using valence heuristic.
            ev            = fast_valence(msg, user["last_response"])
            auto_detected = False
            auto_r        = None

            if user["last_response"] and user["last_x"] is not None and user["last_strategy"]:
                reward = float(np.clip(0.5 + 0.45*ev["pos"] - 0.45*ev["neg"], 0.05, 0.95))
                engine.update(uid, user["last_strategy"], np.array(user["last_x"]), reward)
                auto_detected = True
                auto_r        = reward

            explicit = detect_format_override(msg, config.STRATEGY_NAMES)
            force_explore = bool(detect_explore_trigger(msg) or (ev.get("neg", 0.0) >= config.NEG_EXPLORE_THRESHOLD))
            neg_s = negative_strength(ev)

            strat, scores, x, prev = engine.select(
                uid, msg,
                force_explore=force_explore,
                neg_strength=neg_s,
                explicit_strategy=explicit,
            )

            format_rule = config.STRATEGIES.get(strat, "Be helpful and clear.")

            combined_max_tokens = getattr(config, "COMBINED_MAX_TOKENS", 2800)
            combined_timeout = getattr(config, "COMBINED_TIMEOUT_SECONDS", 30)

            combined_system = build_combined_system_prompt(
                strategy_id=strat,
                format_rule=format_rule,
                primitive_extra_context="",
                user_message=msg,
                forbidden_components=None,
                required_components=None,
            )
            combined_prompt = build_combined_user_prompt(
                user_message=msg,
                history=user["history"],
            )

            adapt_mode = (config.ADAPTIVE_LLM_MODE or config.LLM_MODE).lower()

            # Start NDJSON stream.
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self._cors()
            self.end_headers()

            ub = engine.get_user(USERB_ID)

            def send_nd(evt: dict):
                try:
                    line = json.dumps(evt, ensure_ascii=False)
                    self.wfile.write(line.encode("utf-8") + b"\n")
                    self.wfile.flush()
                except Exception:
                    pass

            def send_done_error(err: str):
                send_nd({
                    "type": "done",
                    "strategy": strat,
                    "elapsed": None,
                    "llm_mode": adapt_mode,
                    "response": "",
                    "widget_html": "",
                    "widget_schema": "",
                    "widget_height": 0,
                    "widget_debug": f"stream_error:{err}",
                    "error": err,
                    "force_explore": force_explore,
                    "scores": {k: round(v, 4) for k, v in scores.items()},
                    "x_vec": x.tolist(),
                    "posterior": engine.user_posterior(uid, x),
                    "global": engine.global_posterior(x),
                    "userb": engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
                    "global_n": engine.global_n,
                    "prev_strategy": prev,
                    "explicit": explicit,
                    "auto_detected": auto_detected,
                    "auto_r": auto_r,
                    "auto_reason": ev["reason"],
                })

            # Initial strategy event so UI updates immediately.
            send_nd({
                "type": "strategy",
                "strategy": strat,
                "instruction": config.STRATEGIES[strat],
                "elapsed": None,
                "force_explore": force_explore,
                "scores": {k: round(v, 4) for k, v in scores.items()},
                "x_vec": x.tolist(),
                "posterior": engine.user_posterior(uid, x),
                "global": engine.global_posterior(x),
                "userb": engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
                "global_n": engine.global_n,
                "prev_strategy": prev,
                "explicit": explicit,
                "auto_detected": auto_detected,
                "auto_r": auto_r,
                "auto_reason": ev["reason"],
            })

            try:
                if adapt_mode == "openai_compat":
                    raw_combined, elapsed, mode = llm.call_openai_compat(
                        combined_prompt,
                        combined_system,
                        timeout=combined_timeout,
                        max_tokens=combined_max_tokens,
                        temperature=0.2,
                    )
                elif adapt_mode == "anthropic":
                    # Use non-streaming call with wall-clock timeout (more robust than the SDK stream
                    # when the provider stalls), then optionally "replay" deltas to the UI.
                    raw_combined, elapsed, mode = llm.call_anthropic(
                        combined_prompt,
                        combined_system,
                        timeout=combined_timeout,
                        max_tokens=combined_max_tokens,
                        temperature=0.2,
                    )
                else:
                    raise RuntimeError("Unsupported ADAPTIVE_LLM_MODE (expected openai_compat or anthropic)")
            except Exception as e:
                msg = str(e).strip()
                if not msg:
                    msg = repr(e)
                send_done_error(f"LLM error ({type(e).__name__}): {msg}")
                return

            response, widget_payload_raw = parse_combined_output(raw_combined)
            if not response:
                response = raw_combined.strip()
            if config.STRICT_PRIMITIVES:
                response = enforce_response(strat, response)

            widget_debug = "nonstream"

            widget_html = ""
            widget_schema = ""
            widget_height = 0
            widget_mode = getattr(config, "WIDGET_MODE", "json").strip().lower()
            if widget_payload_raw:
                if widget_mode == "json":
                    widget_schema = widget_payload_raw
                else:
                    if _looks_truncated_widget_html(widget_payload_raw):
                        widget_debug = "stream_widget_truncated"
                    else:
                        widget_html = widget_payload_raw
                        widget_height = estimate_widget_height(widget_payload_raw)
            else:
                if widget_mode != "json":
                    placeholder = "<html><head></head><body><div class='widget-root card'><div class='card-title'>Interactive widget</div><div class='empty'>No widget returned; click to request missing data.</div><div class='btn-group'><button class='btn' onclick=\"sendPrompt('Provide a date range for the requested plot.')\">Ask for date range</button></div></div></body></html>"
                    widget_html = inject_design_system(placeholder)
                    widget_height = estimate_widget_height(widget_html)
                    widget_debug = "fallback_widget_generated"

            # Replay deltas so the UI doesn't look frozen (best-effort).
            # The final "done" still contains the full response/widget.
            try:
                if response:
                    for i in range(0, len(response), 180):
                        send_nd({"type": "response_delta", "delta": response[i : i + 180]})
                payload = widget_schema if widget_mode == "json" else widget_html
                if payload:
                    for i in range(0, len(payload), 900):
                        send_nd({"type": "widget_delta", "delta": payload[i : i + 900]})
            except Exception:
                pass

            # Update history as in /api/chat.
            user["history"].append({"user": msg, "assistant": response})
            user["history"] = user["history"][-20:]
            user["last_message"]  = msg
            user["last_response"] = response
            user["last_strategy"] = strat
            user["last_x"]        = x.tolist()
            user["msg_count"]    += 1

            send_nd({
                "type": "done",
                "strategy": strat,
                "elapsed": elapsed,
                "llm_mode": mode,
                "response": response,
                "widget_html": widget_html or "",
                "widget_schema": widget_schema or "",
                "widget_height": widget_height,
                "widget_debug": widget_debug,
                "force_explore": force_explore,
                "scores": {k: round(v, 4) for k, v in scores.items()},
                "x_vec": x.tolist(),
                "posterior": engine.user_posterior(uid, x),
                "global": engine.global_posterior(x),
                "userb": engine.posterior_summary(ub["mu"], ub["sigma_inv"], x),
                "global_n": engine.global_n,
                "prev_strategy": prev,
                "explicit": explicit,
                "auto_detected": auto_detected,
                "auto_r": auto_r,
                "auto_reason": ev["reason"],
            })
            return

        self.send_response(404); self.end_headers()


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def run_server():
    print("=" * 60)
    print(f"  http://localhost:5051   mode: {config.LLM_MODE}")
    print("=" * 60)

    PORT = int(os.getenv("PORT", "5051"))
    ThreadedServer(("0.0.0.0", PORT), Handler).serve_forever()
