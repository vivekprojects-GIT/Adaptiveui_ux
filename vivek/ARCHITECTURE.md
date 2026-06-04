# Adaptive Presentation Engine — Architecture

A chat app where an LLM (Anthropic Claude) answers a question **and** generates an
interactive widget for it. The widget is built **only from the app's own UI components**
(a registry), expressed as **JSON** — there is **no LLM-generated HTML, no iframe, and no
external chart libraries shipped by the model**. Charts render with bundled **ECharts**.

There are two independent parts in every turn:

- **Part A — Adaptive text strategy** (the bandit): picks *how the prose answer is written*.
- **Part B — Visualization** (this doc's focus): decides *whether/what to visualize* and
  generates the widget from the component registry.

The two are decoupled: the bandit strategy only shapes the `<RESPONSE>` text; it never
controls the `<WIDGET>`.

---

## The single source of truth: `frontend-vue/src/widget-registry.json`

One JSON catalog of everything the app can render. It is read in **three** places so they
can never drift:

| Use | Reader | Purpose |
|-----|--------|---------|
| **GENERATE** | `backend/combined_prompt.py → build_json_widget_rule()` | turns the registry into the prompt "menu" sent to the LLM |
| **VALIDATE** | `rag_finance/registry.py → allowed_types() / allowed_chart_kinds()` | drops anything off-menu in the LLM's output |
| **RENDER** | `frontend-vue/src/lib/widgetRegistry.ts (COMPONENTS map)` + `lib/echartsOption.ts` | maps each block `type`/chart `kind` to a Vue component / ECharts option |

### Block types (9)
`text, kpi_row, chart, table, action_row, image, stat_card, progress, badge_row`

### Chart kinds (33) and their data shapes
```
bar | hbar | line | area | stacked | combo | histogram | timeseries
        → x_categories + series[].values   (also accepts items[] as one series)
pie | donut | rose | funnel | treemap | sunburst
        → items[{label,value}]
scatter | bubble        → series[].values = [x,y]  (bubble: [x,y,size])
boxplot                 → boxes[[min,q1,med,q3,max]]
candlestick             → candles[[open,close,low,high]]
waterfall               → items[{label,value=delta}]
heatmap                 → x_labels + y_labels + matrix
sankey | graph          → nodes[] + links[{source,target,value}]
tree | mindmap | org    → root{name,children[]}
scatter3d | bar3d | line3d  → series[].values = [x,y,z]   (echarts-gl)
polar | parallel | themeriver | gauge | radar
```
(Removed deliberately: `calendar`, `surface` — they need hundreds of hand-typed values,
which an LLM with no data backend can't author reliably.)

---

## Part B — Visualization flow (step by step)

**Example query:** `Show NVIDIA's FY2024 revenue by segment as a bar chart`

### 1. Assemble the prompt (`server.py → _build_adaptive_prompt → build_combined_system_prompt`)
The system message includes the registry-derived **WIDGET RULES** menu plus the guardrails:
- **OUTPUT CONTRACT** — return exactly `<RESPONSE>…</RESPONSE><WIDGET>{json}</WIDGET>`.
- **RENDER NOW** — if a visual is warranted and buildable, output the widget *this* turn;
  never defer (“I will plot…”), ask permission, or ask “which dataset?” when implied.
- **WIDGET WARRANT** — include a widget only if a visual helps **and** you can fill it with
  **real** values; else empty + say so.
- **NO FABRICATED DATA** — never invent/estimate/“mock”/“illustrative” data; if you don't
  have it, say so and render nothing.
- **STRICT grounding** — every number/label in the widget must equal what's in `<RESPONSE>`.
- **Decline** — if a requested chart kind isn't supported, say so and offer supported ones.
- **action_row = visuals-only** — buttons only redraw the *current* data as another kind.

The user message is just history + the question (no data backend, no tools).

### 2. Synthesizer decides + generates (one LLM call)
After writing the `<RESPONSE>` text, the model:
1. **WARRANT** — is a visual helpful, do I have real data, is there a supported kind?
   - no data → say so, empty widget · unsupported kind → decline · not worth it → text only.
2. **SELECT KIND** — match the data shape to a kind in the menu (here: categorical
   comparison → `bar`; an explicit request wins if supported).
3. **FILL THE SHAPE** — populate that kind's keys with the **same** numbers used in `<RESPONSE>`.
4. **EMIT:**
```json
<RESPONSE>
In FY2024, NVIDIA's revenue was led by Data Center at $47.5B, Gaming $10.4B,
Professional Visualization $1.6B, and Automotive $1.1B.
</RESPONSE>
<WIDGET>
{"version":"1.0","layout":[
  {"type":"text","content":"FY2024 revenue by segment (USD B)."},
  {"type":"chart","title":"NVIDIA FY2024 Revenue by Segment",
   "chart":{"kind":"bar","x_label":"Segment","y_label":"USD B",
     "x_categories":["Data Center","Gaming","Professional Visualization","Automotive"],
     "series":[{"name":"FY2024","values":[47.5,10.4,1.6,1.1]}]}},
  {"type":"action_row","buttons":[
     {"label":"Show as treemap"},{"label":"View as horizontal bars"}]}
]}
</WIDGET>
```

### 3. Parse + validate against the registry (`combined_prompt.py`, deterministic)
`parse_combined_output` splits RESPONSE/WIDGET → `_dispatch_json_mode_widget`:
- normalize type aliases (`_TYPE_ALIASES`), keep only `_BLOCK_TYPES`
- per block `_block_is_renderable` → charts checked by `_chart_has_data`
  (kind in registry **and** the data-bearing key present)
- off-menu / no-data blocks are **dropped** (no empty cards)
- returns `widget_schema` (JSON) + `widget_html=""` (the HTML path does not exist)

### 4. Stream to the UI (SSE)
`strategy → response_delta → widget_delta → done`. While streaming, the response text is
run through `cleanAssistantText`/`stripLeakedJson` (no raw JSON/ASCII ever shows), and the
partial widget renders **block-by-block** via the salvage parser (the chart appears the
moment its JSON block closes).

### 5. Render: type → component (`lib/widgetRegistry.ts`) → fill values → same component
`WidgetRegistryRenderer.vue` runs `parseLayout` (full parse → salvage → drop empty charts),
then for each block does:
```vue
<component
   :is="resolveWidget(block.type).component"   <!-- widgetRegistry.ts: type → your Vue component -->
   :block="block"                              <!-- the LLM's VALUES go in as a prop -->
/>
```

**`lib/widgetRegistry.ts` is the render mapping.** It reads the same `widget-registry.json`
and holds a `COMPONENTS` map of block `type` → your real component code:
```ts
const COMPONENTS = {
  text: TextBlock, kpi_row: KpiRow, chart: ChartBlock, table: TableBlock,
  action_row: ActionRow, image: ImageBlock, stat_card: StatCardBlock,
  progress: ProgressList, badge_row: BadgeRow,   // (lazy-imported)
}
resolveWidget(type) → the mapped component (or null)
```

So the render contract is:
```
type   → widgetRegistry.ts (resolveWidget) → your component (e.g. ChartBlock.vue)
values → passed as :block → the component reads them → renders
```

The LLM **never makes a component** — it only fills DATA into a block. That block (the
LLM's values) is handed to the **pre-existing, registered** component as the `:block` prop.
The component is defined **once** in your codebase and **reused** every turn; the model just
supplies different values. For our example:
```
text       → TextBlock     (renders the sentence, escaped)
chart      → ChartBlock → WidgetSchemaChart → buildEChartsOption(block.chart) → ECharts <div>
action_row → ActionRow     (clickable → re-prompts the same data)
```
`<component :is>` mounts real Vue component instances inline. **No iframe, no HTML, no CDN.**

> The point of the registry pattern: **one component definition, many data fills, zero
> model-generated UI.** `widget-registry.json` says what's possible; `widgetRegistry.ts`
> maps each type to your component; the LLM only supplies the values.

### One-line summary
```
widget-registry.json
  → build_json_widget_rule() → "menu" in the prompt
  → synthesizer: WARRANT? → pick KIND → fill SHAPE with real values → emit <WIDGET>{json}
  → validate against registry (drop off-menu/no-data; html="")
  → SSE stream → WidgetRegistryRenderer → resolveWidget(type) [lib/widgetRegistry.ts → COMPONENTS map]
  → <component :is :block> → your component reads the LLM's values → renders (ECharts for charts)
```

---

## Guarantees (by design)
- **Components only** — everything renders via `<component :is>`; no iframe/HTML/CDN libs.
- **Registry is the single source** — generate, validate, render can't drift.
- **No raw JSON, no ASCII art, no mock data, no empty cards** ever reach the UI.
- **Decline honestly** — unsupported chart type or missing data → say so, render nothing.
- **The model can only use plots it can actually draw** (the menu), and only with real data.

---

## Extras
- **HTML export** (`lib/exportWidgetHtml.ts`): deterministic `JSON → self-contained
  interactive HTML` using the **same** ECharts option builder (no LLM). Charts stay
  interactive offline (ECharts via CDN; `echarts-gl` added only when a 3D chart is present).
- **Live block-by-block rendering** while streaming.
- **Clickable action buttons** that re-prompt (visuals-only).

## Part A (for context)
A per-user Thompson-Sampling bandit (`backend/engine.py`) selects a text strategy from
`strategies.json` using a context feature vector (`x ∈ ℝ¹⁰`). It learns from explicit
(👍/👎 → `/api/rate`) and implicit (next-message sentiment) reward. It shapes only the
`<RESPONSE>` text — never the widget.
