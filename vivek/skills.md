# Widget & Chart Skills — High-Level Guidance

This document helps you create professional, high-quality interactive widgets and charts. Follow these principles for every <WIDGET> you generate.

**Token limit:** You have a fixed token budget for response + widget. Keep the text concise when generating visualizations so the widget fits. A complete, working chart is better than long prose with no widget.

**Never describe a widget you don't generate.** If you mention "the dashboard below", "interactive chart", or "explore visually" in your text, you MUST output a complete, non-empty <WIDGET>. Either generate the full widget or do not mention it in the text.

---

## 0. Library Choice

Use **any public CDN library** (cdnjs, jsdelivr, unpkg, cdn.plot.ly) that produces the best visual for the use case. You MUST support ALL chart types data analysts use (see Section 1b).

- **ECharts** — line, bar, scatter, pie, donut, treemap, heatmap, gauge, funnel, candlestick, sankey, waterfall; excellent for BI dashboards
- **Plotly** — high interactivity, 3D, scientific charts, choropleth maps, box plots
- **D3.js** — custom/bespoke visuals, force graphs, maps, histograms, sankey
- **Chart.js** — simple, lightweight: line, bar, pie, radar
- **ApexCharts** — modern, responsive: bar, line, area, pie, donut, radar, gauge
- **Tabulator** — sortable/filterable tables, Gantt-style timelines

Pick the library that creates the most accurate, readable visualization. You may combine libraries (e.g. ECharts for charts + Tabulator for tables).

---

## 1. Chart Type Selection

Choose the right chart for the data shape:

| Data Shape | Best Chart Type | Notes |
|------------|-----------------|-------|
| **Time-series** (dates, trends over time, ≥5 points) | Line or area chart | Add tooltip + subtle dataZoom; ECharts preferred |
| **Categorical ranking** (categories with values) | Horizontal bar chart | Add click-to-filter, optional cross-filter table |
| **Composition / share** (parts of whole) | Stacked bar or 100% stacked | Pie/donut only for 3–5 short categories |
| **Correlation** (x-y pairs) | Scatter plot | Highlight outliers |
| **Distribution** (raw samples) | Histogram | Use bins |
| **Distribution** (summary stats only) | KPI tiles + short explanation | Do NOT invent bins |
| **Hierarchy** (parent-child) | Treemap only when explicit | Otherwise use grouped table |
| **Many series** | Small multiples or series toggles | Do not plot >6 lines by default |
| **Rows > 8 or user asked for breakdown** | Tabulator table | Sort, filter, scan |

---

## 1b. Full Chart Type Reference — All Data Analyst Capabilities

You MUST be able to generate any of these chart types when the use case fits. Use ECharts, Plotly, Chart.js, ApexCharts, or D3 as appropriate.

### Core charts
| Chart Type | Use Case | Library |
|------------|----------|---------|
| **Bar chart** (vertical/horizontal) | Categorical comparisons, rankings | ECharts, Chart.js, ApexCharts |
| **Line chart** | Time-series trends, continuity | ECharts, Plotly, Chart.js |
| **Pie chart** | Composition of whole (3–5 segments) | ECharts, Chart.js |
| **Donut chart** | Composition with center space | ECharts, Chart.js, ApexCharts |
| **Area chart** (stacked/stream) | Cumulative trends, volumes over time | ECharts, Plotly |
| **Scatter plot** | Correlation, x-y relationship, outliers | ECharts, Plotly, D3 |
| **Histogram** | Distribution of numeric variable, frequency | ECharts, Plotly, D3 |
| **Box plot** | Distribution, quartiles, outliers | ECharts, Plotly |
| **Heatmap** | Values across 2 dimensions, correlation matrix | ECharts, Plotly, D3 |
| **Stacked bar** | Composition across categories | ECharts, Chart.js |
| **Treemap** | Hierarchical part-to-whole | ECharts, D3 |
| **Waterfall chart** | Sequential changes (profit, cash flow) | ECharts (custom), Plotly |
| **Gauge / speedometer** | Single KPI vs target | ECharts, ApexCharts |
| **Funnel chart** | Stage progression (sales, conversion) | ECharts, Plotly |
| **Bubble chart** | Scatter with 3rd variable as size | ECharts, Plotly |
| **Radar / spider chart** | Multi-dimensional comparison | ECharts, Chart.js |
| **Candlestick** | Financial OHLC (open, high, low, close) | ECharts, Plotly |
| **Combo chart** | Mixed (e.g. bars + line) | ECharts, ApexCharts |

### Advanced / specialized
| Chart Type | Use Case | Library |
|------------|----------|---------|
| **Sankey diagram** | Flow between stages, allocations | D3, ECharts (sankey) |
| **Choropleth map** | Values by geographic region | Plotly, D3 + topojson |
| **Bullet chart** | Metric vs target, ranges | D3, custom |
| **Gantt chart** | Timeline, project tasks | Tabulator, D3, custom |
| **Spline / smooth line** | Smoother trends | ECharts, Plotly |
| **100% stacked bar** | Part-to-whole across categories | ECharts, Chart.js |
| **Small multiples** | Many series, side-by-side | ECharts grid, D3 |
| **Semi-donut / half donut** | Progress, gauge-style | ECharts, ApexCharts |

### Tables and KPIs
| Type | Use Case | Library |
|------|----------|---------|
| **KPI tiles** | Key metrics at a glance | CSS grid + design system |
| **Sortable/filterable table** | Rows > 8, breakdown, scan | Tabulator |
| **Comparison table** | Pros/cons, options side-by-side | HTML table + design system |
| **Pivot-style table** | Multi-dimensional aggregation | Tabulator, custom |

When data is unavailable, use illustrative/mock data and label clearly (e.g. "Example data", "Mock data").

---

## 2. BI Layout Structure

When warranted, use this layout order:
1. **KPI row** (3–6 tiles) — key metrics at a glance
2. **Controls row** (optional) — sliders, dropdowns, filters
3. **Primary visualization** — main chart or table
4. **Detail table** (optional) — raw or filtered data
5. **Insights** (2–4 bullets) — computed from data, not generic

---

## 3. Educational & Concept Demos

When the user asks to **explain** something (e.g. compound interest, ROI, percentages):

- **ALWAYS** produce an interactive demo:
  - Sliders for parameters (Principal, Rate, Time, etc.)
  - Live formula result that updates on change
  - Chart showing the concept (e.g. balance over time for compound interest)
- **Use illustrative data** — choose reasonable defaults (e.g. P=1000, r=5%, t=10 years)
- **Do NOT ask for date range or data source first** — produce the app immediately
- Label as "Example" if helpful

---

## 4. Interactivity Rules

- **Sliders** → for continuous numeric inputs (rate, amount, time)
- **Dropdowns** → for categorical choices (category, period)
- **Controls must update** KPIs, chart, and table from the SAME filtered dataset
- **Drilldown** → click chart mark / legend / table row → sendPrompt with clicked entity + metric + context
- **sendPrompt** → only when new knowledge is needed; be specific (e.g. "What are the risks of VTI at 0.03% expense ratio?" not "Tell me more")

---

## 5. Visual Polish

- **NO EMOJIS** — never use emojis in widgets, labels, or text. Use plain text only.
- **Neat and clean for any data** — light backgrounds (#F5F5F5, #F2F2F2), clear typography, generous spacing. Minimal, professional layout. No decorative icons or clutter.
- **Dark mode** — detect with `window.matchMedia('(prefers-color-scheme: dark)').matches` and adjust colors
- **Palette** — define a hex array in JS and apply consistently to KPIs and chart series. Use subtle colors (gray, green, blue for data emphasis).
- **Labels** — axis labels, grid lines, legend text must always be visible; set colors explicitly
- **Borders** — use 0.5px solid, never 1px solid
- **No clutter** — avoid plotting too many series; use toggles or small multiples. Only functional elements.

---

## 6. When to Omit a Widget

Return `<WIDGET></WIDGET>` (empty) only for:
- Greetings (hi, hello, hey)
- Acknowledgements (thanks, ok, got it)
- Chit-chat or simple social responses
- Ultra-short answers with no structure or numbers

For prose with no extractable dataset: still generate a widget using illustrative/mock data and label it clearly. Do not omit when a chart, calculator, or table could help.

---

## 7. Data Grounding

- Prefer data from user request/context OR your <RESPONSE>
- **When real data is unavailable**: use illustrative/mock data and label clearly (e.g. "Example data", "Mock data")
- Use meaningful labels (Principal, Rate, Years) — avoid generic placeholders (Item A, Value 1)
