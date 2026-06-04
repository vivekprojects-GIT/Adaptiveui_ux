/**
 * Deterministic widget → standalone HTML exporter.
 *
 * NO LLM involved. Given the same widget JSON the renderer uses, this emits a
 * self-contained .html file:
 *   - text/kpi_row/table/stat_card/progress/badge_row/image/action_row → plain HTML + inline CSS
 *   - chart → a <div> + the ECharts `option` (built by the SHARED buildEChartsOption)
 *             initialized via the ECharts CDN, so charts stay fully interactive offline.
 *
 * Functions in the option (custom formatters) are dropped during serialization;
 * ECharts falls back to sensible defaults, keeping the export robust.
 */
import { buildEChartsOption, isGlKind, type ChartSpec } from '@/lib/echartsOption'

const ECHARTS_CDN = 'https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js'
const ECHARTS_GL_CDN = 'https://cdn.jsdelivr.net/npm/echarts-gl@2/dist/echarts-gl.min.js'

type Block = Record<string, unknown> & { type?: string }

function esc(v: unknown): string {
  return String(v ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

/** Serialize an ECharts option to a JS literal, dropping function values. */
function serializeOption(option: unknown): string {
  return JSON.stringify(option, (_k, v) => (typeof v === 'function' ? undefined : v))
}

function parseLayout(raw: string): Block[] {
  let s = String(raw || '').trim()
  if (!s) return []
  const fence = s.match(/```(?:json)?\s*([\s\S]*?)```/i)
  if (fence) s = fence[1].trim()
  let parsed: unknown
  try {
    parsed = JSON.parse(s)
  } catch {
    const i = s.indexOf('{')
    const j = s.lastIndexOf('}')
    if (i < 0 || j <= i) return []
    try {
      parsed = JSON.parse(s.slice(i, j + 1))
    } catch {
      return []
    }
  }
  let layout: unknown
  if (Array.isArray(parsed)) layout = parsed
  else if (parsed && typeof parsed === 'object') {
    const o = parsed as Record<string, unknown>
    layout = o.layout ?? o.blocks ?? o.components
    if (!Array.isArray(layout) && typeof o.type === 'string') layout = [o]
  }
  return Array.isArray(layout) ? (layout as Block[]) : []
}

const TONE_HEX: Record<string, string> = {
  positive: '#16a34a',
  success: '#16a34a',
  negative: '#dc2626',
  danger: '#dc2626',
  warning: '#d97706',
  info: '#2563eb',
  neutral: '#475569',
  default: '#475569',
}
const BAR_HEX: Record<string, string> = {
  emerald: '#10b981',
  amber: '#f59e0b',
  red: '#ef4444',
  cyan: '#06b6d4',
  indigo: '#6366f1',
}

type ChartEntry = { id: string; option: string; gl: boolean }

/** Render one block to HTML. Charts push their option into `charts` and return a placeholder div. */
function renderBlock(block: Block, idx: number, charts: ChartEntry[]): string {
  const type = String(block.type || '').toLowerCase()

  if (type === 'text') {
    return `<p class="ws-text">${esc((block as { content?: string }).content)}</p>`
  }

  if (type === 'kpi_row') {
    const items = (block.items as { label?: string; value?: string; tone?: string }[]) || []
    const cells = items
      .map(
        (it) =>
          `<div class="ws-kpi"><div class="ws-kpi-val" style="color:${TONE_HEX[it.tone || 'neutral'] || '#111'}">${esc(it.value)}</div><div class="ws-kpi-lbl">${esc(it.label)}</div></div>`,
      )
      .join('')
    return `<div class="ws-kpi-row">${cells}</div>`
  }

  if (type === 'stat_card') {
    const items = (block.items as { label?: string; value?: string; subtext?: string; tone?: string }[]) || []
    const cells = items
      .map(
        (it) =>
          `<div class="ws-stat"><div class="ws-stat-lbl">${esc(it.label)}</div><div class="ws-stat-val" style="color:${TONE_HEX[it.tone || 'default'] || '#111'}">${esc(it.value)}</div>${it.subtext ? `<div class="ws-stat-sub">${esc(it.subtext)}</div>` : ''}</div>`,
      )
      .join('')
    return `<div class="ws-stat-grid">${cells}</div>`
  }

  if (type === 'progress') {
    const items = (block.items as { label?: string; value?: number; max?: number; tone?: string }[]) || []
    const rows = items
      .map((it) => {
        const max = Number(it.max) || 100
        const pct = Math.max(0, Math.min(100, ((Number(it.value) || 0) / max) * 100))
        const color = BAR_HEX[it.tone || 'indigo'] || '#6366f1'
        const showPct = !it.max || it.max === 100
        return `<div class="ws-prog"><div class="ws-prog-head"><span>${esc(it.label)}</span><span>${esc(it.value)}${showPct ? '%' : ''}</span></div><div class="ws-prog-track"><div class="ws-prog-fill" style="width:${pct}%;background:${color}"></div></div></div>`
      })
      .join('')
    return `<div class="ws-prog-list">${rows}</div>`
  }

  if (type === 'badge_row') {
    const items = (block.items as { label?: string; tone?: string }[]) || []
    const pills = items
      .map((it) => {
        const c = TONE_HEX[it.tone || 'default'] || '#475569'
        return `<span class="ws-badge" style="color:${c};border-color:${c}33;background:${c}14">${esc(it.label)}</span>`
      })
      .join('')
    return `<div class="ws-badge-row">${pills}</div>`
  }

  if (type === 'table') {
    const cols = (block.columns as unknown[]) || []
    const rows = (block.rows as unknown[][]) || []
    const head = `<tr>${cols.map((c) => `<th>${esc(c)}</th>`).join('')}</tr>`
    const body = rows.map((r) => `<tr>${(Array.isArray(r) ? r : [r]).map((c) => `<td>${esc(c)}</td>`).join('')}</tr>`).join('')
    const title = (block as { title?: string }).title
    return `${title ? `<div class="ws-h">${esc(title)}</div>` : ''}<table class="ws-table"><thead>${head}</thead><tbody>${body}</tbody></table>`
  }

  if (type === 'action_row') {
    const buttons = (block.buttons as { label?: string }[]) || []
    const btns = buttons.map((b) => `<button class="ws-action" type="button" disabled>${esc(b.label)}</button>`).join('')
    return `<div class="ws-action-row">${btns}</div>`
  }

  if (type === 'image') {
    const b = block as { src?: string; alt?: string; caption?: string; fit?: string }
    if (!b.src) return ''
    const fit = b.fit === 'cover' ? 'cover' : 'contain'
    return `<figure class="ws-fig"><img src="${esc(b.src)}" alt="${esc(b.alt)}" style="object-fit:${fit}" />${b.caption ? `<figcaption>${esc(b.caption)}</figcaption>` : ''}</figure>`
  }

  if (type === 'chart') {
    const chart = (block.chart as ChartSpec) || {}
    const title = (block as { title?: string }).title || ''
    const id = `ws-chart-${idx}`
    const gl = isGlKind(chart.kind)
    try {
      const option = buildEChartsOption(chart, title, { dark: false })
      charts.push({ id, option: serializeOption(option), gl })
      const cls = gl ? 'ws-chart ws-chart-3d' : 'ws-chart'
      return `<div class="ws-chart-wrap">${title ? `<div class="ws-h">${esc(title)}</div>` : ''}<div id="${id}" class="${cls}"></div></div>`
    } catch {
      return `<div class="ws-note">Chart could not be exported.</div>`
    }
  }

  // Unknown block type — show nothing (kept consistent with the renderer's fail-safe).
  return ''
}

const STYLES = `
*{box-sizing:border-box}
body{margin:0;background:#f7f8fa;color:#111318;font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:24px}
.ws-doc{max-width:880px;margin:0 auto;display:flex;flex-direction:column;gap:14px}
.ws-card{background:#fff;border:1px solid #e6e8ec;border-radius:14px;padding:16px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.ws-text{margin:0;color:#2b2f38}
.ws-h{font-weight:600;font-size:13px;margin-bottom:8px}
.ws-kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px}
.ws-kpi{border:1px solid #eef0f3;border-radius:10px;padding:10px 12px}
.ws-kpi-val{font-size:20px;font-weight:700}
.ws-kpi-lbl{font-size:12px;color:#6b7280;margin-top:2px}
.ws-stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.ws-stat{border:1px solid #eef0f3;border-radius:12px;padding:14px}
.ws-stat-lbl{font-size:12px;color:#6b7280}
.ws-stat-val{font-size:24px;font-weight:700;margin-top:4px}
.ws-stat-sub{font-size:12px;color:#9aa1ad;margin-top:2px}
.ws-prog-list{display:flex;flex-direction:column;gap:12px}
.ws-prog-head{display:flex;justify-content:space-between;font-size:12px;color:#6b7280;margin-bottom:4px}
.ws-prog-track{height:8px;background:#eef0f3;border-radius:99px;overflow:hidden}
.ws-prog-fill{height:100%;border-radius:99px}
.ws-badge-row{display:flex;flex-wrap:wrap;gap:8px}
.ws-badge{font-size:12px;font-weight:500;padding:3px 10px;border-radius:99px;border:1px solid}
.ws-table{width:100%;border-collapse:collapse;font-size:13px}
.ws-table th,.ws-table td{text-align:left;padding:8px 10px;border-bottom:1px solid #eef0f3}
.ws-table th{color:#6b7280;font-weight:600;background:#fafbfc}
.ws-action-row{display:flex;flex-wrap:wrap;gap:8px}
.ws-action{font-size:13px;padding:6px 12px;border:1px solid #d7dae0;border-radius:8px;background:#fff;color:#475569}
.ws-fig{margin:0}
.ws-fig img{width:100%;max-height:420px;border-radius:10px}
.ws-fig figcaption{font-size:12px;color:#9aa1ad;margin-top:6px;text-align:center}
.ws-chart{width:100%;height:380px}
.ws-chart-3d{height:460px}
.ws-note{font-size:12px;color:#9aa1ad}
.ws-footer{text-align:center;font-size:11px;color:#b3b9c4;margin-top:6px}
`

/** Build a complete, self-contained, interactive HTML document from widget JSON. */
export function widgetToHtml(jsonStr: string, title = 'Widget'): string {
  const blocks = parseLayout(jsonStr)
  const charts: ChartEntry[] = []
  const body = blocks
    .map((b, i) => {
      const inner = renderBlock(b, i, charts)
      return inner ? `<section class="ws-card">${inner}</section>` : ''
    })
    .filter(Boolean)
    .join('\n')

  const needsGl = charts.some((c) => c.gl)
  const initScript = charts.length
    ? `<script src="${ECHARTS_CDN}"></script>
${needsGl ? `<script src="${ECHARTS_GL_CDN}"></script>` : ''}
<script>
(function(){
  function init(){
    if(!window.echarts){return;}
    var defs=${JSON.stringify(charts.map((c) => c.id))};
    var opts=[${charts.map((c) => c.option).join(',')}];
    defs.forEach(function(id,i){
      var el=document.getElementById(id);
      if(!el)return;
      var chart=echarts.init(el);
      chart.setOption(opts[i]);
      window.addEventListener('resize',function(){chart.resize();});
    });
  }
  if(document.readyState!=='loading')init();else document.addEventListener('DOMContentLoaded',init);
})();
</script>`
    : ''

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>${esc(title)}</title>
<style>${STYLES}</style>
</head>
<body>
<div class="ws-doc">
${body}
<div class="ws-footer">Exported widget · interactive charts via ECharts</div>
</div>
${initScript}
</body>
</html>`
}
