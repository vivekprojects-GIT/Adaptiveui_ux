/**
 * Shared ECharts option builder — the SINGLE source for chart configuration.
 *
 * Used by:
 *   - WidgetSchemaChart.vue (live interactive render)
 *   - exportWidgetHtml.ts    (standalone HTML export)
 *
 * Pure functions only (no Vue, no DOM): given the chart spec + a `dark` flag,
 * return the ECharts `option`. This guarantees the exported HTML matches what
 * the user sees on screen.
 */
import type { EChartsOption } from 'echarts'

export type ChartSeries = { name?: string; color?: string; kind?: string; values?: (number | number[])[] }

export type TreeNodeSpec = { name?: string; label?: string; value?: number; children?: TreeNodeSpec[] }

export type ChartSpec = {
  kind?: string
  x_label?: string
  y_label?: string
  max?: number
  x_labels?: string[]
  y_labels?: string[]
  categories?: string[]
  labels?: string[]
  matrix?: number[][]
  series?: ChartSeries[]
  x_categories?: string[]
  items?: { label?: string; name?: string; value?: number }[]
  // tree / mindmap / org: a single hierarchical root
  root?: TreeNodeSpec
  candles?: (number | string)[][]
  boxes?: (number | string)[][]
  nodes?: ({ name?: string } | string)[]
  links?: { source?: string | number; target?: string | number; value?: number }[]
}

const COLOR_MAP: Record<string, string> = {
  blue: '#3b82f6',
  orange: '#f97316',
  green: '#22c55e',
  red: '#ef4444',
  purple: '#a855f7',
  teal: '#14b8a6',
  yellow: '#eab308',
  pink: '#ec4899',
  indigo: '#4f46e5',
  cyan: '#0891b2',
  gray: '#64748b',
}
const PALETTE = ['#3b82f6', '#f97316', '#22c55e', '#ef4444', '#a855f7', '#14b8a6', '#eab308', '#ec4899']
const POSITIVE = '#16a34a'
const NEGATIVE = '#dc2626'

function resolveColor(c?: string, i = 0): string {
  if (!c) return PALETTE[i % PALETTE.length]
  return COLOR_MAP[c.toLowerCase()] || c
}

function toNum(v: unknown): number | null {
  if (typeof v === 'number') return Number.isFinite(v) ? v : null
  if (typeof v === 'string') {
    const m = v.replace(/,/g, '').match(/-?\d+(\.\d+)?/)
    return m ? parseFloat(m[0]) : null
  }
  return null
}

function chartKind(chart: ChartSpec): string {
  return String(chart?.kind || 'line').toLowerCase()
}

function cartesianSeries(chart: ChartSpec) {
  const raw = chart?.series || []
  return raw
    .map((s, i) => {
      const vals = Array.isArray(s.values) ? s.values : []
      let pts: [number, number | null][] = []
      if (vals.length && !Array.isArray(vals[0])) {
        pts = (vals as unknown[]).map((v, j) => [j, toNum(v)])
      } else {
        pts = (vals as unknown[])
          .filter((p) => Array.isArray(p) && (p as unknown[]).length >= 2)
          .map((p) => [(toNum((p as unknown[])[0]) ?? 0), toNum((p as unknown[])[1])] as [number, number | null])
      }
      return { name: s.name || `Series ${i + 1}`, color: resolveColor(s.color, i), kind: s.kind, points: pts }
    })
    .filter((s) => s.points.some((p) => p[1] != null))
}

function heatmapData(chart: ChartSpec) {
  const xs = chart?.x_labels || chart?.categories || chart?.labels || []
  const ys = chart?.y_labels || chart?.categories || chart?.labels || []
  const m = chart?.matrix
  const data: [number, number, number][] = []
  let vmin = Infinity
  let vmax = -Infinity
  if (Array.isArray(m)) {
    for (let r = 0; r < m.length; r++) {
      const row = Array.isArray(m[r]) ? m[r] : []
      for (let c = 0; c < row.length; c++) {
        const v = Number(row[c])
        if (!Number.isFinite(v)) continue
        data.push([c, r, v])
        if (v < vmin) vmin = v
        if (v > vmax) vmax = v
      }
    }
  }
  if (!Number.isFinite(vmin)) {
    vmin = 0
    vmax = 1
  }
  return { xs, ys, data, vmin, vmax }
}

function pieData(chart: ChartSpec) {
  const items = chart?.items || []
  if (items.length) {
    return items.map((it, i) => ({ name: it.label || it.name || `Item ${i + 1}`, value: toNum(it.value) ?? 0 }))
  }
  const s = (chart?.series || [])[0]
  if (s?.values?.length) {
    const cats = chart?.x_categories
    return s.values.map((p, i) => ({
      name: cats?.[i] ?? String(i + 1),
      value: toNum(Array.isArray(p) ? p[1] : p) ?? 0,
    }))
  }
  return []
}

function radarData(chart: ChartSpec) {
  const cats = chart?.x_categories || chart?.categories || chart?.labels || []
  const series = cartesianSeries(chart)
  const allVals = series.flatMap((s) => s.points.map((p) => p[1])).filter((v): v is number => v != null)
  const max = allVals.length ? Math.max(...allVals) * 1.1 : 1
  const names = cats.length ? cats : series[0]?.points.map((_, i) => `#${i + 1}`) || []
  const indicator = names.map((name) => ({ name, max }))
  const data = series.map((s) => ({ name: s.name, value: s.points.map((p) => p[1] ?? 0) }))
  return { indicator, data }
}

function gaugeData(chart: ChartSpec, title: string) {
  const d = pieData(chart)
  let value = 0
  let name = title || 'Value'
  if (d.length) {
    value = d[0].value
    name = d[0].name
  } else {
    const s = cartesianSeries(chart)[0]
    const last = s?.points[s.points.length - 1]
    if (last && last[1] != null) value = last[1]
  }
  const max = Number(chart?.max) || (value >= 0 && value <= 100 ? 100 : value * 1.3 || 1)
  return { value, name, max }
}

/** 3D series: each series' values are [x, y, z] triples. */
function series3d(chart: ChartSpec) {
  return (chart?.series || [])
    .map((s, i) => ({
      name: s.name || `Series ${i + 1}`,
      color: resolveColor(s.color, i),
      data: (Array.isArray(s.values) ? s.values : [])
        .filter((v) => Array.isArray(v) && (v as number[]).length >= 3)
        .map((v) => (v as number[]).slice(0, 3).map((x) => toNum(x) ?? 0)),
    }))
    .filter((s) => s.data.length > 0)
}

/** Normalize a hierarchical node (tree/mindmap/org): ensure `name`, recurse `children`. */
function normalizeTreeNode(n: TreeNodeSpec | undefined, depth = 0): { name: string; value?: number; children?: any[] } | null {
  if (!n || typeof n !== 'object' || depth > 8) return null
  const name = String(n.name || n.label || '').trim()
  const kids = Array.isArray(n.children)
    ? n.children.map((c) => normalizeTreeNode(c, depth + 1)).filter(Boolean)
    : []
  if (!name && kids.length === 0) return null
  const out: { name: string; value?: number; children?: any[] } = { name: name || '·' }
  if (typeof n.value === 'number') out.value = n.value
  if (kids.length) out.children = kids
  return out
}

function treeRoot(chart: ChartSpec) {
  const c = chart as ChartSpec & { tree?: TreeNodeSpec; data?: unknown; children?: TreeNodeSpec[]; name?: string }
  // Accept common variants the model may emit: root | tree | data | a top-level node | items as children.
  let raw: TreeNodeSpec | undefined = c.root || c.tree
  if (!raw && c.data && typeof c.data === 'object') raw = (Array.isArray(c.data) ? c.data[0] : c.data) as TreeNodeSpec
  if (!raw && (c.name || c.children)) raw = { name: c.name, children: c.children }
  if (!raw && Array.isArray(chart?.items) && chart.items.length) {
    raw = { name: '', children: chart.items.map((it) => ({ name: it.label || it.name, value: it.value })) }
  }
  return normalizeTreeNode(raw)
}

const GL_KINDS = new Set(['scatter3d', 'bar3d', 'line3d'])
export function isGlKind(kind?: string): boolean {
  return GL_KINDS.has(String(kind || '').toLowerCase())
}

export function chartHasRenderableData(chart: ChartSpec): boolean {
  const k = chartKind(chart)
  const c = chart || {}
  if (k === 'heatmap') return heatmapData(chart).data.length > 0
  if (k === 'candlestick') return Array.isArray(c.candles) && c.candles.length > 0
  if (k === 'boxplot') return Array.isArray(c.boxes) && c.boxes.length > 0
  if (k === 'sankey' || k === 'graph') return Array.isArray(c.links) && c.links.length > 0
  if (k === 'treemap' || k === 'sunburst') return (c.items?.length || 0) > 0 || pieData(chart).length > 0
  if (k === 'pie' || k === 'donut' || k === 'funnel' || k === 'waterfall' || k === 'rose') return pieData(chart).length > 0
  if (k === 'tree' || k === 'mindmap' || k === 'org' || k === 'orgchart') return treeRoot(chart) != null
  if (k === 'scatter3d' || k === 'bar3d' || k === 'line3d') return series3d(chart).length > 0
  if (k === 'gauge') return pieData(chart).length > 0 || cartesianSeries(chart).length > 0
  return cartesianSeries(chart).length > 0
}

export function buildEChartsOption(chart: ChartSpec, title = '', opts: { dark?: boolean } = {}): EChartsOption {
  const dark = !!opts.dark
  const tc = dark ? '#8d93aa' : '#5a5f72'
  const k = chartKind(chart)

  if (k === 'heatmap') {
    const { xs, ys, data, vmin, vmax } = heatmapData(chart)
    const absMax = Math.max(Math.abs(vmin), Math.abs(vmax)) || 1
    const symmetric = vmin < 0
    return {
      animation: true,
      backgroundColor: 'transparent',
      textStyle: { color: tc, fontSize: 11 },
      tooltip: {
        position: 'top',
        formatter: (p: any) =>
          `${ys[p.value?.[1]] ?? ''} × ${xs[p.value?.[0]] ?? ''}: ${Number(p.value?.[2]).toFixed(2)}`,
      },
      grid: { left: 60, right: 20, top: 16, bottom: 64, containLabel: true },
      xAxis: {
        type: 'category',
        data: xs,
        splitArea: { show: true },
        axisLabel: { fontSize: 10, rotate: xs.length > 5 ? 30 : 0 },
      },
      yAxis: { type: 'category', data: ys, splitArea: { show: true }, axisLabel: { fontSize: 10 } },
      visualMap: {
        min: symmetric ? -absMax : vmin,
        max: symmetric ? absMax : vmax,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        itemHeight: 80,
        textStyle: { color: tc, fontSize: 10 },
        inRange: { color: symmetric ? ['#ef4444', '#f8fafc', '#3b82f6'] : ['#dbeafe', '#3b82f6', '#1e3a8a'] },
      },
      series: [
        {
          type: 'heatmap',
          data,
          label: {
            show: xs.length <= 10 && ys.length <= 10,
            fontSize: 10,
            formatter: (p: any) => (typeof p.value?.[2] === 'number' ? p.value[2].toFixed(2) : ''),
          },
          emphasis: { itemStyle: { shadowBlur: 8, shadowColor: 'rgba(0,0,0,0.3)' } },
        },
      ],
    }
  }

  if (k === 'pie' || k === 'donut') {
    const d = pieData(chart)
    return {
      animation: true,
      backgroundColor: 'transparent',
      textStyle: { color: tc, fontSize: 11 },
      tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      legend: { bottom: 0, textStyle: { color: tc, fontSize: 10 } },
      series: [
        {
          type: 'pie',
          radius: k === 'donut' ? ['42%', '70%'] : '65%',
          center: ['50%', '46%'],
          data: d.map((x, i) => ({ ...x, itemStyle: { color: PALETTE[i % PALETTE.length] } })),
          label: { fontSize: 10, color: tc },
        },
      ],
    }
  }

  if (k === 'funnel') {
    const d = pieData(chart)
    return {
      animation: true,
      backgroundColor: 'transparent',
      textStyle: { color: tc, fontSize: 11 },
      tooltip: { trigger: 'item', formatter: '{b}: {c}' },
      legend: { bottom: 0, textStyle: { color: tc, fontSize: 10 } },
      series: [
        {
          type: 'funnel',
          left: '10%',
          right: '10%',
          top: 20,
          bottom: 40,
          sort: 'descending',
          gap: 2,
          label: { show: true, position: 'inside', fontSize: 10 },
          data: d.map((x, i) => ({ ...x, itemStyle: { color: PALETTE[i % PALETTE.length] } })),
        },
      ],
    }
  }

  if (k === 'gauge') {
    const g = gaugeData(chart, title)
    return {
      animation: true,
      backgroundColor: 'transparent',
      series: [
        {
          type: 'gauge',
          min: 0,
          max: g.max,
          progress: { show: true, width: 14 },
          axisLine: { lineStyle: { width: 14 } },
          axisLabel: { fontSize: 9, color: tc },
          detail: { valueAnimation: true, fontSize: 22, color: tc, formatter: '{value}' },
          data: [{ value: Number(g.value.toFixed(2)), name: g.name }],
          title: { fontSize: 11, color: tc },
        },
      ],
    }
  }

  if (k === 'radar') {
    const { indicator, data } = radarData(chart)
    return {
      animation: true,
      backgroundColor: 'transparent',
      textStyle: { color: tc, fontSize: 11 },
      tooltip: { trigger: 'item' },
      legend: data.length > 1 ? { bottom: 0, textStyle: { color: tc, fontSize: 10 } } : undefined,
      radar: { indicator, axisName: { fontSize: 10, color: tc }, splitLine: { lineStyle: { opacity: 0.3 } } },
      series: [
        {
          type: 'radar',
          data: data.map((s, i) => ({
            ...s,
            areaStyle: { opacity: 0.12, color: PALETTE[i % PALETTE.length] },
            lineStyle: { color: PALETTE[i % PALETTE.length] },
            itemStyle: { color: PALETTE[i % PALETTE.length] },
          })),
        },
      ],
    }
  }

  if (k === 'candlestick') {
    const cats = chart?.x_categories || []
    const data = (chart?.candles || []).map((c) => (Array.isArray(c) ? c.slice(0, 4).map((v) => toNum(v) ?? 0) : []))
    return {
      backgroundColor: 'transparent',
      textStyle: { color: tc, fontSize: 11 },
      tooltip: { trigger: 'axis' },
      grid: { left: 48, right: 16, top: 20, bottom: 56, containLabel: true },
      xAxis: { type: 'category', data: cats, axisLabel: { fontSize: 10 } },
      yAxis: { type: 'value', scale: true, splitLine: { lineStyle: { opacity: 0.2 } } },
      dataZoom: [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 8 }],
      series: [
        {
          type: 'candlestick',
          data,
          itemStyle: { color: POSITIVE, color0: NEGATIVE, borderColor: POSITIVE, borderColor0: NEGATIVE },
        },
      ],
    }
  }

  if (k === 'boxplot') {
    const cats = chart?.x_categories || []
    const data = (chart?.boxes || []).map((b) => (Array.isArray(b) ? b.slice(0, 5).map((v) => toNum(v) ?? 0) : []))
    return {
      backgroundColor: 'transparent',
      textStyle: { color: tc, fontSize: 11 },
      tooltip: { trigger: 'item' },
      grid: { left: 48, right: 16, top: 20, bottom: 40, containLabel: true },
      xAxis: { type: 'category', data: cats, axisLabel: { fontSize: 10 } },
      yAxis: { type: 'value', scale: true, splitLine: { lineStyle: { opacity: 0.2 } } },
      series: [{ type: 'boxplot', data, itemStyle: { color: 'rgba(59,130,246,0.25)', borderColor: '#3b82f6' } }],
    }
  }

  if (k === 'treemap' || k === 'sunburst') {
    const src = chart?.items?.length
      ? chart.items.map((it, i) => ({ name: it.label || it.name || `Item ${i + 1}`, value: toNum(it.value) ?? 0 }))
      : pieData(chart)
    const d = src.map((x, i) => ({ ...x, itemStyle: { color: PALETTE[i % PALETTE.length] } }))
    return {
      backgroundColor: 'transparent',
      textStyle: { color: tc, fontSize: 11 },
      tooltip: { trigger: 'item', formatter: '{b}: {c}' },
      series: [
        k === 'sunburst'
          ? { type: 'sunburst', data: d, radius: [0, '92%'], label: { fontSize: 10 } }
          : { type: 'treemap', data: d, breadcrumb: { show: false }, roam: false, label: { fontSize: 11 } },
      ],
    }
  }

  if (k === 'sankey' || k === 'graph') {
    const nodes = (chart?.nodes || []).map((n) => ({ name: typeof n === 'string' ? n : n.name || '' }))
    const links = (chart?.links || []).map((l) => ({ source: l.source, target: l.target, value: toNum(l.value) ?? 1 }))
    const named = new Set(nodes.map((n) => n.name))
    for (const l of links) {
      for (const e of [l.source, l.target]) {
        const s = String(e)
        if (e != null && !named.has(s)) {
          nodes.push({ name: s })
          named.add(s)
        }
      }
    }
    return {
      backgroundColor: 'transparent',
      textStyle: { color: tc, fontSize: 11 },
      tooltip: { trigger: 'item' },
      series: [
        k === 'graph'
          ? { type: 'graph', layout: 'force', roam: true, data: nodes, links, label: { show: true, fontSize: 10 }, force: { repulsion: 140 } }
          : { type: 'sankey', data: nodes, links, label: { fontSize: 10, color: tc }, emphasis: { focus: 'adjacency' }, lineStyle: { color: 'gradient', opacity: 0.5 } },
      ],
    }
  }

  if (k === 'waterfall') {
    const pts = pieData(chart)
    const cats = pts.map((p) => p.name)
    const base: (number | string)[] = []
    const inc: (number | string)[] = []
    const dec: (number | string)[] = []
    let run = 0
    for (const p of pts) {
      const v = Number(p.value) || 0
      if (v >= 0) {
        base.push(run)
        inc.push(v)
        dec.push('-')
      } else {
        base.push(run + v)
        inc.push('-')
        dec.push(-v)
      }
      run += v
    }
    return {
      backgroundColor: 'transparent',
      textStyle: { color: tc, fontSize: 11 },
      tooltip: { trigger: 'axis' },
      grid: { left: 48, right: 16, top: 20, bottom: 40, containLabel: true },
      xAxis: { type: 'category', data: cats, axisLabel: { fontSize: 10, rotate: cats.length > 6 ? 30 : 0 } },
      yAxis: { type: 'value', splitLine: { lineStyle: { opacity: 0.2 } } },
      series: [
        { type: 'bar', stack: 'wf', itemStyle: { color: 'transparent' }, emphasis: { itemStyle: { color: 'transparent' } }, data: base },
        { type: 'bar', stack: 'wf', name: 'Increase', itemStyle: { color: POSITIVE }, data: inc },
        { type: 'bar', stack: 'wf', name: 'Decrease', itemStyle: { color: NEGATIVE }, data: dec },
      ],
    }
  }

  if (k === 'bubble') {
    const eb = (chart?.series || []).map((s, i) => ({
      name: s.name || `Series ${i + 1}`,
      type: 'scatter' as const,
      data: (Array.isArray(s.values) ? s.values : [])
        .filter((v) => Array.isArray(v) && (v as number[]).length >= 2)
        .map((v) => (v as number[]).map((x) => toNum(x) ?? 0)),
      symbolSize: (val: number[]) => Math.max(8, Math.sqrt(Math.abs(Number(val?.[2]) || 1)) * 5),
      itemStyle: { color: PALETTE[i % PALETTE.length], opacity: 0.7 },
    }))
    return {
      backgroundColor: 'transparent',
      textStyle: { color: tc, fontSize: 11 },
      tooltip: { trigger: 'item' },
      legend: eb.length > 1 ? { bottom: 0, textStyle: { color: tc, fontSize: 10 } } : undefined,
      grid: { left: 48, right: 24, top: 20, bottom: 40, containLabel: true },
      xAxis: { type: 'value', name: chart?.x_label || '', nameLocation: 'middle', nameGap: 26, splitLine: { lineStyle: { opacity: 0.2 } } },
      yAxis: { type: 'value', name: chart?.y_label || '', nameLocation: 'middle', nameGap: 36, splitLine: { lineStyle: { opacity: 0.2 } } },
      series: eb,
    }
  }

  if (k === 'rose') {
    // Nightingale rose ("pizza") — pie with variable-radius slices.
    const d = pieData(chart)
    return {
      animation: true,
      backgroundColor: 'transparent',
      textStyle: { color: tc, fontSize: 11 },
      tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      legend: { bottom: 0, textStyle: { color: tc, fontSize: 10 } },
      series: [
        {
          type: 'pie',
          radius: ['18%', '72%'],
          center: ['50%', '46%'],
          roseType: 'area',
          itemStyle: { borderRadius: 4 },
          data: d.map((x, i) => ({ ...x, itemStyle: { color: PALETTE[i % PALETTE.length] } })),
          label: { fontSize: 10, color: tc },
        },
      ],
    }
  }

  if (k === 'polar') {
    // Polar bar — categorical bars wrapped around a circle.
    const s = cartesianSeries(chart)
    const cats = chart?.x_categories || s[0]?.points.map((_, i) => `#${i + 1}`) || []
    return {
      animation: true,
      backgroundColor: 'transparent',
      textStyle: { color: tc, fontSize: 11 },
      tooltip: { trigger: 'item' },
      legend: s.length > 1 ? { bottom: 0, textStyle: { color: tc, fontSize: 10 } } : undefined,
      polar: { radius: ['12%', '72%'] },
      angleAxis: { type: 'category', data: cats, axisLabel: { fontSize: 10 } },
      radiusAxis: { axisLabel: { fontSize: 9 } },
      series: s.map((ss, i) => ({
        type: 'bar',
        coordinateSystem: 'polar',
        name: ss.name,
        data: ss.points.map((p) => p[1]),
        itemStyle: { color: ss.color || PALETTE[i % PALETTE.length] },
      })),
    } as EChartsOption
  }

  if (k === 'parallel') {
    // Parallel coordinates — one polyline per series across the x_categories dimensions.
    const s = cartesianSeries(chart)
    const dims = chart?.x_categories || s[0]?.points.map((_, i) => `Dim ${i + 1}`) || []
    return {
      animation: true,
      backgroundColor: 'transparent',
      textStyle: { color: tc, fontSize: 11 },
      tooltip: {},
      legend: s.length > 1 ? { bottom: 0, textStyle: { color: tc, fontSize: 10 } } : undefined,
      parallelAxis: dims.map((name, i) => ({ dim: i, name, nameTextStyle: { fontSize: 10 } })),
      parallel: { left: 60, right: 40, top: 30, bottom: 40 },
      series: s.map((ss, i) => ({
        type: 'parallel',
        name: ss.name,
        lineStyle: { width: 2, opacity: 0.6, color: ss.color || PALETTE[i % PALETTE.length] },
        data: [ss.points.map((p) => p[1] ?? 0)],
      })),
    } as EChartsOption
  }

  if (k === 'themeriver') {
    // Streamgraph — series.values aligned to x_categories → [time, value, name] triples.
    const cats = chart?.x_categories || []
    const data: [string, number, string][] = []
    for (const s of chart?.series || []) {
      const name = s.name || 'series'
      const vals = Array.isArray(s.values) ? s.values : []
      vals.forEach((v, i) => {
        const t = cats[i] ?? String(i)
        data.push([t, toNum(Array.isArray(v) ? v[1] : v) ?? 0, name])
      })
    }
    return {
      backgroundColor: 'transparent',
      textStyle: { color: tc, fontSize: 11 },
      tooltip: { trigger: 'axis', axisPointer: { type: 'line' } },
      legend: { bottom: 0, textStyle: { color: tc, fontSize: 10 } },
      singleAxis: { type: 'category', data: cats, top: 20, bottom: 56, axisLabel: { fontSize: 10 } },
      series: [{ type: 'themeRiver', data, label: { show: false }, emphasis: { focus: 'self' } }],
    } as EChartsOption
  }

  if (k === 'scatter3d' || k === 'bar3d' || k === 'line3d') {
    const s3 = series3d(chart)
    const glType = k === 'bar3d' ? 'bar3D' : k === 'line3d' ? 'line3D' : 'scatter3D'
    return {
      backgroundColor: 'transparent',
      tooltip: {},
      xAxis3D: { type: 'value', name: chart?.x_label || 'X', nameTextStyle: { color: tc } },
      yAxis3D: { type: 'value', name: chart?.y_label || 'Y', nameTextStyle: { color: tc } },
      zAxis3D: { type: 'value', name: 'Z', nameTextStyle: { color: tc } },
      grid3D: {
        boxWidth: 100,
        boxDepth: 100,
        axisLabel: { textStyle: { color: tc, fontSize: 9 } },
        viewControl: { autoRotate: false, distance: 220 },
        light: { main: { intensity: 1.2 }, ambient: { intensity: 0.3 } },
      },
      series: s3.map((ss, i) => ({
        type: glType,
        name: ss.name,
        data: ss.data,
        symbolSize: 10,
        itemStyle: { color: ss.color || PALETTE[i % PALETTE.length], opacity: glType === 'scatter3D' ? 0.85 : 1 },
        shading: glType === 'bar3D' ? 'lambert' : undefined,
        lineStyle: glType === 'line3D' ? { width: 3 } : undefined,
      })),
    } as unknown as EChartsOption
  }

  if (k === 'tree' || k === 'mindmap' || k === 'org' || k === 'orgchart') {
    const root = treeRoot(chart) || { name: '·' }
    const radial = k === 'mindmap'
    const topDown = k === 'org' || k === 'orgchart'
    return {
      backgroundColor: 'transparent',
      textStyle: { color: tc, fontSize: 11 },
      tooltip: { trigger: 'item', triggerOn: 'mousemove' },
      series: [
        {
          type: 'tree',
          data: [root],
          layout: radial ? 'radial' : 'orthogonal',
          orient: topDown ? 'TB' : 'LR',
          roam: true,
          symbol: 'circle',
          symbolSize: 9,
          itemStyle: { color: PALETTE[0] },
          lineStyle: { color: dark ? 'rgba(255,255,255,0.25)' : 'rgba(0,0,0,0.2)', width: 1.2, curveness: radial ? 0.3 : 0.5 },
          label: {
            position: radial ? 'right' : topDown ? 'top' : 'left',
            verticalAlign: 'middle',
            align: radial ? 'left' : topDown ? 'center' : 'right',
            fontSize: 11,
            color: tc,
          },
          leaves: { label: { position: topDown ? 'bottom' : 'right', align: topDown ? 'center' : 'left' } },
          expandAndCollapse: true,
          initialTreeDepth: -1,
          animationDuration: 600,
        },
      ],
    } as EChartsOption
  }

  // cartesian family: line / bar / hbar / area / scatter / stacked / combo / histogram / timeseries
  const series = cartesianSeries(chart)
  const isHBar = k === 'hbar' || k === 'horizontal-bar' || k === 'horizontal_bar'
  const isStacked = k === 'stacked' || k === 'stacked-bar' || k === 'stacked_bar'
  const isCombo = k === 'combo'
  const isBar = k === 'bar' || k === 'histogram' || isStacked || isHBar
  const isScatter = k === 'scatter' || k === 'bubble'
  const isArea = k === 'area' || k === 'timeseries'

  const cats = chart?.x_categories
  const useCat = Array.isArray(cats) && cats.length > 0

  const eSeries = series.map((s, i) => {
    const seriesType = (
      isCombo ? (s.kind === 'line' ? 'line' : s.kind === 'bar' ? 'bar' : i === 0 ? 'bar' : 'line')
      : isBar ? 'bar'
      : isScatter ? 'scatter'
      : 'line'
    ) as 'bar' | 'scatter' | 'line'
    const asLine = seriesType === 'line'
    return {
      name: s.name,
      type: seriesType,
      data: useCat ? s.points.map((p) => p[1]) : s.points,
      stack: isStacked ? 'total' : undefined,
      smooth: asLine,
      showSymbol: seriesType === 'scatter' || asLine,
      symbolSize: seriesType === 'scatter' ? 10 : 6,
      itemStyle: { color: s.color },
      areaStyle: isArea && asLine ? { opacity: 0.18, color: s.color } : undefined,
      lineStyle: asLine ? { width: 2 } : undefined,
      emphasis: { focus: 'series' as const },
      animationDuration: 900,
      animationEasing: 'cubicOut' as const,
    }
  })

  const legendNames = series.map((s) => s.name)
  const valueAxis = {
    type: 'value' as const,
    name: (isHBar ? chart?.x_label : chart?.y_label) || '',
    nameLocation: 'middle' as const,
    nameGap: 36,
    splitLine: { show: true, lineStyle: { opacity: 0.2 } },
  }
  const catAxis = useCat
    ? {
        type: 'category' as const,
        data: cats,
        name: (isHBar ? chart?.y_label : chart?.x_label) || '',
        nameLocation: 'middle' as const,
        nameGap: 30,
        axisLabel: { fontSize: 10, rotate: !isHBar && (cats as string[]).length > 6 ? 30 : 0 },
      }
    : {
        type: 'value' as const,
        name: (isHBar ? chart?.y_label : chart?.x_label) || '',
        nameLocation: 'middle' as const,
        nameGap: 28,
        splitLine: { show: true, lineStyle: { opacity: 0.2 } },
      }

  // Interactive timeseries: zoom/pan for time-oriented line/area, especially with many points.
  const wantsZoom =
    !isHBar &&
    !isScatter &&
    (k === 'timeseries' || ((k === 'line' || k === 'area') && useCat && (cats as string[]).length > 10))

  return {
    animation: true,
    animationDuration: 1100,
    backgroundColor: 'transparent',
    textStyle: { color: tc, fontSize: 11 },
    grid: { left: 48, right: 24, top: 36, bottom: wantsZoom ? 64 : 40, containLabel: true },
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    legend: legendNames.length > 1 ? { data: legendNames, bottom: 0, textStyle: { color: tc, fontSize: 10 } } : undefined,
    dataZoom: wantsZoom ? [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 8 }] : undefined,
    xAxis: isHBar ? valueAxis : catAxis,
    yAxis: isHBar ? catAxis : valueAxis,
    series: eSeries,
  }
}
