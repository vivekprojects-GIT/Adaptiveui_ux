<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch, computed, nextTick } from 'vue'
import * as echarts from 'echarts'
import { ArrowDownTrayIcon } from '@/components/icons'

type Series = { name?: string; color?: string; kind?: string; values?: (number | number[])[] }

const props = defineProps<{
  title?: string
  chart: {
    kind?: string
    x_label?: string
    y_label?: string
    max?: number
    // heatmap / matrix
    x_labels?: string[]
    y_labels?: string[]
    categories?: string[]
    labels?: string[]
    matrix?: number[][]
    // cartesian (line/bar/area/scatter/hbar/stacked/combo/bubble) + radar
    series?: Series[]
    x_categories?: string[]
    // pie / donut / funnel / gauge / treemap / sunburst
    items?: { label?: string; name?: string; value?: number }[]
    // candlestick: [open, close, low, high] per category
    candles?: (number | string)[][]
    // boxplot: [min, q1, median, q3, max] per category
    boxes?: (number | string)[][]
    // sankey / graph
    nodes?: ({ name?: string } | string)[]
    links?: { source?: string | number; target?: string | number; value?: number }[]
  }
}>()

const rootEl = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null

// Named colors the LLM may use; anything else (e.g. a hex string) passes through.
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
// Fallback palette when the LLM does not specify a color for a series.
const PALETTE = ['#3b82f6', '#f97316', '#22c55e', '#ef4444', '#a855f7', '#14b8a6', '#eab308', '#ec4899']
// Semantic tokens for meaning-bearing series (gains vs. losses, etc.)
const POSITIVE = '#16a34a'
const NEGATIVE = '#dc2626'

// Honor the LLM-chosen color (named or hex); fall back to the palette by index.
function resolveColor(c?: string, i = 0): string {
  if (!c) return PALETTE[i % PALETTE.length]
  return COLOR_MAP[c.toLowerCase()] || c
}

const kind = computed(() => String(props.chart?.kind || 'line').toLowerCase())

function _toNum(v: unknown): number | null {
  if (typeof v === 'number') return Number.isFinite(v) ? v : null
  if (typeof v === 'string') {
    const m = v.replace(/,/g, '').match(/-?\d+(\.\d+)?/) // "$47.5B" → 47.5, "—" → null
    return m ? parseFloat(m[0]) : null
  }
  return null
}

function cartesianSeries() {
  const raw = props.chart?.series || []
  return raw
    .map((s, i) => {
      const vals = Array.isArray(s.values) ? s.values : []
      let pts: [number, number | null][] = []
      if (vals.length && !Array.isArray(vals[0])) {
        // plain values (numbers OR strings like "$47.5B"/"—") aligned to x_categories;
        // keep position (null = gap) so values line up with the categories.
        pts = (vals as unknown[]).map((v, j) => [j, _toNum(v)])
      } else {
        pts = (vals as unknown[])
          .filter((p) => Array.isArray(p) && (p as unknown[]).length >= 2)
          .map((p) => [(_toNum((p as unknown[])[0]) ?? 0), _toNum((p as unknown[])[1])] as [number, number | null])
      }
      return { name: s.name || `Series ${i + 1}`, color: resolveColor(s.color, i), kind: s.kind, points: pts }
    })
    .filter((s) => s.points.some((p) => p[1] != null)) // keep series with ≥1 real value
}

function heatmapData() {
  const xs = props.chart?.x_labels || props.chart?.categories || props.chart?.labels || []
  const ys = props.chart?.y_labels || props.chart?.categories || props.chart?.labels || []
  const m = props.chart?.matrix
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

function pieData() {
  const items = props.chart?.items || []
  if (items.length) {
    return items.map((it, i) => ({ name: it.label || it.name || `Item ${i + 1}`, value: _toNum(it.value) ?? 0 }))
  }
  const s = (props.chart?.series || [])[0]
  if (s?.values?.length) {
    const cats = props.chart?.x_categories
    return s.values.map((p, i) => ({
      name: cats?.[i] ?? String(i + 1),
      value: _toNum(Array.isArray(p) ? p[1] : p) ?? 0,
    }))
  }
  return []
}

function radarData() {
  const cats = props.chart?.x_categories || props.chart?.categories || props.chart?.labels || []
  const series = cartesianSeries()
  const allVals = series.flatMap((s) => s.points.map((p) => p[1])).filter((v): v is number => v != null)
  const max = allVals.length ? Math.max(...allVals) * 1.1 : 1
  const names = cats.length ? cats : (series[0]?.points.map((_, i) => `#${i + 1}`) || [])
  const indicator = names.map((name) => ({ name, max }))
  const data = series.map((s) => ({ name: s.name, value: s.points.map((p) => p[1] ?? 0) }))
  return { indicator, data }
}

function gaugeData() {
  const d = pieData()
  let value = 0
  let name = props.title || 'Value'
  if (d.length) {
    value = d[0].value
    name = d[0].name
  } else {
    const s = cartesianSeries()[0]
    const last = s?.points[s.points.length - 1]
    if (last && last[1] != null) value = last[1]
  }
  const max = Number(props.chart?.max) || (value >= 0 && value <= 100 ? 100 : value * 1.3 || 1)
  return { value, name, max }
}

function hasRenderableData(): boolean {
  const k = kind.value
  const c = props.chart || {}
  if (k === 'heatmap') return heatmapData().data.length > 0
  if (k === 'candlestick') return Array.isArray(c.candles) && c.candles.length > 0
  if (k === 'boxplot') return Array.isArray(c.boxes) && c.boxes.length > 0
  if (k === 'sankey' || k === 'graph') return Array.isArray(c.links) && c.links.length > 0
  if (k === 'treemap' || k === 'sunburst') return (c.items?.length || 0) > 0 || pieData().length > 0
  if (k === 'pie' || k === 'donut' || k === 'funnel' || k === 'waterfall') return pieData().length > 0
  if (k === 'gauge') return pieData().length > 0 || cartesianSeries().length > 0
  return cartesianSeries().length > 0
}

const renderable = computed(() => hasRenderableData())

function buildOption(): echarts.EChartsOption {
  const dark = typeof matchMedia !== 'undefined' && matchMedia('(prefers-color-scheme: dark)').matches
  const tc = dark ? '#8d93aa' : '#5a5f72'
  const k = kind.value

  if (k === 'heatmap') {
    const { xs, ys, data, vmin, vmax } = heatmapData()
    const absMax = Math.max(Math.abs(vmin), Math.abs(vmax)) || 1
    const symmetric = vmin < 0 // correlation-style data spans negatives → diverging scale
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
    const d = pieData()
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
    const d = pieData()
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
    const g = gaugeData()
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
    const { indicator, data } = radarData()
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
    const cats = props.chart?.x_categories || []
    const data = (props.chart?.candles || []).map((c) => (Array.isArray(c) ? c.slice(0, 4).map((v) => _toNum(v) ?? 0) : []))
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
    const cats = props.chart?.x_categories || []
    const data = (props.chart?.boxes || []).map((b) => (Array.isArray(b) ? b.slice(0, 5).map((v) => _toNum(v) ?? 0) : []))
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
    const src = props.chart?.items?.length
      ? props.chart.items.map((it, i) => ({ name: it.label || it.name || `Item ${i + 1}`, value: _toNum(it.value) ?? 0 }))
      : pieData()
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
    const nodes = (props.chart?.nodes || []).map((n) => ({ name: typeof n === 'string' ? n : n.name || '' }))
    const links = (props.chart?.links || []).map((l) => ({ source: l.source, target: l.target, value: _toNum(l.value) ?? 1 }))
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
    const pts = pieData()
    const cats = pts.map((p) => p.name)
    const base: (number | string)[] = []
    const inc: (number | string)[] = []
    const dec: (number | string)[] = []
    let run = 0
    for (const p of pts) {
      const v = Number(p.value) || 0
      if (v >= 0) {
        base.push(run); inc.push(v); dec.push('-')
      } else {
        base.push(run + v); inc.push('-'); dec.push(-v)
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
    const eb = (props.chart?.series || []).map((s, i) => ({
      name: s.name || `Series ${i + 1}`,
      type: 'scatter' as const,
      data: (Array.isArray(s.values) ? s.values : [])
        .filter((v) => Array.isArray(v) && (v as number[]).length >= 2)
        .map((v) => (v as number[]).map((x) => _toNum(x) ?? 0)),
      symbolSize: (val: number[]) => Math.max(8, Math.sqrt(Math.abs(Number(val?.[2]) || 1)) * 5),
      itemStyle: { color: PALETTE[i % PALETTE.length], opacity: 0.7 },
    }))
    return {
      backgroundColor: 'transparent',
      textStyle: { color: tc, fontSize: 11 },
      tooltip: { trigger: 'item' },
      legend: eb.length > 1 ? { bottom: 0, textStyle: { color: tc, fontSize: 10 } } : undefined,
      grid: { left: 48, right: 24, top: 20, bottom: 40, containLabel: true },
      xAxis: { type: 'value', name: props.chart?.x_label || '', nameLocation: 'middle', nameGap: 26, splitLine: { lineStyle: { opacity: 0.2 } } },
      yAxis: { type: 'value', name: props.chart?.y_label || '', nameLocation: 'middle', nameGap: 36, splitLine: { lineStyle: { opacity: 0.2 } } },
      series: eb,
    }
  }

  // cartesian family: line / bar / hbar / area / scatter / stacked / combo / histogram
  const series = cartesianSeries()
  const isHBar = k === 'hbar' || k === 'horizontal-bar' || k === 'horizontal_bar'
  const isStacked = k === 'stacked' || k === 'stacked-bar' || k === 'stacked_bar'
  const isCombo = k === 'combo'
  const isBar = k === 'bar' || k === 'histogram' || isStacked || isHBar
  const isScatter = k === 'scatter' || k === 'bubble'
  const isArea = k === 'area'

  // Category axis (e.g. "Q1 2023") when x_categories provided; else numeric.
  const cats = props.chart?.x_categories
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
    name: (isHBar ? props.chart?.x_label : props.chart?.y_label) || '',
    nameLocation: 'middle' as const,
    nameGap: 36,
    splitLine: { show: true, lineStyle: { opacity: 0.2 } },
  }
  const catAxis = useCat
    ? {
        type: 'category' as const,
        data: cats,
        name: (isHBar ? props.chart?.y_label : props.chart?.x_label) || '',
        nameLocation: 'middle' as const,
        nameGap: 30,
        axisLabel: { fontSize: 10, rotate: !isHBar && (cats as string[]).length > 6 ? 30 : 0 },
      }
    : {
        type: 'value' as const,
        name: (isHBar ? props.chart?.y_label : props.chart?.x_label) || '',
        nameLocation: 'middle' as const,
        nameGap: 28,
        splitLine: { show: true, lineStyle: { opacity: 0.2 } },
      }

  return {
    animation: true,
    animationDuration: 1100,
    backgroundColor: 'transparent',
    textStyle: { color: tc, fontSize: 11 },
    grid: { left: 48, right: 24, top: 36, bottom: 40, containLabel: true },
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    legend: legendNames.length > 1 ? { data: legendNames, bottom: 0, textStyle: { color: tc, fontSize: 10 } } : undefined,
    xAxis: isHBar ? valueAxis : catAxis,
    yAxis: isHBar ? catAxis : valueAxis,
    series: eSeries,
  }
}

function resize() {
  chart?.resize()
}

function initChart() {
  if (!rootEl.value) return
  if (chart) {
    chart.dispose()
    chart = null
  }
  chart = echarts.init(rootEl.value, undefined, { renderer: 'canvas' })
  chart.setOption(buildOption(), true)
}

function refresh() {
  // The chart div is rendered with v-if="renderable"; wait for the DOM before init.
  nextTick(() => {
    if (!renderable.value) {
      chart?.dispose()
      chart = null
      return
    }
    if (!chart || !rootEl.value) {
      initChart()
      return
    }
    chart.setOption(buildOption(), true)
  })
}

function downloadPng() {
  if (!chart) return
  const dark = typeof matchMedia !== 'undefined' && matchMedia('(prefers-color-scheme: dark)').matches
  const url = chart.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: dark ? '#13151c' : '#ffffff' })
  const name =
    (props.title || 'chart').replace(/[^a-zA-Z0-9_-]+/g, '-').replace(/^-|-$/g, '').slice(0, 60) || 'chart'
  const a = document.createElement('a')
  a.href = url
  a.download = `${name}.png`
  document.body.appendChild(a)
  a.click()
  a.remove()
}

onMounted(() => {
  refresh()
  window.addEventListener('resize', resize)
})

onUnmounted(() => {
  window.removeEventListener('resize', resize)
  chart?.dispose()
  chart = null
})

watch(() => props.chart, refresh, { deep: true })
</script>

<template>
  <div
    class="wsc-root rounded-xl border bg-card overflow-hidden shadow-sm transition-shadow duration-300 hover:shadow-md"
  >
    <div v-if="title || renderable" class="px-3 py-2 border-b flex items-center justify-between gap-2">
      <span class="text-xs font-medium truncate">{{ title }}</span>
      <button
        v-if="renderable"
        type="button"
        class="shrink-0 inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground border rounded-md px-2 py-0.5 transition"
        title="Download chart as PNG"
        @click="downloadPng"
      >
        <ArrowDownTrayIcon class="h-3 w-3" /> PNG
      </button>
    </div>
    <div v-if="renderable" ref="rootEl" class="w-full h-[min(360px,52vh)] min-h-[220px]" />
    <div v-else class="px-3 py-6 text-xs text-muted-foreground text-center">
      No renderable data for chart type "{{ kind }}".
    </div>
  </div>
</template>
