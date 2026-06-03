<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch, computed, nextTick } from 'vue'
import * as echarts from 'echarts'

type Series = { name?: string; color?: string; values?: [number, number][] }

const props = defineProps<{
  title?: string
  chart: {
    kind?: string
    x_label?: string
    y_label?: string
    // heatmap / matrix
    x_labels?: string[]
    y_labels?: string[]
    categories?: string[]
    labels?: string[]
    matrix?: number[][]
    // cartesian
    series?: Series[]
    // pie / donut
    items?: { label?: string; name?: string; value?: number }[]
  }
}>()

const rootEl = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null

const COLOR_MAP: Record<string, string> = {
  blue: '#3b82f6',
  orange: '#f97316',
  green: '#22c55e',
  red: '#ef4444',
  purple: '#a855f7',
}
const PALETTE = ['#3b82f6', '#f97316', '#22c55e', '#ef4444', '#a855f7', '#14b8a6', '#eab308', '#ec4899']

function resolveColor(c?: string, i = 0): string {
  if (!c) return PALETTE[i % PALETTE.length]
  const k = c.toLowerCase()
  return COLOR_MAP[k] || c
}

const kind = computed(() => String(props.chart?.kind || 'line').toLowerCase())

function cartesianSeries() {
  const raw = props.chart?.series || []
  return raw
    .map((s, i) => {
      const pts = Array.isArray(s.values)
        ? s.values.filter((p) => Array.isArray(p) && p.length >= 2)
        : []
      return { name: s.name || `Series ${i + 1}`, color: resolveColor(s.color, i), points: pts as [number, number][] }
    })
    .filter((s) => s.points.length)
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
    return items.map((it, i) => ({ name: it.label || it.name || `Item ${i + 1}`, value: Number(it.value) || 0 }))
  }
  const s = (props.chart?.series || [])[0]
  if (s?.values?.length) {
    return s.values.map((p, i) => ({ name: String(i + 1), value: Number(p[1]) || 0 }))
  }
  return []
}

function hasRenderableData(): boolean {
  const k = kind.value
  if (k === 'heatmap') return heatmapData().data.length > 0
  if (k === 'pie' || k === 'donut') return pieData().length > 0
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

  // cartesian: line / bar / area / scatter (unknown kinds fall through to line)
  const series = cartesianSeries()
  const isBar = k === 'bar'
  const isScatter = k === 'scatter'
  const isArea = k === 'area'

  const eSeries = series.map((s) => ({
    name: s.name,
    type: (isBar ? 'bar' : isScatter ? 'scatter' : 'line') as 'bar' | 'scatter' | 'line',
    data: s.points,
    smooth: !isBar && !isScatter,
    showSymbol: isScatter || !isBar,
    symbolSize: isScatter ? 10 : 6,
    itemStyle: { color: s.color },
    areaStyle: isArea ? { opacity: 0.18, color: s.color } : undefined,
    lineStyle: isBar || isScatter ? undefined : { width: 2 },
    emphasis: { focus: 'series' as const },
    animationDuration: 900,
    animationEasing: 'cubicOut' as const,
  }))

  const legendNames = series.map((s) => s.name)

  return {
    animation: true,
    animationDuration: 1100,
    backgroundColor: 'transparent',
    textStyle: { color: tc, fontSize: 11 },
    grid: { left: 48, right: 24, top: 36, bottom: 40, containLabel: true },
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    legend: legendNames.length > 1 ? { data: legendNames, bottom: 0, textStyle: { color: tc, fontSize: 10 } } : undefined,
    xAxis: {
      type: 'value',
      name: props.chart?.x_label || '',
      nameLocation: 'middle',
      nameGap: 28,
      splitLine: { show: true, lineStyle: { opacity: 0.2 } },
    },
    yAxis: {
      type: 'value',
      name: props.chart?.y_label || '',
      nameLocation: 'middle',
      nameGap: 36,
      splitLine: { show: true, lineStyle: { opacity: 0.2 } },
    },
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
    <div v-if="title" class="px-3 py-2 border-b text-xs font-medium">{{ title }}</div>
    <div v-if="renderable" ref="rootEl" class="w-full h-[min(360px,52vh)] min-h-[220px]" />
    <div v-else class="px-3 py-6 text-xs text-muted-foreground text-center">
      No renderable data for chart type "{{ kind }}".
    </div>
  </div>
</template>
