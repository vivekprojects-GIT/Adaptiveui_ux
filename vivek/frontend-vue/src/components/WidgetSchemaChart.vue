<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import * as echarts from 'echarts'

const props = defineProps<{
  title?: string
  chart: {
    kind?: string
    x_label?: string
    y_label?: string
    series?: { name?: string; color?: string; values?: [number, number][] }[]
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

function resolveColor(c?: string): string {
  if (!c) return COLOR_MAP.blue
  const k = c.toLowerCase()
  return COLOR_MAP[k] || c
}

function buildOption() {
  const kind = (props.chart?.kind || 'line').toLowerCase()
  const rawSeries = props.chart?.series || []
  const xLabel = props.chart?.x_label || ''
  const yLabel = props.chart?.y_label || ''

  const seriesList = rawSeries.map((s) => {
    const pts = Array.isArray(s.values) ? s.values.filter((p) => Array.isArray(p) && p.length >= 2) : []
    return {
      name: s.name || 'Series',
      color: resolveColor(s.color),
      points: pts as [number, number][],
    }
  })

  const isBar = kind === 'bar'

  const eSeries = seriesList.map((s) => ({
    name: s.name,
    type: isBar ? ('bar' as const) : ('line' as const),
    data: s.points,
    smooth: !isBar,
    showSymbol: !isBar,
    symbolSize: isBar ? 0 : 6,
    itemStyle: { color: s.color },
    lineStyle: isBar ? undefined : { width: 2 },
    emphasis: { focus: 'series' as const },
    animationDuration: 900,
    animationEasing: 'cubicOut',
  }))

  const legendNames = seriesList.map((s) => s.name)

  return {
    animation: true,
    animationDuration: 1100,
    backgroundColor: 'transparent',
    textStyle: {
      color: '#64748b',
      fontSize: 11,
    },
    grid: { left: 48, right: 24, top: 36, bottom: 40, containLabel: true },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' },
    },
    legend: legendNames.length > 1 ? { data: legendNames, bottom: 0, textStyle: { fontSize: 10 } } : undefined,
    xAxis: {
      type: 'value',
      name: xLabel,
      nameLocation: 'middle',
      nameGap: 28,
      splitLine: { show: true, lineStyle: { opacity: 0.2 } },
    },
    yAxis: {
      type: 'value',
      name: yLabel,
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

onMounted(() => {
  initChart()
  window.addEventListener('resize', resize)
})

onUnmounted(() => {
  window.removeEventListener('resize', resize)
  chart?.dispose()
  chart = null
})

watch(
  () => props.chart,
  () => {
    if (!chart || !rootEl.value) {
      initChart()
      return
    }
    chart.setOption(buildOption(), true)
  },
  { deep: true },
)
</script>

<template>
  <div
    class="wsc-root rounded-xl border bg-card overflow-hidden shadow-sm transition-shadow duration-300 hover:shadow-md"
  >
    <div v-if="title" class="px-3 py-2 border-b text-xs font-medium">{{ title }}</div>
    <div ref="rootEl" class="w-full h-[min(360px,52vh)] min-h-[220px]" />
  </div>
</template>
