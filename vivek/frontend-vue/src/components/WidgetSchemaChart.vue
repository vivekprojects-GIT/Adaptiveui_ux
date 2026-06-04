<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch, computed, nextTick } from 'vue'
import * as echarts from 'echarts'
import 'echarts-gl' // registers 3D series: scatter3D / bar3D / line3D / surface
import { ArrowDownTrayIcon } from '@/components/icons'
import { buildEChartsOption, chartHasRenderableData, type ChartSpec } from '@/lib/echartsOption'

const props = defineProps<{
  title?: string
  chart: ChartSpec
}>()

const rootEl = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null

const kind = computed(() => String(props.chart?.kind || 'line').toLowerCase())

const renderable = computed(() => chartHasRenderableData(props.chart))

function buildOption(): echarts.EChartsOption {
  const dark = typeof matchMedia !== 'undefined' && matchMedia('(prefers-color-scheme: dark)').matches
  return buildEChartsOption(props.chart, props.title || '', { dark })
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
