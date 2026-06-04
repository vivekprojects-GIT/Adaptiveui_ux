<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'
import {
  analyticsState,
  computeAvgResponseTimeSec,
  computeLastResponseTimes,
  computeRewardPieLast,
  computeRewardRateLast,
  computeStrategyCounts,
  computeWidgetRenderRate,
} from '@/lib/analyticsStore'

const strategyChartEl = ref<HTMLDivElement | null>(null)
const rewardChartEl = ref<HTMLDivElement | null>(null)
const responseTimeChartEl = ref<HTMLDivElement | null>(null)
const widgetRateChartEl = ref<HTMLDivElement | null>(null)

let strategyChart: echarts.ECharts | null = null
let rewardChart: echarts.ECharts | null = null
let responseTimeChart: echarts.ECharts | null = null
let widgetRateChart: echarts.ECharts | null = null

const chartWindowSize = 30
const rewardWindowSize = 50

const metrics = computed(() => {
  const done = analyticsState.doneEvents
  const widgetRenderedCount = done.filter((d) => (d.widgetSchema ?? '').trim().length > 0).length
  const widgetTotal = done.length

  return {
    widgetRenderRate: computeWidgetRenderRate(),
    avgResponseTime: computeAvgResponseTimeSec(),
    rewardRate: computeRewardRateLast(rewardWindowSize),
    strategyCounts: computeStrategyCounts(chartWindowSize),
    responseTimes: computeLastResponseTimes(chartWindowSize),
    rewardPie: computeRewardPieLast(rewardWindowSize),
    widgetCounts: { rendered: widgetRenderedCount, total: widgetTotal },
  }
})

function updateCharts() {
  if (!strategyChart || !rewardChart || !responseTimeChart || !widgetRateChart) return

  const stratKeys = Object.keys(metrics.value.strategyCounts)
  const stratVals = stratKeys.map((k) => metrics.value.strategyCounts[k] ?? 0)

  strategyChart.setOption(
    {
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: stratKeys.length ? stratKeys : ['—'] },
      yAxis: { type: 'value' },
      series: [
        {
          data: stratVals.length ? stratVals : [0],
          type: 'bar',
          itemStyle: { color: '#aa3bff' },
        },
      ],
    },
    true,
  )

  const rp = metrics.value.rewardPie
  const totalRewards = (rp.pos ?? 0) + (rp.neg ?? 0)
  rewardChart.setOption(
    {
      tooltip: { trigger: 'item' },
      series: [
        {
          type: 'pie',
          radius: ['55%', '80%'],
          avoidLabelOverlap: true,
          label: { formatter: '{b}: {c}' },
          data:
            totalRewards > 0
              ? [
                  { name: 'Positive', value: rp.pos },
                  { name: 'Negative', value: rp.neg },
                ]
              : [{ name: 'No rewards', value: 1 }],
        },
      ],
    },
    true,
  )

  const respTimes = metrics.value.responseTimes
  responseTimeChart.setOption(
    {
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: respTimes.length ? respTimes.map((d) => String(d.idx)) : ['—'] },
      yAxis: { type: 'value' },
      series: [
        {
          data: respTimes.length ? respTimes.map((d) => d.elapsed) : [0],
          type: 'line',
          smooth: true,
          symbol: 'circle',
        },
      ],
    },
    true,
  )

  const wc = metrics.value.widgetCounts
  const noWidget = wc.total - wc.rendered
  widgetRateChart.setOption(
    {
      tooltip: { trigger: 'item' },
      series: [
        {
          type: 'pie',
          radius: ['55%', '80%'],
          label: { formatter: '{b}: {c}' },
          data:
            wc.total > 0
              ? [
                  { name: 'Widgets', value: wc.rendered },
                  { name: 'No widget', value: Math.max(0, noWidget) },
                ]
              : [{ name: 'No data', value: 1 }],
        },
      ],
    },
    true,
  )
}

function resizeCharts() {
  strategyChart?.resize()
  rewardChart?.resize()
  responseTimeChart?.resize()
  widgetRateChart?.resize()
}

onMounted(() => {
  if (strategyChartEl.value) strategyChart = echarts.init(strategyChartEl.value)
  if (rewardChartEl.value) rewardChart = echarts.init(rewardChartEl.value)
  if (responseTimeChartEl.value) responseTimeChart = echarts.init(responseTimeChartEl.value)
  if (widgetRateChartEl.value) widgetRateChart = echarts.init(widgetRateChartEl.value)

  updateCharts()

  watch(
    () => [analyticsState.doneEvents.length, analyticsState.rewardEvents.length],
    () => updateCharts(),
  )

  window.addEventListener('resize', resizeCharts)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', resizeCharts)
  strategyChart?.dispose()
  rewardChart?.dispose()
  responseTimeChart?.dispose()
  widgetRateChart?.dispose()
})
</script>

<template>
  <div>
    <h2 class="text-sm font-semibold mb-3">Analytics (live)</h2>

    <div class="grid grid-cols-1 gap-3">
      <div class="rounded-lg border bg-card p-3">
        <div class="text-xs text-muted-foreground mb-2">Strategy distribution (last {{ chartWindowSize }})</div>
        <div ref="strategyChartEl" class="h-[160px] w-full" />
      </div>

      <div class="rounded-lg border bg-card p-3">
        <div class="text-xs text-muted-foreground mb-2">
          Reward rate (last {{ rewardWindowSize }})
          <span v-if="metrics.rewardRate !== null" class="ml-2 font-medium text-foreground">
            {{ Math.round((metrics.rewardRate as number) * 100) }}%
          </span>
        </div>
        <div ref="rewardChartEl" class="h-[160px] w-full" />
      </div>

      <div class="rounded-lg border bg-card p-3">
        <div class="text-xs text-muted-foreground mb-2">
          Avg response time
          <span v-if="metrics.avgResponseTime !== null" class="ml-2 font-medium text-foreground">
            {{ (metrics.avgResponseTime as number).toFixed(2) }}s
          </span>
        </div>
        <div ref="responseTimeChartEl" class="h-[160px] w-full" />
      </div>

      <div class="rounded-lg border bg-card p-3">
        <div class="text-xs text-muted-foreground mb-2">
          Widget render rate
          <span class="ml-2 font-medium text-foreground" v-if="metrics.widgetCounts.total > 0">
            {{ Math.round(metrics.widgetRenderRate * 100) }}%
          </span>
        </div>
        <div ref="widgetRateChartEl" class="h-[160px] w-full" />
      </div>
    </div>
  </div>
</template>

