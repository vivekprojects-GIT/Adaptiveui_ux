<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { Motion } from '@motionone/vue'
import * as echarts from 'echarts'
import BanditInsightsSection from '@/components/BanditInsightsSection.vue'
import Card from '@/components/ui/Card.vue'
import { getAccessToken } from '@/lib/auth'
import { banditState, fetchBanditStateFromApi } from '@/lib/banditState'
import { analyticsState } from '@/lib/analyticsStore'
import { MOTION_BASE } from '@/lib/motion'
import { enabledStrategyIds, getStrategyLabel } from '@/lib/strategiesStore'
import { sessionUserState } from '@/lib/sessionUser'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ((typeof location !== 'undefined' && /^(localhost|127.0.0.1)$/.test(location.hostname)) ? 'http://localhost:5051' : '')

const rewardWindow = 50
const strategyWindow = 60
const rangeOptions = [
  { id: '24h', label: '24h', ms: 24 * 60 * 60 * 1000 },
  { id: '7d', label: '7d', ms: 7 * 24 * 60 * 60 * 1000 },
  { id: '30d', label: '30d', ms: 30 * 24 * 60 * 60 * 1000 },
] as const
const selectedRange = ref<(typeof rangeOptions)[number]['id']>('7d')
const selectedStrategy = ref<string>('all')
const showDeployMarkers = ref(false)
const showAdvancedDiagnostics = ref(false)
const nowTick = ref(Date.now())

function authHeaders(): HeadersInit {
  const token = getAccessToken()
  const h: Record<string, string> = {}
  if (token) h.Authorization = `Bearer ${token}`
  return h
}

let banditPoll: ReturnType<typeof setInterval> | null = null
let nowTicker: ReturnType<typeof setInterval> | null = null

const strategyMixEl = ref<HTMLDivElement | null>(null)
const rewardTrendEl = ref<HTMLDivElement | null>(null)
const latencyTrendEl = ref<HTMLDivElement | null>(null)
const latencyHistEl = ref<HTMLDivElement | null>(null)
const widgetSplitEl = ref<HTMLDivElement | null>(null)
const throughputEl = ref<HTMLDivElement | null>(null)
const heatmapEl = ref<HTMLDivElement | null>(null)
const calibrationEl = ref<HTMLDivElement | null>(null)
const cumulativeRewardEl = ref<HTMLDivElement | null>(null)
const winRateByStrategyEl = ref<HTMLDivElement | null>(null)
const latencyScatterEl = ref<HTMLDivElement | null>(null)

let strategyMixChart: echarts.ECharts | null = null
let rewardTrendChart: echarts.ECharts | null = null
let latencyTrendChart: echarts.ECharts | null = null
let latencyHistChart: echarts.ECharts | null = null
let widgetSplitChart: echarts.ECharts | null = null
let throughputChart: echarts.ECharts | null = null
let heatmapChart: echarts.ECharts | null = null
let calibrationChart: echarts.ECharts | null = null
let cumulativeRewardChart: echarts.ECharts | null = null
let winRateByStrategyChart: echarts.ECharts | null = null
let latencyScatterChart: echarts.ECharts | null = null

const metrics = computed(() => {
  const done = filteredDoneEvents.value
  const rewards = filteredRewardEvents.value
  const latencyVals = done
    .map((d) => (typeof d.elapsed === 'number' ? d.elapsed : null))
    .filter((x): x is number => x != null)
  const avgLatency = latencyVals.length ? latencyVals.reduce((a, b) => a + b, 0) / latencyVals.length : null
  const positives = rewards.filter((r) => r.reward >= 1).length
  const rewardRate = rewards.length ? positives / rewards.length : null
  const widgetRate = done.length ? done.filter((d) => (d.widgetSchema ?? '').trim().length > 0).length / done.length : 0
  const withWidget = done.filter((d) => (d.widgetSchema ?? '').trim().length > 0).length

  return {
    totalResponses: done.length,
    totalRewards: rewards.length,
    avgLatency,
    rewardRate,
    widgetRate,
    withWidget,
  }
})

const selectedRangeMs = computed(() => rangeOptions.find((x) => x.id === selectedRange.value)?.ms ?? rangeOptions[1].ms)
const rangeStartTs = computed(() => Date.now() - selectedRangeMs.value)
const previousRangeStartTs = computed(() => rangeStartTs.value - selectedRangeMs.value)

const filteredDoneEvents = computed(() => {
  return analyticsState.doneEvents.filter((d) => {
    if (d.ts < rangeStartTs.value) return false
    if (selectedStrategy.value !== 'all' && d.strategy !== selectedStrategy.value) return false
    return true
  })
})

const filteredRewardEvents = computed(() => {
  return analyticsState.rewardEvents.filter((r) => {
    if (r.ts < rangeStartTs.value) return false
    if (selectedStrategy.value !== 'all' && r.strategy !== selectedStrategy.value) return false
    return true
  })
})

const previousDoneEvents = computed(() => {
  return analyticsState.doneEvents.filter((d) => {
    if (d.ts < previousRangeStartTs.value || d.ts >= rangeStartTs.value) return false
    if (selectedStrategy.value !== 'all' && d.strategy !== selectedStrategy.value) return false
    return true
  })
})

const previousRewardEvents = computed(() => {
  return analyticsState.rewardEvents.filter((r) => {
    if (r.ts < previousRangeStartTs.value || r.ts >= rangeStartTs.value) return false
    if (selectedStrategy.value !== 'all' && r.strategy !== selectedStrategy.value) return false
    return true
  })
})

function pctDelta(current: number | null, previous: number | null): number | null {
  if (current == null || previous == null) return null
  if (Math.abs(previous) < 1e-9) return null
  return ((current - previous) / Math.abs(previous)) * 100
}

const previousMetrics = computed(() => {
  const done = previousDoneEvents.value
  const rewards = previousRewardEvents.value
  const latVals = done
    .map((d) => (typeof d.elapsed === 'number' ? d.elapsed : null))
    .filter((x): x is number => x != null)
  const avgLatency = latVals.length ? latVals.reduce((a, b) => a + b, 0) / latVals.length : null
  const rewardRate = rewards.length ? rewards.filter((r) => r.reward >= 1).length / rewards.length : null
  const widgetRate = done.length ? done.filter((d) => (d.widgetSchema ?? '').trim().length > 0).length / done.length : null
  return {
    totalResponses: done.length,
    totalRewards: rewards.length,
    avgLatency,
    rewardRate,
    widgetRate,
  }
})

const metricDeltas = computed(() => ({
  responses: pctDelta(metrics.value.totalResponses, previousMetrics.value.totalResponses),
  rewards: pctDelta(metrics.value.totalRewards, previousMetrics.value.totalRewards),
  rewardRate: pctDelta(metrics.value.rewardRate, previousMetrics.value.rewardRate),
  avgLatency: pctDelta(metrics.value.avgLatency, previousMetrics.value.avgLatency),
  widgetRate: pctDelta(metrics.value.widgetRate, previousMetrics.value.widgetRate),
}))

const kpiSparkline = computed(() => {
  const buckets = 10
  const done = filteredDoneEvents.value
  if (!done.length) return []
  const size = Math.max(1, Math.ceil(done.length / buckets))
  const values: number[] = []
  for (let i = 0; i < done.length; i += size) {
    values.push(done.slice(i, i + size).length)
  }
  return values.slice(-buckets)
})

function deltaLabel(v: number | null, suffix = '%') {
  if (v == null || Number.isNaN(v)) return 'vs prev: —'
  const sign = v > 0 ? '+' : ''
  return `vs prev: ${sign}${v.toFixed(1)}${suffix}`
}

const lastUpdatedLabel = computed(() => {
  const sec = Math.max(0, Math.floor((nowTick.value - Math.max(0, ...analyticsState.doneEvents.map((d) => d.ts), ...analyticsState.rewardEvents.map((r) => r.ts))) / 1000))
  if (!analyticsState.doneEvents.length && !analyticsState.rewardEvents.length) return 'Updated just now'
  return `Updated ${sec}s ago`
})

const hasFilteredData = computed(() => filteredDoneEvents.value.length > 0 || filteredRewardEvents.value.length > 0)

const activeFilterChips = computed(() => {
  const chips: string[] = []
  if (selectedRange.value !== '7d') chips.push(`Range: ${selectedRange.value}`)
  if (selectedStrategy.value !== 'all') chips.push(`Strategy: ${getStrategyLabel(selectedStrategy.value)}`)
  if (showDeployMarkers.value) chips.push('Deploy markers: On')
  return chips
})

const dashboardInsights = computed(() => {
  const rr = metrics.value.rewardRate == null ? null : metrics.value.rewardRate * 100
  const lat = metrics.value.avgLatency
  const wr = metrics.value.widgetRate * 100
  const rrText = rr == null ? 'Reward trend is still collecting enough feedback.' : `Reward quality is ${rr.toFixed(1)}% in selected range.`
  const latText = lat == null ? 'Latency trend unavailable for current slice.' : `Average response latency is ${lat.toFixed(2)}s.`
  const widgetText = `Widget attach rate is ${wr.toFixed(0)}%, helpful for visual engagement.`
  return [rrText, latText, widgetText]
})

function resetDashboardFilters() {
  selectedRange.value = '7d'
  selectedStrategy.value = 'all'
  showDeployMarkers.value = false
}

const rewardWindowEvents = computed(() => filteredRewardEvents.value.slice(-rewardWindow))

const performanceMetrics = computed(() => {
  const alpha = rewardWindowEvents.value.filter((r) => r.reward >= 1).length
  const beta = rewardWindowEvents.value.filter((r) => r.reward < 1).length
  const total = alpha + beta
  const successRate = total ? alpha / total : null
  return { alpha, beta, total, successRate }
})

const successTrendBuckets = computed(() => {
  const events = rewardWindowEvents.value
  const bucketCount = 7
  if (!events.length) return []
  const n = events.length
  const size = Math.max(1, Math.ceil(n / bucketCount))
  const buckets: number[] = []
  for (let i = 0; i < n; i += size) {
    const slice = events.slice(i, i + size)
    const wins = slice.filter((e) => e.reward >= 1).length
    const rate = slice.length ? wins / slice.length : 0
    buckets.push(rate)
    if (buckets.length >= bucketCount) break
  }
  return buckets.slice(-bucketCount)
})

const strategyConfusionBars = computed(() => {
  const map: Record<string, { pos: number; neg: number; total: number }> = {}
  for (const r of rewardWindowEvents.value) {
    if (!map[r.strategy]) map[r.strategy] = { pos: 0, neg: 0, total: 0 }
    if (r.reward >= 1) map[r.strategy].pos += 1
    else map[r.strategy].neg += 1
    map[r.strategy].total += 1
  }
  return Object.entries(map)
    .map(([strategy, v]) => ({ strategy, ...v, winRate: v.total ? v.pos / v.total : 0 }))
    .sort((a, b) => b.total - a.total)
    .slice(0, 5)
})

const negativeFeedbackEntries = computed(() => banditState.rewardLog.filter((e) => e.reward < 1).slice(0, rewardWindow))

const negativeReasons = computed(() => {
  const map = new Map<string, number>()
  for (const e of negativeFeedbackEntries.value) {
    const detail = String(e.detail ?? '').trim()
    const reason =
      detail.length > 0
        ? detail
        : e.source === 'manual'
          ? 'Manual feedback'
          : e.source === 'auto'
            ? 'Auto-detected'
            : 'Negative feedback'
    map.set(reason, (map.get(reason) ?? 0) + 1)
  }

  const items = Array.from(map.entries()).map(([reason, count]) => ({ reason, count }))
  items.sort((a, b) => b.count - a.count)
  return items.slice(0, 5)
})

const feedbackSourceBreakdown = computed(() => {
  let manual = 0
  let auto = 0
  for (const e of negativeFeedbackEntries.value) {
    if (e.source === 'manual') manual += 1
    else if (e.source === 'auto') auto += 1
  }
  const total = manual + auto
  return { manual, auto, total }
})

const topReasonSuggestion = computed(() => {
  const top = negativeReasons.value[0]?.reason ?? ''
  if (top.toLowerCase().includes('simpl')) return 'Simplification needed: tighten the instruction and reduce cognitive load.'
  if (top.toLowerCase().includes('ascii')) return 'Visualization issues: add clearer formatting constraints and examples.'
  return 'Instruction tuning: adjust the weakest strategy’s wording and keep only the most effective formatting rules.'
})

const strategyTable = computed(() => {
  const done = filteredDoneEvents.value.slice(-strategyWindow)
  const rewards = filteredRewardEvents.value
  const map: Record<string, { count: number; pos: number; neg: number }> = {}

  for (const d of done) {
    if (!map[d.strategy]) map[d.strategy] = { count: 0, pos: 0, neg: 0 }
    map[d.strategy].count += 1
  }
  for (const r of rewards) {
    if (!map[r.strategy]) map[r.strategy] = { count: 0, pos: 0, neg: 0 }
    if (r.reward >= 1) map[r.strategy].pos += 1
    else map[r.strategy].neg += 1
  }

  return Object.entries(map)
    .map(([strategy, v]) => {
      const total = v.pos + v.neg
      const rr = total > 0 ? v.pos / total : null
      return { strategy, ...v, rewardRate: rr }
    })
    .sort((a, b) => b.count - a.count)
})

function updateCharts() {
  if (
    !strategyMixChart ||
    !rewardTrendChart ||
    !latencyTrendChart ||
    !latencyHistChart ||
    !widgetSplitChart ||
    !throughputChart ||
    !winRateByStrategyChart ||
    (showAdvancedDiagnostics.value && (!heatmapChart || !calibrationChart || !cumulativeRewardChart || !latencyScatterChart))
  ) return

  const done = filteredDoneEvents.value.slice(-strategyWindow)
  const rewards = filteredRewardEvents.value.slice(-rewardWindow)

  const strategyNames = Array.from(new Set(done.map((d) => d.strategy)))
  const bucketCount = 10
  const start = done[0]?.ts ?? Date.now()
  const end = done[done.length - 1]?.ts ?? Date.now()
  const step = Math.max(1, Math.floor((end - start) / bucketCount))
  const xLabels = Array.from({ length: bucketCount }, (_, i) => `${i + 1}`)
  const strategySeriesMap: Record<string, number[]> = {}
  for (const s of strategyNames) strategySeriesMap[s] = Array.from({ length: bucketCount }, () => 0)
  for (const d of done) {
    const idx = Math.min(bucketCount - 1, Math.max(0, Math.floor((d.ts - start) / step)))
    if (!strategySeriesMap[d.strategy]) strategySeriesMap[d.strategy] = Array.from({ length: bucketCount }, () => 0)
    strategySeriesMap[d.strategy][idx] += 1
  }

  strategyMixChart.setOption(
    {
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: xLabels },
      yAxis: { type: 'value' },
      legend: { type: 'scroll' },
      series: strategyNames.map((s) => ({
        name: getStrategyLabel(s),
        type: 'line',
        areaStyle: { opacity: 0.2 },
        smooth: true,
        stack: 'strategy',
        symbol: 'none',
        data: strategySeriesMap[s] ?? Array.from({ length: bucketCount }, () => 0),
      })),
    },
    true,
  )

  const rewardY = rewards.map<number>((r) => (r.reward >= 1 ? 1 : 0))
  const rollingY = rewardY.map((_, idx) => {
    const a = Math.max(0, idx - 4)
    const slice = rewardY.slice(a, idx + 1)
    return slice.reduce((x, y) => x + y, 0) / slice.length
  })
  const rewardX = rewards.map((_, i) => String(i + 1))
  rewardTrendChart.setOption(
    {
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: rewardX.length ? rewardX : ['—'] },
      yAxis: { type: 'value', min: 0, max: 1, axisLabel: { formatter: '{value}' } },
      series: [
        {
          type: 'line',
          smooth: true,
          data: rewardY.length ? rewardY : [0],
          areaStyle: { opacity: 0.2 },
          symbol: 'none',
          itemStyle: { color: '#06b6d4' },
        },
        {
          type: 'line',
          smooth: true,
          data: rollingY.length ? rollingY : [0],
          symbol: 'none',
          lineStyle: { width: 2, type: 'dashed', color: '#10b981' },
        },
      ],
      markLine: showDeployMarkers.value
        ? {
            symbol: 'none',
            data: [{ xAxis: Math.max(1, Math.floor(rewardX.length / 2)), label: { formatter: 'Deploy' } }],
          }
        : undefined,
    },
    true,
  )

  const latencyPoints = done
    .map((d, i) => ({ x: i + 1, y: typeof d.elapsed === 'number' ? d.elapsed : null }))
    .filter((p) => p.y !== null) as { x: number; y: number }[]

  const sortedLat = [...latencyPoints.map((p) => p.y)].sort((a, b) => a - b)
  const p50 = sortedLat.length ? sortedLat[Math.floor(0.5 * (sortedLat.length - 1))] : 0
  const p95 = sortedLat.length ? sortedLat[Math.floor(0.95 * (sortedLat.length - 1))] : 0
  latencyTrendChart.setOption(
    {
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: latencyPoints.length ? latencyPoints.map((p) => String(p.x)) : ['—'] },
      yAxis: [{ type: 'value', name: 'Latency (s)' }, { type: 'value', name: 'Volume' }],
      series: [
        {
          type: 'bar',
          yAxisIndex: 1,
          data: latencyPoints.length ? latencyPoints.map(() => 1) : [0],
          itemStyle: { color: 'rgba(148,163,184,0.35)' },
        },
        {
          type: 'line',
          smooth: true,
          data: latencyPoints.length ? latencyPoints.map((p) => p.y) : [0],
          symbol: 'none',
          itemStyle: { color: '#10b981' },
        },
        {
          type: 'line',
          smooth: true,
          data: latencyPoints.length ? latencyPoints.map(() => p50) : [0],
          symbol: 'none',
          lineStyle: { type: 'dashed', color: '#06b6d4' },
        },
        {
          type: 'line',
          smooth: true,
          data: latencyPoints.length ? latencyPoints.map(() => p95) : [0],
          symbol: 'none',
          lineStyle: { type: 'dashed', color: '#f59e0b' },
        },
      ],
    },
    true,
  )

  const latVals = latencyPoints.map((p) => p.y)
  const bins = [0, 1, 2, 3, 5, 8, 13, 20]
  const histCounts = Array.from({ length: bins.length - 1 }, () => 0)
  for (const v of latVals) {
    for (let i = 0; i < bins.length - 1; i += 1) {
      if (v >= bins[i] && v < bins[i + 1]) {
        histCounts[i] += 1
        break
      }
      if (i === bins.length - 2 && v >= bins[i + 1]) histCounts[i] += 1
    }
  }
  latencyHistChart.setOption(
    {
      tooltip: { trigger: 'axis' },
      xAxis: {
        type: 'category',
        data: bins.slice(0, -1).map((b, i) => `${b}-${bins[i + 1]}s`),
      },
      yAxis: { type: 'value' },
      series: [{ type: 'bar', data: histCounts, itemStyle: { color: '#06b6d4' } }],
    },
    true,
  )

  const withWidget = done.filter((d) => (d.widgetSchema ?? '').trim().length > 0).length
  const withoutWidget = done.length - withWidget
  const jsonSchemaCount = 0
  widgetSplitChart.setOption(
    {
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: ['Current range'] },
      yAxis: { type: 'value' },
      legend: { type: 'plain' },
      series: [
        { type: 'bar', name: 'JSON schema', data: [jsonSchemaCount], itemStyle: { color: '#22c55e' } },
        { type: 'bar', name: 'HTML widget', data: [withWidget], itemStyle: { color: '#06b6d4' } },
        { type: 'bar', name: 'None', data: [Math.max(0, withoutWidget)], itemStyle: { color: '#94a3b8' } },
      ],
    },
    true,
  )

  const minuteMap: Record<string, number> = {}
  for (const d of done) {
    const m = new Date(d.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    minuteMap[m] = (minuteMap[m] ?? 0) + 1
  }
  const minuteLabels = Object.keys(minuteMap)
  throughputChart.setOption(
    {
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: minuteLabels.length ? minuteLabels : ['—'] },
      yAxis: { type: 'value' },
      series: [
        {
          type: 'bar',
          data: minuteLabels.length ? minuteLabels.map((k) => minuteMap[k]) : [0],
          itemStyle: { color: '#06b6d4' },
        },
      ],
    },
    true,
  )

  // --- Time series: cumulative reward trend ---
  const cumulativeX = rewards.map((_, i) => String(i + 1))
  let running = 0
  const cumulativeY = rewards.map((r) => {
    running += r.reward >= 1 ? 1 : 0
    return running
  })
  if (cumulativeRewardChart) {
    cumulativeRewardChart.setOption(
      {
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: cumulativeX.length ? cumulativeX : ['—'] },
        yAxis: { type: 'value', name: 'Cumulative wins' },
        series: [
          {
            type: 'line',
            smooth: true,
            symbol: 'circle',
            data: cumulativeY.length ? cumulativeY : [0],
            areaStyle: { opacity: 0.2 },
            lineStyle: { width: 2, color: '#10b981' },
            itemStyle: { color: '#10b981' },
          },
        ],
      },
      true,
    )
  }

  // --- Comparison: strategy win-rate bar chart ---
  const strategyPerfMap: Record<string, { wins: number; total: number }> = {}
  for (const r of rewards) {
    if (!strategyPerfMap[r.strategy]) strategyPerfMap[r.strategy] = { wins: 0, total: 0 }
    strategyPerfMap[r.strategy].total += 1
    if (r.reward >= 1) strategyPerfMap[r.strategy].wins += 1
  }
  const perfRows = Object.entries(strategyPerfMap)
    .map(([strategy, v]) => ({ strategy, rate: v.total ? (v.wins / v.total) * 100 : 0, total: v.total }))
    .sort((a, b) => b.rate - a.rate)
    .slice(0, 10)
  winRateByStrategyChart.setOption(
    {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any) => {
          const p = Array.isArray(params) ? params[0] : params
          const row = perfRows[p?.dataIndex ?? 0]
          return `${row?.strategy ?? ''}<br/>Win rate: ${(row?.rate ?? 0).toFixed(1)}%<br/>Samples: ${row?.total ?? 0}`
        },
      },
      grid: { left: 140, right: 24, top: 16, bottom: 24 },
      xAxis: { type: 'value', min: 0, max: 100, axisLabel: { formatter: '{value}%' } },
      yAxis: { type: 'category', data: perfRows.map((r) => r.strategy) },
      series: [
        {
          type: 'bar',
          data: perfRows.map((r) => Number(r.rate.toFixed(1))),
          itemStyle: {
            color: (p: any) => (p.data >= 70 ? '#10b981' : p.data >= 45 ? '#f59e0b' : '#ef4444'),
          },
        },
      ],
    },
    true,
  )

  // --- Diagnostics: latency vs reward scatter ---
  const rewardByStrategy = new Map<string, number[]>(Object.keys(strategyPerfMap).map((k) => [k, []]))
  for (const r of rewards) {
    const arr = rewardByStrategy.get(r.strategy)
    if (arr) arr.push(r.reward >= 1 ? 1 : 0)
  }
  const scatterPoints = done
    .filter((d) => typeof d.elapsed === 'number')
    .map((d) => {
      const rv = rewardByStrategy.get(d.strategy) ?? []
      const avgReward = rv.length ? rv.reduce((a, b) => a + b, 0) / rv.length : 0
      return [Number(d.elapsed), Number((avgReward * 100).toFixed(1))]
    })
  if (latencyScatterChart) {
    latencyScatterChart.setOption(
      {
        tooltip: { trigger: 'item', formatter: (p: any) => `Latency: ${p.value?.[0] ?? 0}s<br/>Reward: ${p.value?.[1] ?? 0}%` },
        xAxis: { type: 'value', name: 'Latency (s)' },
        yAxis: { type: 'value', name: 'Avg reward (%)', min: 0, max: 100 },
        series: [
          {
            type: 'scatter',
            data: scatterPoints.length ? scatterPoints : [[0, 0]],
            symbolSize: 10,
            itemStyle: { color: '#8b5cf6', opacity: 0.85 },
          },
        ],
      },
      true,
    )
  }

  // --- Strategy Performance Heatmap (Observed Rewards) ---
  const rewardEventsForHeatmap = rewards
  if (heatmapChart && !rewardEventsForHeatmap.length) {
    heatmapChart.setOption(
      {
        tooltip: { show: false },
        xAxis: { type: 'category', data: ['—'] },
        yAxis: { type: 'category', data: ['—'] },
        visualMap: { show: false },
        series: [{ type: 'heatmap', data: [] }],
      },
      true,
    )
  } else if (heatmapChart) {
    const rewardMinTs = Math.min(...rewardEventsForHeatmap.map((e) => e.ts))
    const rewardMaxTs = Math.max(...rewardEventsForHeatmap.map((e) => e.ts))

    const niceBucketsMs = [
      60 * 1000, // 1m
      5 * 60 * 1000, // 5m
      15 * 60 * 1000, // 15m
      30 * 60 * 1000, // 30m
      60 * 60 * 1000, // 1h
      2 * 60 * 60 * 1000, // 2h
      4 * 60 * 60 * 1000, // 4h
      8 * 60 * 60 * 1000, // 8h
      24 * 60 * 60 * 1000, // 1d
    ]

    const desired = (rewardMaxTs - rewardMinTs) / 12
    let bucketMs = niceBucketsMs.find((s) => s >= desired) ?? niceBucketsMs[niceBucketsMs.length - 1]

    let startTs = Math.floor(rewardMinTs / bucketMs) * bucketMs
    let endTs = Math.ceil(rewardMaxTs / bucketMs) * bucketMs
    let bucketCount = Math.ceil((endTs - startTs) / bucketMs)

    while (bucketCount > 24) {
      const idx = niceBucketsMs.indexOf(bucketMs)
      if (idx < 0 || idx === niceBucketsMs.length - 1) break
      bucketMs = niceBucketsMs[idx + 1]
      startTs = Math.floor(rewardMinTs / bucketMs) * bucketMs
      endTs = Math.ceil(rewardMaxTs / bucketMs) * bucketMs
      bucketCount = Math.ceil((endTs - startTs) / bucketMs)
    }
    bucketCount = Math.max(1, bucketCount)

    const strategyCountMap: Record<string, number> = {}
    for (const r of rewardEventsForHeatmap) {
      strategyCountMap[r.strategy] = (strategyCountMap[r.strategy] ?? 0) + 1
    }
    const yLabels = Object.keys(strategyCountMap).sort((a, b) => strategyCountMap[b] - strategyCountMap[a])
    const yIndex = new Map<string, number>(yLabels.map((s, i) => [s, i]))

    const xLabels = Array.from({ length: bucketCount }, (_, i) =>
      new Date(startTs + i * bucketMs).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    )

    const cellMap = new Map<string, { sum: number; count: number }>()
    for (const r of rewardEventsForHeatmap) {
      const xIdx = Math.floor((r.ts - startTs) / bucketMs)
      const yIdx = yIndex.get(r.strategy)
      if (yIdx == null) continue
      if (xIdx < 0 || xIdx >= bucketCount) continue

      const k = `${xIdx}|${yIdx}`
      const cur = cellMap.get(k) ?? { sum: 0, count: 0 }
      cur.sum += r.reward >= 1 ? 1 : 0
      cur.count += 1
      cellMap.set(k, cur)
    }

    const heatmapData = Array.from(cellMap.entries()).map(([k, v]) => {
      const [xStr, yStr] = k.split('|')
      const xIdx = Number(xStr)
      const yIdx = Number(yStr)
      const avg = v.count ? v.sum / v.count : 0
      return { value: [xIdx, yIdx, avg], count: v.count }
    })

    heatmapChart.setOption(
      {
        tooltip: {
          trigger: 'item',
          formatter: (params: any) => {
            const xIdx = params?.value?.[0]
            const yIdx = params?.value?.[1]
            const avg = params?.value?.[2]
            const count = params?.data?.count ?? 0
            const xLabel = typeof xIdx === 'number' && xLabels[xIdx] ? xLabels[xIdx] : '—'
            const yLabel = typeof yIdx === 'number' && yLabels[yIdx] ? yLabels[yIdx] : '—'
            return `Time: ${xLabel}<br/>Strategy: ${yLabel}<br/>Avg observed reward: ${(avg * 100).toFixed(1)}%<br/>Events: ${count}`
          },
        },
        xAxis: {
          type: 'category',
          data: xLabels,
          axisLabel: { interval: Math.max(0, Math.floor(xLabels.length / 6) - 1) },
        },
        yAxis: { type: 'category', data: yLabels },
        visualMap: {
          min: 0,
          max: 1,
          inRange: { color: ['#ef4444', '#22c55e'] },
          text: ['0%', '100%'],
          top: 10,
          left: 'right',
        },
        series: [
          {
            type: 'heatmap',
            data: heatmapData,
            label: { show: false },
            emphasis: { itemStyle: { borderColor: '#e5e7eb', borderWidth: 1 } },
          },
        ],
      },
      true,
    )
  }

  // --- Calibration Curve (Predicted confidence r -> realized win-rate) ---
  const calibrationEvents = rewards.filter(
    (e) => typeof e.predictedR === 'number' && Number.isFinite(e.predictedR),
  )
  const binsCount = 10
  const clamp01 = (x: number) => Math.max(0, Math.min(1, x))

  if (calibrationChart && !calibrationEvents.length) {
    calibrationChart.setOption(
      {
        tooltip: { show: false },
        xAxis: {
          type: 'value',
          min: 0,
          max: 1,
          axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` },
        },
        yAxis: {
          type: 'value',
          min: 0,
          max: 1,
          axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` },
        },
        series: [
          {
            type: 'line',
            name: 'Observed win-rate',
            data: [],
            showSymbol: false,
            lineStyle: { color: '#06b6d4', width: 2 },
          },
          {
            type: 'line',
            name: 'Perfect calibration',
            data: [
              [0, 0],
              [1, 1],
            ],
            lineStyle: { type: 'dashed', color: '#94a3b8' },
            symbol: 'none',
          },
        ],
      },
      true,
    )
  } else if (calibrationChart) {
    const bins = Array.from({ length: binsCount }, () => ({ wins: 0, count: 0 }))
    for (const e of calibrationEvents) {
      const pred = clamp01(Number(e.predictedR))
      const bin = Math.min(binsCount - 1, Math.floor(pred * binsCount))
      bins[bin].count += 1
      if (e.reward >= 1) bins[bin].wins += 1
    }

    const scatterPoints = bins.map((b, i) => {
      const x = (i + 0.5) / binsCount
      const y = b.count ? b.wins / b.count : 0
      return { value: [x, y], count: b.count, bin: i }
    })

    calibrationChart.setOption(
      {
        tooltip: {
          trigger: 'item',
          formatter: (params: any) => {
            const x = params?.value?.[0]
            const y = params?.value?.[1]
            const count = params?.data?.count ?? 0
            if (!params?.data?.bin && params?.data?.bin !== 0) return `Predicted: ${(x * 100).toFixed(0)}%<br/>Observed: ${(y * 100).toFixed(1)}%<br/>Events: ${count}`
            if (count === 0) return `Bin: —<br/>Predicted: ${(x * 100).toFixed(0)}%<br/>Observed win-rate: —<br/>Events: 0`
            const lo = (params.data.bin * 100) / binsCount
            const hi = ((params.data.bin + 1) * 100) / binsCount
            return `Predicted: ${lo.toFixed(0)}% - ${hi.toFixed(0)}%<br/>Observed win-rate: ${(y * 100).toFixed(1)}%<br/>Events: ${count}`
          },
        },
        xAxis: {
          type: 'value',
          min: 0,
          max: 1,
          axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` },
        },
        yAxis: {
          type: 'value',
          min: 0,
          max: 1,
          axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` },
        },
        series: [
          {
            type: 'line',
            name: 'Observed win-rate',
            data: scatterPoints.map((p) => p.value),
            smooth: true,
            showSymbol: false,
            lineStyle: { color: '#06b6d4', width: 2 },
          },
          {
            type: 'scatter',
            name: 'Binned samples',
            data: scatterPoints,
            symbolSize: (params: any) => {
              const c = params?.data?.count ?? 0
              if (!c) return 6
              return Math.min(12, 4 + Math.sqrt(c))
            },
            itemStyle: { color: '#06b6d4' },
          },
          {
            type: 'line',
            name: 'Perfect calibration',
            data: [
              [0, 0],
              [1, 1],
            ],
            lineStyle: { type: 'dashed', color: '#94a3b8' },
            symbol: 'none',
          },
        ],
      },
      true,
    )
  }

}

function initBaseCharts() {
  if (strategyMixEl.value && !strategyMixChart) strategyMixChart = echarts.init(strategyMixEl.value)
  if (rewardTrendEl.value && !rewardTrendChart) rewardTrendChart = echarts.init(rewardTrendEl.value)
  if (latencyTrendEl.value && !latencyTrendChart) latencyTrendChart = echarts.init(latencyTrendEl.value)
  if (latencyHistEl.value && !latencyHistChart) latencyHistChart = echarts.init(latencyHistEl.value)
  if (widgetSplitEl.value && !widgetSplitChart) widgetSplitChart = echarts.init(widgetSplitEl.value)
  if (throughputEl.value && !throughputChart) throughputChart = echarts.init(throughputEl.value)
  if (winRateByStrategyEl.value && !winRateByStrategyChart) winRateByStrategyChart = echarts.init(winRateByStrategyEl.value)
}

function initAdvancedCharts() {
  if (cumulativeRewardEl.value && !cumulativeRewardChart) cumulativeRewardChart = echarts.init(cumulativeRewardEl.value)
  if (heatmapEl.value && !heatmapChart) heatmapChart = echarts.init(heatmapEl.value)
  if (calibrationEl.value && !calibrationChart) calibrationChart = echarts.init(calibrationEl.value)
  if (latencyScatterEl.value && !latencyScatterChart) latencyScatterChart = echarts.init(latencyScatterEl.value)
}

function resizeCharts() {
  strategyMixChart?.resize()
  rewardTrendChart?.resize()
  latencyTrendChart?.resize()
  latencyHistChart?.resize()
  widgetSplitChart?.resize()
  throughputChart?.resize()
  heatmapChart?.resize()
  calibrationChart?.resize()
  cumulativeRewardChart?.resize()
  winRateByStrategyChart?.resize()
  latencyScatterChart?.resize()
}

onMounted(() => {
  initBaseCharts()
  if (showAdvancedDiagnostics.value) initAdvancedCharts()

  void fetchBanditStateFromApi(API_BASE, authHeaders)
  banditPoll = setInterval(() => void fetchBanditStateFromApi(API_BASE, authHeaders), 5000)
  nowTicker = setInterval(() => {
    nowTick.value = Date.now()
  }, 1000)

  updateCharts()
  watch(
    () => [
      analyticsState.doneEvents.length,
      analyticsState.rewardEvents.length,
      selectedRange.value,
      selectedStrategy.value,
      showDeployMarkers.value,
    ],
    updateCharts,
  )
  watch(
    () => showAdvancedDiagnostics.value,
    async (isOpen) => {
      if (!isOpen) return
      await nextTick()
      initAdvancedCharts()
      updateCharts()
      resizeCharts()
    },
  )
  window.addEventListener('resize', resizeCharts)
})

onBeforeUnmount(() => {
  if (banditPoll) clearInterval(banditPoll)
  if (nowTicker) clearInterval(nowTicker)
  window.removeEventListener('resize', resizeCharts)
  strategyMixChart?.dispose()
  rewardTrendChart?.dispose()
  latencyTrendChart?.dispose()
  latencyHistChart?.dispose()
  widgetSplitChart?.dispose()
  throughputChart?.dispose()
  heatmapChart?.dispose()
  calibrationChart?.dispose()
  cumulativeRewardChart?.dispose()
  winRateByStrategyChart?.dispose()
  latencyScatterChart?.dispose()
})
</script>

<template>
  <div class="flex flex-col gap-6 max-w-[1440px] mx-auto p-5 lg:p-8 rounded-3xl glass-panel page-shell page-shell-scrollable page-shell-analytics page-content">
    <div class="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
      <div>
        <h1
          class="text-2xl font-semibold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500"
        >
          Analytics
        </h1>
        <p class="text-sm text-muted-foreground mt-1">
          Bandit posteriors, operational telemetry, and session quality — production-style dashboard.
        </p>
      </div>
      <div class="flex flex-wrap items-center gap-2 sticky top-2 z-20">
        <div class="inline-flex items-center rounded-xl border bg-card/85 p-1 shadow-sm">
          <button
            v-for="r in rangeOptions"
            :key="r.id"
            class="h-9 px-3 text-sm rounded-lg transition-colors"
            :class="selectedRange === r.id ? 'bg-cyan-500/20 text-foreground' : 'text-muted-foreground hover:bg-accent/20'"
            @click="selectedRange = r.id"
          >
            {{ r.label }}
          </button>
        </div>
        <select
          v-model="selectedStrategy"
          class="h-9 rounded-lg border bg-card/85 px-3 text-sm text-foreground shadow-sm"
        >
          <option value="all">All strategies</option>
          <option v-for="sid in enabledStrategyIds" :key="sid" :value="sid">
            {{ getStrategyLabel(sid) }}
          </option>
        </select>
        <button
          class="h-9 rounded-lg border bg-card/85 px-3 text-sm text-muted-foreground hover:text-foreground shadow-sm"
          @click="showDeployMarkers = !showDeployMarkers"
        >
          {{ showDeployMarkers ? 'Hide markers' : 'Show markers' }}
        </button>
        <button
          class="h-9 rounded-lg border bg-card/85 px-3 text-sm text-muted-foreground hover:text-foreground shadow-sm"
          @click="resetDashboardFilters"
        >
          Reset filters
        </button>
      </div>
    </div>
    <div v-if="activeFilterChips.length" class="flex flex-wrap gap-2">
      <span
        v-for="chip in activeFilterChips"
        :key="chip"
        class="inline-flex items-center rounded-full border px-2.5 py-1 text-xs text-muted-foreground bg-card/75"
      >
        {{ chip }}
      </span>
    </div>
    <div class="flex items-center justify-between text-sm text-muted-foreground">
      <div>{{ sessionUserState.username ? `Analytics for ${sessionUserState.username}` : 'Analytics overview' }}</div>
      <div class="rounded-full border border-cyan-500/30 bg-card/85 px-3 py-1.5 shadow-sm">
        {{ lastUpdatedLabel }} · {{ selectedRange }} · Last {{ strategyWindow }} responses
      </div>
    </div>

    <BanditInsightsSection />

    <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
      <Motion tag="div" :initial="{ opacity: 0, y: 10 }" :animate="{ opacity: 1, y: 0 }" :transition="{ ...MOTION_BASE, delay: 0.04 }">
      <Card class="p-5 premium-reveal card-shape card-soft-border" style="--stagger: 0.04s">
        <div class="text-xs font-semibold tracking-widest uppercase text-muted-foreground">Performance Metrics</div>

        <div class="grid grid-cols-3 gap-3 mt-4">
          <div class="rounded-xl border border-border/45 bg-background/50 p-3 min-h-[104px]">
            <div class="text-xs text-muted-foreground">Alpha (α)</div>
            <div class="text-3xl font-semibold mt-1 text-emerald-500">{{ performanceMetrics.alpha }}</div>
            <div class="text-[11px] text-muted-foreground mt-1">successes</div>
          </div>
          <div class="rounded-xl border border-border/45 bg-background/50 p-3 min-h-[104px]">
            <div class="text-xs text-muted-foreground">Beta (β)</div>
            <div class="text-3xl font-semibold mt-1 text-red-500">{{ performanceMetrics.beta }}</div>
            <div class="text-[11px] text-muted-foreground mt-1">failures</div>
          </div>
          <div class="rounded-xl border border-border/45 bg-background/50 p-3 min-h-[104px]">
            <div class="text-xs text-muted-foreground">Total Trials</div>
            <div class="text-3xl font-semibold mt-1 text-cyan-500">{{ performanceMetrics.total }}</div>
            <div class="text-[11px] text-muted-foreground mt-1">interactions</div>
          </div>
        </div>

        <div class="mt-5">
          <div class="flex items-end justify-between gap-3">
            <div>
              <div class="text-xs font-medium text-muted-foreground">Success Rate Trend</div>
              <div class="text-[11px] text-muted-foreground mt-1">
                Last {{ rewardWindow }} rewards · success = reward=1
              </div>
            </div>
            <div class="text-xs text-muted-foreground">
              <span class="font-medium text-foreground">{{
                performanceMetrics.successRate == null ? '—' : `${Math.round(performanceMetrics.successRate * 100)}%`
              }}</span>
            </div>
          </div>

          <div class="mt-3 flex items-end gap-2">
            <div
              v-for="(rate, i) in successTrendBuckets"
              :key="`trend-${i}`"
              class="flex-1 rounded-md bg-muted/60 overflow-hidden"
              :style="{ height: `${Math.round(14 + rate * 42)}px` }"
            >
              <div
                class="h-full bg-gradient-to-t from-purple-500/75 via-cyan-500/55 to-emerald-500/55"
                :style="{ height: `${Math.round(rate * 100)}%` }"
              />
            </div>
          </div>
          <div class="mt-2 flex gap-2">
            <div v-for="(rate, i) in successTrendBuckets" :key="`trend-label-${i}`" class="flex-1 text-center text-[10px] text-muted-foreground">
              {{ Math.round(rate * 100) }}%
            </div>
          </div>
        </div>

        <div class="mt-5 space-y-2">
          <div class="text-xs font-medium text-muted-foreground">Vs. other strategies</div>
          <div v-if="strategyConfusionBars.length === 0" class="text-sm text-muted-foreground py-3">No feedback yet.</div>
          <div v-else class="space-y-2">
            <div v-for="s in strategyConfusionBars" :key="s.strategy" class="space-y-1">
              <div class="flex items-center justify-between text-xs text-muted-foreground gap-3">
                <span class="truncate">{{ s.strategy }}</span>
                <span class="font-medium text-foreground">{{ Math.round(s.winRate * 100) }}%</span>
              </div>
              <div class="h-2.5 rounded bg-muted/60 overflow-hidden">
                <div
                  class="h-full bg-gradient-to-r from-emerald-500/85 to-cyan-500/65"
                  :style="{ width: `${Math.round(s.winRate * 100)}%` }"
                />
              </div>
            </div>
          </div>
        </div>
      </Card>
      </Motion>

      <Motion tag="div" :initial="{ opacity: 0, y: 10 }" :animate="{ opacity: 1, y: 0 }" :transition="{ ...MOTION_BASE, delay: 0.1 }">
      <Card class="p-5 premium-reveal card-shape card-soft-border" style="--stagger: 0.1s">
        <div class="text-xs font-semibold tracking-widest uppercase text-muted-foreground">Why Users Didn’t Like It</div>

        <div class="mt-4 space-y-3">
          <div class="text-xs font-medium text-muted-foreground">Top negative reasons</div>
          <div v-if="negativeReasons.length === 0" class="text-sm text-muted-foreground py-3">No negative feedback yet.</div>
          <div v-else class="space-y-3">
            <div v-for="r in negativeReasons" :key="r.reason" class="space-y-1">
              <div class="flex items-center justify-between gap-3 text-xs text-muted-foreground">
                <span class="truncate">{{ r.reason }}</span>
                <span class="font-medium text-foreground">{{ r.count }}</span>
              </div>
              <div class="h-2.5 rounded bg-muted/60 overflow-hidden">
                <div
                  class="h-full bg-gradient-to-r from-red-500/85 to-amber-500/55"
                  :style="{
                    width: `${Math.round((r.count / Math.max(...negativeReasons.map((x) => x.count))) * 100)}%`,
                  }"
                />
              </div>
            </div>
          </div>
        </div>

        <div class="mt-5 rounded-xl border border-border/45 bg-background/60 p-3">
          <div class="text-xs font-semibold text-muted-foreground">AI-powered suggestion</div>
          <div class="mt-2 text-sm leading-relaxed">
            {{ topReasonSuggestion }}
          </div>
        </div>

        <div class="mt-5">
          <div class="flex items-center justify-between gap-3">
            <div class="text-xs font-medium text-muted-foreground">Feedback Source Breakdown</div>
            <div class="text-xs text-muted-foreground">
              Total: <span class="font-medium text-foreground">{{ feedbackSourceBreakdown.total }}</span>
            </div>
          </div>

          <div class="grid grid-cols-2 gap-3 mt-3">
            <div class="rounded-xl border border-border/45 bg-background/50 p-3">
              <div class="text-xs text-muted-foreground">Manual</div>
              <div class="text-xl font-semibold mt-1 text-amber-500">{{ feedbackSourceBreakdown.manual }}</div>
            </div>
            <div class="rounded-xl border border-border/45 bg-background/50 p-3">
              <div class="text-xs text-muted-foreground">Auto-inferred</div>
              <div class="text-xl font-semibold mt-1 text-cyan-500">{{ feedbackSourceBreakdown.auto }}</div>
            </div>
          </div>
        </div>
      </Card>
      </Motion>
    </div>

    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
      <Card v-for="(item, i) in [
        { label: 'Responses', value: String(metrics.totalResponses), delta: deltaLabel(metricDeltas.responses) },
        { label: 'Rewards', value: String(metrics.totalRewards), delta: deltaLabel(metricDeltas.rewards) },
        { label: 'Reward avg', value: metrics.rewardRate == null ? '—' : `${Math.round(metrics.rewardRate * 100)}%`, delta: deltaLabel(metricDeltas.rewardRate) },
        { label: 'Avg latency', value: metrics.avgLatency == null ? '—' : `${metrics.avgLatency.toFixed(2)}s`, delta: deltaLabel(metricDeltas.avgLatency) },
        { label: 'Widget attach rate', value: `${Math.round(metrics.widgetRate * 100)}%`, delta: deltaLabel(metricDeltas.widgetRate) },
      ]" :key="item.label" class="premium-card premium-reveal p-4 card-shape card-soft-border" :style="`--stagger: ${i * 0.06}s`">
        <div class="text-xs text-muted-foreground">{{ item.label }}</div>
        <div class="mt-1 text-[30px] leading-none font-semibold">{{ item.value }}</div>
        <div class="mt-1 text-[11px] text-muted-foreground">{{ item.delta }}</div>
        <div class="mt-2 h-7 flex items-end gap-1">
          <div
            v-for="(v, idx) in kpiSparkline"
            :key="`${item.label}-${idx}`"
            class="flex-1 rounded bg-cyan-500/35"
            :style="{ height: `${Math.max(8, Math.round((v / Math.max(1, ...kpiSparkline)) * 100))}%` }"
          />
        </div>
      </Card>
    </div>

    <Card class="p-5 card-shape card-soft-border">
      <div class="text-xs font-semibold tracking-widest uppercase text-muted-foreground">Quick takeaways</div>
      <div class="mt-3 grid grid-cols-1 md:grid-cols-3 gap-2">
        <div v-for="insight in dashboardInsights" :key="insight" class="rounded-xl border border-border/45 bg-background/50 p-3 text-sm text-muted-foreground">
          {{ insight }}
        </div>
      </div>
    </Card>

    <Card v-if="!hasFilteredData" class="p-6 text-center card-shape card-soft-border">
      <div class="text-sm font-medium">No data for this range</div>
      <div class="text-xs text-muted-foreground mt-1">Try switching to 7d/30d or clear the strategy filter.</div>
    </Card>

    <div v-else class="grid grid-cols-1 xl:grid-cols-2 gap-4">
      <Card class="premium-card premium-reveal p-5 card-shape card-soft-border"
        style="--stagger: 0.30s">
        <div class="text-sm font-medium mb-2">Strategy mix over time</div>
        <div ref="strategyMixEl" class="chart-frame w-full" />
      </Card>
      <Card class="premium-card premium-reveal p-5 card-shape card-soft-border"
        style="--stagger: 0.36s">
        <div class="text-sm font-medium mb-2">Reward trend (raw + rolling avg)</div>
        <div ref="rewardTrendEl" class="chart-frame w-full" />
      </Card>

      <Card class="premium-card premium-reveal p-5 card-shape card-soft-border"
        style="--stagger: 0.42s">
        <div class="text-sm font-medium mb-2">Latency (p50/p95 + volume)</div>
        <div ref="latencyTrendEl" class="chart-frame w-full" />
      </Card>
      <Card class="premium-card premium-reveal p-5 card-shape card-soft-border"
        style="--stagger: 0.48s">
        <div class="text-sm font-medium mb-2">Latency distribution</div>
        <div ref="latencyHistEl" class="chart-frame w-full" />
      </Card>

      <Card class="premium-card premium-reveal p-5 card-shape card-soft-border"
        style="--stagger: 0.54s">
        <div class="text-sm font-medium mb-2">Widget types</div>
        <div ref="widgetSplitEl" class="chart-frame w-full" />
      </Card>
      <Card class="premium-card premium-reveal p-5 card-shape card-soft-border"
        style="--stagger: 0.60s">
        <div class="text-sm font-medium mb-2">Throughput per minute</div>
        <div ref="throughputEl" class="chart-frame w-full" />
      </Card>
      <div class="xl:col-span-2 flex items-center justify-between rounded-2xl border border-border/55 bg-card/75 px-4 py-2.5 shadow-sm">
        <div class="text-sm font-medium">Advanced diagnostics</div>
        <button
          class="h-8 rounded-lg border bg-card/80 px-3 text-xs text-muted-foreground hover:text-foreground"
          @click="showAdvancedDiagnostics = !showAdvancedDiagnostics"
        >
          {{ showAdvancedDiagnostics ? 'Hide advanced' : 'Show advanced' }}
        </button>
      </div>
      <Card v-if="showAdvancedDiagnostics" class="premium-card premium-reveal p-5 card-shape card-soft-border"
        style="--stagger: 0.63s">
        <div class="text-sm font-medium mb-2">Cumulative reward (time series)</div>
        <div class="text-xs text-muted-foreground mb-2">
          Running wins over reward events.
        </div>
        <div ref="cumulativeRewardEl" class="chart-frame w-full" />
      </Card>
      <Card v-if="showAdvancedDiagnostics" class="premium-card premium-reveal p-5 card-shape card-soft-border"
        style="--stagger: 0.66s">
        <div class="text-sm font-medium mb-2">Strategy performance heatmap</div>
        <div class="text-xs text-muted-foreground mb-2">
          Avg observed reward by time bucket (realized signals).
        </div>
        <div ref="heatmapEl" class="chart-frame w-full" />
      </Card>
      <Card v-if="showAdvancedDiagnostics" class="premium-card premium-reveal p-5 card-shape card-soft-border"
        style="--stagger: 0.72s">
        <div class="text-sm font-medium mb-2">Calibration curve</div>
        <div class="text-xs text-muted-foreground mb-2">
          Predicted confidence (`r`) vs realized win-rate.
        </div>
        <div ref="calibrationEl" class="chart-frame w-full" />
      </Card>
      <Card class="premium-card premium-reveal p-5 card-shape card-soft-border"
        style="--stagger: 0.75s">
        <div class="text-sm font-medium mb-2">Win rate by strategy (bar)</div>
        <div class="text-xs text-muted-foreground mb-2">
          Top strategies ranked by realized win rate.
        </div>
        <div ref="winRateByStrategyEl" class="chart-frame w-full" />
      </Card>
      <Card v-if="showAdvancedDiagnostics" class="premium-card premium-reveal p-5 card-shape card-soft-border"
        style="--stagger: 0.78s">
        <div class="text-sm font-medium mb-2">Latency vs reward (scatter)</div>
        <div class="text-xs text-muted-foreground mb-2">
          Diagnostic view to spot quality-speed tradeoffs.
        </div>
        <div ref="latencyScatterEl" class="chart-frame w-full" />
      </Card>
    </div>

    <Card v-if="hasFilteredData" class="premium-card premium-reveal p-5 card-shape card-soft-border"
      style="--stagger: 0.84s">
      <div class="flex items-center justify-between mb-3">
        <h2
          class="text-sm font-semibold text-transparent bg-clip-text bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500"
        >
          Per-strategy performance
        </h2>
        <div class="text-xs text-muted-foreground">Counts from recent responses + all feedback signals</div>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left border-b">
              <th class="py-2 pr-3 font-medium text-muted-foreground">Strategy</th>
              <th class="py-2 pr-3 font-medium text-muted-foreground">Response count</th>
              <th class="py-2 pr-3 font-medium text-muted-foreground">Positive</th>
              <th class="py-2 pr-3 font-medium text-muted-foreground">Negative</th>
              <th class="py-2 pr-3 font-medium text-muted-foreground">Reward rate</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in strategyTable" :key="row.strategy" class="border-b last:border-0">
              <td class="py-2 pr-3">{{ getStrategyLabel(row.strategy) }}</td>
              <td class="py-2 pr-3">{{ row.count }}</td>
              <td class="py-2 pr-3 text-emerald-400">{{ row.pos }}</td>
              <td class="py-2 pr-3 text-red-400">{{ row.neg }}</td>
              <td class="py-2 pr-3">
                {{ row.rewardRate == null ? '—' : `${Math.round(row.rewardRate * 100)}%` }}
              </td>
            </tr>
            <tr v-if="strategyTable.length === 0">
              <td class="py-4 text-muted-foreground" colspan="5">No analytics yet. Send a few messages and feedback.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </Card>
  </div>
</template>

<style scoped>
@keyframes premiumFadeIn {
  from {
    opacity: 0;
    filter: blur(6px);
  }
  to {
    opacity: 1;
    filter: blur(0);
  }
}

.premium-reveal {
  animation: premiumFadeIn 650ms cubic-bezier(0.2, 0.8, 0.2, 1) both;
  animation-delay: var(--stagger, 0s);
}

.premium-card {
  will-change: transform;
}

.card-shape {
  border-radius: 1rem;
}

.card-soft-border {
  border-color: color-mix(in oklab, var(--border) 85%, transparent);
}

.chart-frame {
  height: 290px;
}

@media (min-width: 1280px) {
  .chart-frame {
    height: 310px;
  }
}
</style>

