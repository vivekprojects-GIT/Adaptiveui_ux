import { reactive } from 'vue'

export type DoneEventForAnalytics = {
  ts: number
  strategy: string
  elapsed: number | null | undefined
  widgetHtml: string
}

export type RewardEventForAnalytics = {
  ts: number
  strategy: string
  reward: number
  /**
   * Predicted confidence for this (0..1) at the time the model produced the response.
   * Used for calibration curve. Optional to keep backward compatibility.
   */
  predictedR?: number | null
  /**
   * Posterior uncertainty scalar (u). Optional (for future uncertainty diagnostics).
   */
  predictedU?: number | null
}

export const analyticsState = reactive({
  doneEvents: [] as DoneEventForAnalytics[],
  rewardEvents: [] as RewardEventForAnalytics[],
})

const MAX_EVENTS = 300

export function ingestDone(evt: {
  strategy: string
  elapsed?: number | null
  widget_html?: string
  widgetHtml?: string
}) {
  const widgetHtml = (evt.widgetHtml ?? evt.widget_html ?? '') as string
  analyticsState.doneEvents.push({
    ts: Date.now(),
    strategy: evt.strategy,
    elapsed: evt.elapsed,
    widgetHtml,
  })
  if (analyticsState.doneEvents.length > MAX_EVENTS) {
    analyticsState.doneEvents.splice(0, analyticsState.doneEvents.length - MAX_EVENTS)
  }
}

export function ingestReward(evt: { strategy: string; reward: number; predictedR?: number | null; predictedU?: number | null }) {
  analyticsState.rewardEvents.push({
    ts: Date.now(),
    strategy: evt.strategy,
    reward: evt.reward,
    predictedR: evt.predictedR ?? null,
    predictedU: evt.predictedU ?? null,
  })
  if (analyticsState.rewardEvents.length > MAX_EVENTS) {
    analyticsState.rewardEvents.splice(0, analyticsState.rewardEvents.length - MAX_EVENTS)
  }
}

export function computeWidgetRenderRate(): number {
  const ds = analyticsState.doneEvents
  if (!ds.length) return 0
  const withWidgets = ds.filter((d) => (d.widgetHtml ?? '').trim().length > 0).length
  return withWidgets / ds.length
}

export function computeAvgResponseTimeSec(): number | null {
  const ds = analyticsState.doneEvents.filter((d) => typeof d.elapsed === 'number' && Number.isFinite(d.elapsed))
  if (!ds.length) return null
  const sum = ds.reduce((acc, d) => acc + (d.elapsed as number), 0)
  return sum / ds.length
}

export function computeStrategyCounts(windowSize: number): Record<string, number> {
  const ds = analyticsState.doneEvents.slice(-windowSize)
  const out: Record<string, number> = {}
  for (const d of ds) {
    out[d.strategy] = (out[d.strategy] ?? 0) + 1
  }
  return out
}

export function computeLastResponseTimes(windowSize: number): { idx: number; elapsed: number }[] {
  const ds = analyticsState.doneEvents
    .filter((d) => typeof d.elapsed === 'number' && Number.isFinite(d.elapsed))
    .slice(-windowSize)
  return ds.map((d, i) => ({ idx: i + 1, elapsed: d.elapsed as number }))
}

export function computeRewardRateLast(windowSize: number): number | null {
  const rs = analyticsState.rewardEvents.slice(-windowSize)
  if (!rs.length) return null
  const positives = rs.filter((r) => r.reward >= 1).length
  return positives / rs.length
}

export function computeRewardPieLast(windowSize: number): { pos: number; neg: number } {
  const rs = analyticsState.rewardEvents.slice(-windowSize)
  const pos = rs.filter((r) => r.reward >= 1).length
  return { pos, neg: rs.length - pos }
}

