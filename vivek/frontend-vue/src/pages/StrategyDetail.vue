<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import Plotly from 'plotly.js-dist-min'
import Card from '@/components/ui/Card.vue'
import Button from '@/components/ui/Button.vue'
import Input from '@/components/ui/Input.vue'
import { getAccessToken } from '@/lib/auth'
import { showToast } from '@/lib/toast'
import { fetchStrategies, resetStrategies } from '@/lib/strategiesStore'

type StrategyItem = { id: string; label: string; instruction: string; enabled: boolean; event_name?: string }
type UsageEntry = { wins: number; trials: number; last_ts: number | null }
type PosteriorEntry = { r: number; u: number }
type VersionEntry = { ts: number; label: string; instruction: string; enabled: boolean }
type StrategyAnalyticsResp = {
  strategy_id: string
  days: number
  labels: string[]
  usage_series: number[]
  summary: { total_usage: number; wins: number; trials: number }
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5051'
const HISTORY_KEY = 'prism_strategy_version_history_v1'
const MAX_HISTORY_PER_STRATEGY = 20
const EVENT_TYPE_SUGGESTIONS = ['Decision', 'Confusion', 'Follow-up', 'Summary', 'Escalation'] as const

const route = useRoute()
const router = useRouter()

const loading = ref(false)
const item = ref<StrategyItem | null>(null)
const usage = ref<UsageEntry>({ wins: 0, trials: 0, last_ts: null })
const posterior = ref<PosteriorEntry>({ r: 0.5, u: 1.0 })
const allPosterior = ref<Record<string, PosteriorEntry>>({})
const analytics = ref<StrategyAnalyticsResp | null>(null)
const chartMode = ref<'line' | 'bar'>('line')
const chartDays = ref<7 | 14 | 30>(30)
const chartRef = ref<HTMLElement | null>(null)
const previewMode = ref(false)
const saving = ref(false)
const edit = ref<{ label: string; instruction: string; enabled: boolean; event_name: string } | null>(null)
const history = ref<Record<string, VersionEntry[]>>({})

const strategyId = computed(() => String(route.params.id ?? '').trim())
const successPct = computed(() => {
  if (usage.value.trials > 0) return (usage.value.wins / usage.value.trials) * 100
  return posterior.value.r * 100
})
const globalAvgSuccessPct = computed(() => {
  const vals = Object.values(allPosterior.value).map((x) => Number(x?.r ?? 0.5) * 100)
  if (!vals.length) return 50
  return vals.reduce((acc, n) => acc + n, 0) / vals.length
})
const upliftVsGlobal = computed(() => successPct.value - globalAvgSuccessPct.value)
const reliabilityScore = computed(() => {
  const trialWeight = Math.min(1, usage.value.trials / 12)
  const confidenceWeight =
    confidence.value === 'High' ? 1 : confidence.value === 'Medium' ? 0.8 : confidence.value === 'Need review' ? 0.55 : 0.35
  return Math.round(((trialWeight * 0.6 + confidenceWeight * 0.4) * 100))
})
function confidenceFromSuccess(success: number): 'High' | 'Medium' | 'Need review' | 'Failing' {
  if (success > 80) return 'High'
  if (success > 50 && success <= 80) return 'Medium'
  if (success > 40 && success <= 50) return 'Need review'
  return 'Failing'
}
const confidence = computed(() => {
  return confidenceFromSuccess(successPct.value)
})

function authHeaders(json = false): HeadersInit {
  const token = getAccessToken()
  const h: Record<string, string> = {}
  if (json) h['Content-Type'] = 'application/json'
  if (token) h.Authorization = `Bearer ${token}`
  return h
}

function formatLastUsed(ts: number | null): string {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleString()
}

async function renderTrendChart() {
  if (!chartRef.value || !analytics.value) return
  const labels = analytics.value.labels
  const vals = analytics.value.usage_series
  const yMax = Math.max(1, ...vals.map((n) => Number(n || 0)))
  const hasActivity = vals.some((n) => Number(n || 0) > 0)
  const data =
    chartMode.value === 'bar'
      ? [
          {
            type: 'bar',
            x: labels,
            y: vals,
            marker: { color: '#22c55e' },
            name: 'Usage',
          },
        ]
      : [
          {
            type: 'scatter',
            mode: 'lines+markers',
            x: labels,
            y: vals,
            line: { color: '#f59e0b', width: 3, shape: 'spline' },
            marker: { size: 6, color: '#f59e0b' },
            name: 'Usage',
          },
        ]

  const layout = {
    margin: { l: 40, r: 12, t: 18, b: 34 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    xaxis: { tickfont: { size: 10 } },
    yaxis: { tickfont: { size: 10 }, rangemode: 'tozero', range: [0, yMax] },
    showlegend: false,
    dragmode: 'pan',
    annotations: hasActivity
      ? []
      : [
          {
            text: 'No activity in selected range',
            x: 0.5,
            y: 0.5,
            xref: 'paper',
            yref: 'paper',
            showarrow: false,
            font: { size: 12, color: '#94a3b8' },
          },
        ],
  }
  const config = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }
  await Plotly.react(chartRef.value, data as any, layout as any, config as any)
}

function saveHistorySnapshot(current: StrategyItem) {
  const cur = history.value[current.id] ?? []
  history.value = {
    ...history.value,
    [current.id]: [
      { ts: Date.now(), label: current.label, instruction: current.instruction, enabled: current.enabled },
      ...cur,
    ].slice(0, MAX_HISTORY_PER_STRATEGY),
  }
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.value))
  } catch {
    // best effort
  }
}

function restoreVersion(v: VersionEntry) {
  edit.value = { label: v.label, instruction: v.instruction, enabled: v.enabled, event_name: item.value?.event_name || 'Decision' }
  previewMode.value = false
}

async function refreshGlobalStrategyStore() {
  resetStrategies()
  await fetchStrategies(API_BASE)
}

async function loadData() {
  if (!strategyId.value) return
  loading.value = true
  try {
    const [resStrategies, resUsage, resState, resAnalytics] = await Promise.all([
      fetch(`${API_BASE}/api/strategies`, { headers: authHeaders() }),
      fetch(`${API_BASE}/api/strategies/usage`, { headers: authHeaders() }),
      fetch(`${API_BASE}/api/state`, { headers: authHeaders() }),
      fetch(`${API_BASE}/api/strategies/${encodeURIComponent(strategyId.value)}/analytics?days=${chartDays.value}`, { headers: authHeaders() }),
    ])
    const dStrategies = await resStrategies.json().catch(() => ({}))
    const dUsage = await resUsage.json().catch(() => ({}))
    const dState = await resState.json().catch(() => ({}))
    const dAnalytics = await resAnalytics.json().catch(() => ({}))
    if (!resStrategies.ok) throw new Error(dStrategies?.detail || 'Failed to load strategies')
    const items = Array.isArray(dStrategies?.items) ? (dStrategies.items as StrategyItem[]) : []
    const found = items.find((x) => x.id === strategyId.value) ?? null
    if (!found) {
      showToast({ title: 'Not found', message: `Strategy "${strategyId.value}" is unavailable.` })
      router.push('/app/strategies')
      return
    }
    item.value = found
    edit.value = {
      label: found.label,
      instruction: found.instruction,
      enabled: found.enabled,
      event_name: String(found.event_name || 'Decision'),
    }
    usage.value = dUsage?.by_id?.[strategyId.value] ?? { wins: 0, trials: 0, last_ts: null }
    allPosterior.value = typeof dState?.global === 'object' && dState?.global ? dState.global : {}
    posterior.value = allPosterior.value[strategyId.value] ?? { r: 0.5, u: 1.0 }
    analytics.value = resAnalytics.ok
      ? {
          strategy_id: String(dAnalytics?.strategy_id ?? strategyId.value),
          days: Number(dAnalytics?.days ?? chartDays.value),
          labels: Array.isArray(dAnalytics?.labels) ? dAnalytics.labels : [],
          usage_series: Array.isArray(dAnalytics?.usage_series) ? dAnalytics.usage_series.map((n: unknown) => Number(n || 0)) : [],
          summary: {
            total_usage: Number(dAnalytics?.summary?.total_usage ?? 0),
            wins: Number(dAnalytics?.summary?.wins ?? 0),
            trials: Number(dAnalytics?.summary?.trials ?? 0),
          },
        }
      : null
    await nextTick()
    await renderTrendChart()
  } catch (e) {
    showToast({ title: 'Failed to load', message: e instanceof Error ? e.message : 'Request failed' })
  } finally {
    loading.value = false
  }
}

async function saveChanges() {
  if (!item.value || !edit.value) return
  saving.value = true
  try {
    saveHistorySnapshot(item.value)
    const res = await fetch(`${API_BASE}/api/strategies/${item.value.id}/`, {
      method: 'PUT',
      headers: authHeaders(true),
      body: JSON.stringify({
        label: edit.value.label.trim(),
        instruction: edit.value.instruction.trim(),
        enabled: Boolean(edit.value.enabled),
        event_name: String(edit.value.event_name || '').trim(),
      }),
    })
    const d = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(d?.detail || 'Update failed')
    showToast({ title: 'Saved', message: 'Strategy updated successfully.' })
    await refreshGlobalStrategyStore()
    await loadData()
  } catch (e) {
    showToast({ title: 'Save failed', message: e instanceof Error ? e.message : 'Request failed' })
  } finally {
    saving.value = false
  }
}

async function setEnabled(enabled: boolean) {
  if (!item.value) return
  try {
    const endpoint = enabled ? 'enable' : 'disable'
    const res = await fetch(`${API_BASE}/api/strategies/${item.value.id}/${endpoint}/`, {
      method: 'POST',
      headers: authHeaders(),
    })
    const d = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(d?.detail || 'Update failed')
    await refreshGlobalStrategyStore()
    await loadData()
  } catch (e) {
    showToast({ title: 'Update failed', message: e instanceof Error ? e.message : 'Request failed' })
  }
}

async function deleteStrategy() {
  if (!item.value) return
  const ok = confirm(`Delete strategy "${item.value.id}" permanently?`)
  if (!ok) return
  try {
    const res = await fetch(`${API_BASE}/api/strategies/${item.value.id}/`, {
      method: 'DELETE',
      headers: authHeaders(),
    })
    const d = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(d?.detail || 'Delete failed')
    showToast({ title: 'Deleted', message: 'Strategy removed successfully.' })
    await refreshGlobalStrategyStore()
    router.push('/app/strategies')
  } catch (e) {
    showToast({ title: 'Delete failed', message: e instanceof Error ? e.message : 'Request failed' })
  }
}

onMounted(async () => {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    if (raw) history.value = JSON.parse(raw) as Record<string, VersionEntry[]>
  } catch {
    history.value = {}
  }
  await loadData()
})

onUnmounted(() => {
  if (chartRef.value) {
    Plotly.purge(chartRef.value)
  }
})

watch(
  () => strategyId.value,
  async () => {
    await loadData()
  },
)

watch(
  () => [chartMode.value, analytics.value?.labels?.length, analytics.value?.usage_series?.length],
  async () => {
    await nextTick()
    await renderTrendChart()
  },
)
</script>

<template>
  <div class="max-w-7xl mx-auto page-shell page-shell-analytics">
    <div class="space-y-4 p-5 lg:p-7 page-content">
    <div class="flex items-center justify-between gap-3 flex-wrap">
      <Button variant="outline" class="h-9" @click="router.push('/app/strategies')">← Back to Strategies</Button>
      <Button variant="outline" class="h-9" @click="loadData" :disabled="loading">{{ loading ? 'Refreshing…' : 'Refresh' }}</Button>
    </div>

    <Card class="p-4 detail-card premium-card">
      <div v-if="item" class="space-y-3">
        <div>
          <div class="text-xs uppercase tracking-wide text-muted-foreground">Strategy detail</div>
          <div class="text-xl font-semibold">{{ item.label }}</div>
          <div class="text-[12px] font-mono text-muted-foreground">{{ item.id }}</div>
        </div>
        <div class="grid grid-cols-2 lg:grid-cols-5 gap-3">
          <div class="rounded-lg border border-border/70 bg-card/70 p-3">
            <div class="text-xs text-muted-foreground">Wins / Trials</div>
            <div class="mt-1 text-lg font-semibold text-emerald-600 dark:text-emerald-300">{{ usage.wins }} / {{ usage.trials }}</div>
          </div>
          <div class="rounded-lg border border-border/70 bg-card/70 p-3">
            <div class="text-xs text-muted-foreground">Success rate</div>
            <div class="mt-1 text-lg font-semibold text-amber-600 dark:text-amber-200">{{ Math.round(successPct) }}%</div>
          </div>
          <div class="rounded-lg border border-border/70 bg-card/70 p-3">
            <div class="text-xs text-muted-foreground">Confidence</div>
            <div
              class="mt-1 text-lg font-semibold"
              :class="
                confidence === 'High'
                  ? 'text-emerald-700 dark:text-emerald-300'
                  : confidence === 'Medium'
                    ? 'text-cyan-700 dark:text-cyan-300'
                    : confidence === 'Need review'
                      ? 'text-amber-700 dark:text-amber-300'
                      : 'text-rose-700 dark:text-rose-300'
              "
            >
              {{ confidence }}
            </div>
          </div>
          <div class="rounded-lg border border-border/70 bg-card/70 p-3">
            <div class="text-xs text-muted-foreground">Posterior r / u</div>
            <div class="mt-1 text-lg font-semibold text-foreground">{{ posterior.r.toFixed(3) }} / {{ posterior.u.toFixed(3) }}</div>
          </div>
          <div class="rounded-lg border border-border/70 bg-card/70 p-3">
            <div class="text-xs text-muted-foreground">Last used</div>
            <div class="mt-1 text-sm font-medium text-foreground">{{ formatLastUsed(usage.last_ts) }}</div>
          </div>
          <div class="rounded-lg border border-border/70 bg-card/70 p-3">
            <div class="text-xs text-muted-foreground">Event name</div>
            <div class="mt-1 text-sm font-medium text-foreground">{{ item.event_name || '—' }}</div>
          </div>
        </div>
      </div>
      <div v-else class="text-sm text-muted-foreground">Loading strategy details...</div>
    </Card>

    <Card v-if="item" class="p-4 detail-card premium-card">
      <div class="text-sm font-medium mb-3">Analytics</div>
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <div class="rounded-lg border border-border/70 bg-card/70 p-3">
          <div class="text-xs text-muted-foreground">Uplift vs strategy average</div>
          <div class="mt-1 text-xl font-semibold" :class="upliftVsGlobal >= 0 ? 'text-emerald-600 dark:text-emerald-300' : 'text-rose-600 dark:text-rose-300'">
            {{ upliftVsGlobal >= 0 ? '+' : '' }}{{ upliftVsGlobal.toFixed(1) }}%
          </div>
          <div class="text-xs text-muted-foreground mt-1">Compared against global posterior mean across strategies.</div>
        </div>
        <div class="rounded-lg border border-border/70 bg-card/70 p-3">
          <div class="text-xs text-muted-foreground">Reliability score</div>
          <div class="mt-1 text-xl font-semibold text-cyan-700 dark:text-cyan-200">{{ reliabilityScore }}/100</div>
          <div class="mt-2 h-2 rounded-full bg-muted overflow-hidden">
            <div class="h-full rounded-full bg-gradient-to-r from-cyan-500 to-emerald-500" :style="{ width: `${reliabilityScore}%` }" />
          </div>
        </div>
        <div class="rounded-lg border border-border/70 bg-card/70 p-3">
          <div class="text-xs text-muted-foreground">Signal source</div>
          <div class="mt-1 text-sm font-medium text-foreground">{{ usage.trials > 0 ? 'Observed feedback (wins/trials)' : 'Posterior estimate (no trials yet)' }}</div>
          <div class="text-xs text-muted-foreground mt-1">Confidence bands are relative to current strategy set.</div>
        </div>
      </div>
      <div class="mt-3 rounded-lg border border-border/70 bg-card/70 p-3">
        <div class="flex items-center justify-between gap-2 flex-wrap mb-2">
          <div class="text-xs text-muted-foreground">Performance trend (interactive)</div>
          <div class="flex items-center gap-2">
            <select v-model="chartDays" class="h-8 rounded-md border border-input bg-background px-2 text-xs text-foreground" @change="loadData">
              <option :value="7">7d</option>
              <option :value="14">14d</option>
              <option :value="30">30d</option>
            </select>
            <Button size="sm" variant="outline" class="h-8 px-3" @click="chartMode = 'line'">Line</Button>
            <Button size="sm" variant="outline" class="h-8 px-3" @click="chartMode = 'bar'">Bar</Button>
          </div>
        </div>
        <div ref="chartRef" class="h-[280px] w-full" />
      </div>
    </Card>

    <Card v-if="item && edit" class="p-4 detail-card premium-card">
      <div class="flex items-center justify-between gap-3 flex-wrap mb-3">
        <div class="text-sm font-medium">Instruction editor</div>
        <div class="flex items-center gap-2">
          <Button variant="outline" class="h-9" @click="previewMode = false">Edit</Button>
          <Button variant="outline" class="h-9" @click="previewMode = true">Preview</Button>
          <Button variant="outline" class="h-9" @click="setEnabled(!item.enabled)">{{ item.enabled ? 'Disable' : 'Enable' }}</Button>
          <Button variant="destructive" class="h-9" @click="deleteStrategy">Delete</Button>
        </div>
      </div>

      <div class="space-y-3">
        <label class="block" v-if="!previewMode">
          <div class="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1">Label</div>
          <Input v-model="edit.label" placeholder="Label" />
        </label>

        <label class="block" v-if="!previewMode">
          <div class="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1">Event name</div>
          <Input v-model="edit.event_name" list="event-name-suggestions-detail" placeholder="Enter event type name" />
          <datalist id="event-name-suggestions-detail">
            <option v-for="ev in EVENT_TYPE_SUGGESTIONS" :key="ev" :value="ev" />
          </datalist>
        </label>

        <label class="flex items-center gap-2 cursor-pointer select-none" v-if="!previewMode">
          <input type="checkbox" class="rounded border-input" v-model="edit.enabled" />
          <span class="text-sm text-muted-foreground">Enabled</span>
        </label>

        <div>
          <div class="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1">Instruction</div>
          <textarea v-if="!previewMode" v-model="edit.instruction" class="w-full min-h-[240px] rounded-lg border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2" />
          <pre v-else class="whitespace-pre-wrap leading-relaxed rounded-lg border border-border bg-card/75 p-4 text-sm">{{ edit.instruction }}</pre>
        </div>
      </div>

      <div class="flex justify-end gap-2 mt-4">
        <Button variant="outline" class="h-9" @click="loadData">Reset</Button>
        <Button class="h-9" :disabled="saving || previewMode || !edit.label.trim() || !edit.instruction.trim()" @click="saveChanges">
          {{ saving ? 'Saving…' : 'Save Changes' }}
        </Button>
      </div>
    </Card>

    <Card v-if="item" class="p-4 detail-card premium-card">
      <div class="flex items-center justify-between mb-2">
        <div>
          <div class="text-sm font-medium">Version History</div>
          <div class="text-xs text-muted-foreground mt-1">Stored locally on this browser when you save changes.</div>
        </div>
      </div>
      <div v-if="(history[item.id] ?? []).length === 0" class="text-sm text-muted-foreground py-4">No versions yet.</div>
      <div v-else class="space-y-2 max-h-[260px] overflow-y-auto pr-1">
        <div v-for="v in history[item.id]" :key="v.ts" class="rounded-lg border border-border bg-card/75 p-3">
          <div class="flex items-start justify-between gap-3">
            <div class="min-w-0">
              <div class="text-xs text-muted-foreground uppercase tracking-wide">Saved</div>
              <div class="mt-1 text-sm font-medium">{{ new Date(v.ts).toLocaleString() }}</div>
              <div class="text-xs text-muted-foreground mt-1 font-mono truncate">{{ v.label }}</div>
            </div>
            <Button type="button" variant="outline" class="h-8 px-3" @click="restoreVersion(v)">Restore</Button>
          </div>
          <div class="mt-2 text-xs text-muted-foreground whitespace-pre-wrap leading-relaxed line-clamp-4">{{ v.instruction }}</div>
        </div>
      </div>
    </Card>
    </div>
  </div>
</template>

<style scoped>
.detail-card {
  background: color-mix(in oklab, var(--card) 94%, white 6%);
}

:global(.dark) .detail-card {
  background: rgba(15, 23, 42, 0.52);
}
</style>
