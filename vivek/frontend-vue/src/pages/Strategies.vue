<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import Card from '@/components/ui/Card.vue'
import Input from '@/components/ui/Input.vue'
import Button from '@/components/ui/Button.vue'
import { getAccessToken } from '@/lib/auth'
import { showToast } from '@/lib/toast'
import { fetchStrategies, resetStrategies } from '@/lib/strategiesStore'

type StrategyItem = { id: string; label: string; instruction: string; enabled: boolean; event_name?: string }
type PosteriorEntry = { r: number; u: number }
type UsageEntry = { wins: number; trials: number; last_ts: number | null }
type RowStatus = 'Healthy' | 'Review' | 'Failing'
type ConfidenceLevel = 'High' | 'Medium' | 'Need review' | 'Failing'
type SortBy = 'status' | 'success' | 'label' | 'last_used'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ((typeof location !== 'undefined' && /^(localhost|127.0.0.1)$/.test(location.hostname)) ? 'http://localhost:5051' : '')
const EVENT_TYPE_SUGGESTIONS = ['Decision', 'Confusion', 'Follow-up', 'Summary', 'Escalation'] as const

const router = useRouter()
const loading = ref(false)
const items = ref<StrategyItem[]>([])
const isAdmin = ref<boolean | null>(null)
const globalPosterior = ref<Record<string, PosteriorEntry>>({})
const usageById = ref<Record<string, UsageEntry>>({})
const selectedMap = ref<Record<string, boolean>>({})
const showCreate = ref(false)
const query = ref('')
const eventFilter = ref<string>('All')
const statusFilter = ref<'All' | RowStatus>('All')
const sortBy = ref<SortBy>('status')
const newId = ref('')
const newLabel = ref('')
const newInstruction = ref('')
const newEnabled = ref(true)
const newEventName = ref<string>('Decision')

function authHeaders(json = false): HeadersInit {
  const token = getAccessToken()
  const h: Record<string, string> = {}
  if (json) h['Content-Type'] = 'application/json'
  if (token) h.Authorization = `Bearer ${token}`
  return h
}

function hashString(input: string): number {
  let h = 2166136261
  for (let i = 0; i < input.length; i += 1) {
    h ^= input.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

function seededPoints(seed: number, baseline: number): number[] {
  let x = seed || 1
  const out: number[] = []
  for (let i = 0; i < 16; i += 1) {
    x = (x * 1664525 + 1013904223) >>> 0
    const noise = ((x % 1000) / 1000 - 0.5) * 18
    out.push(Math.max(6, Math.min(94, baseline + noise)))
  }
  return out
}

function sparklinePath(points: number[]): string {
  const width = 110
  const height = 28
  return points.map((v, i) => `${i === 0 ? 'M' : 'L'}${((i / (points.length - 1)) * width).toFixed(1)},${(height - (v / 100) * height).toFixed(1)}`).join(' ')
}

function eventTypeFor(id: string): string {
  return EVENT_TYPE_SUGGESTIONS[hashString(id) % EVENT_TYPE_SUGGESTIONS.length]
}

function eventTypeForItem(item: StrategyItem): string {
  const raw = String(item.event_name || '').trim()
  if (raw) return raw
  return eventTypeFor(item.id)
}

function statusForRow(enabled: boolean, _successPct: number, confidence: ConfidenceLevel, _trials: number): RowStatus {
  if (!enabled) return 'Failing'
  if (confidence === 'Failing') return 'Failing'
  if (confidence === 'Need review') return 'Review'
  return 'Healthy'
}

function confidenceFromSuccess(successPct: number): ConfidenceLevel {
  // As requested: >80 High, >50 to <=80 Medium, >40 to <=50 Need review, <40 Failing.
  if (successPct > 80) return 'High'
  if (successPct > 50 && successPct <= 80) return 'Medium'
  if (successPct > 40 && successPct <= 50) return 'Need review'
  return 'Failing'
}

const canCreate = computed(() => newId.value.trim() && newLabel.value.trim() && newInstruction.value.trim())
const selectedCount = computed(() => Object.values(selectedMap.value).filter(Boolean).length)
const eventFilterOptions = computed(() => {
  const opts = new Set<string>(EVENT_TYPE_SUGGESTIONS)
  for (const it of items.value) {
    const ev = eventTypeForItem(it).trim()
    if (ev) opts.add(ev)
  }
  return Array.from(opts).sort((a, b) => a.localeCompare(b))
})

const tableRows = computed(() => {
  const q = query.value.trim().toLowerCase()
  const rawRows = items.value.map((item) => {
    const usage = usageById.value[item.id] ?? { wins: 0, trials: 0, last_ts: null }
    const posterior = globalPosterior.value[item.id] ?? { r: 0.5, u: 1.0 }
    return {
      item,
      usage,
      posterior,
      successPct: usage.trials > 0 ? (usage.wins / usage.trials) * 100 : posterior.r * 100,
      eventType: eventTypeForItem(item),
      sparkline: [] as number[],
      confidence: 'Failing' as ConfidenceLevel,
      status: 'Review' as RowStatus,
    }
  })

  const rows = rawRows.map((row) => {
    const confidence = confidenceFromSuccess(row.successPct)
    const sparkline = seededPoints(hashString(row.item.id), row.successPct)
    return {
      ...row,
      confidence,
      sparkline,
      status: statusForRow(row.item.enabled, row.successPct, confidence, row.usage.trials),
    }
  })
  const filtered = rows.filter((row) => {
    if (q && !`${row.item.id} ${row.item.label} ${row.item.instruction}`.toLowerCase().includes(q)) return false
    if (eventFilter.value !== 'All' && row.eventType !== eventFilter.value) return false
    if (statusFilter.value !== 'All' && row.status !== statusFilter.value) return false
    return true
  })
  filtered.sort((a, b) => {
    if (sortBy.value === 'label') return a.item.label.localeCompare(b.item.label)
    if (sortBy.value === 'success') return b.successPct - a.successPct
    if (sortBy.value === 'last_used') return (b.usage.last_ts ?? 0) - (a.usage.last_ts ?? 0)
    const rank = (s: RowStatus) => (s === 'Failing' ? 0 : s === 'Review' ? 1 : 2)
    return rank(a.status) - rank(b.status) || b.successPct - a.successPct
  })
  return filtered
})

const healthyCount = computed(() => tableRows.value.filter((x) => x.status === 'Healthy').length)
const reviewCount = computed(() => tableRows.value.filter((x) => x.status === 'Review').length)
const failingCount = computed(() => tableRows.value.filter((x) => x.status === 'Failing').length)
const avgSuccessRate = computed(() => (tableRows.value.length ? tableRows.value.reduce((acc, row) => acc + row.successPct, 0) / tableRows.value.length : 0))

function formatLastUsed(ts: number | null): string {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleDateString(undefined, { month: 'short', day: '2-digit', year: 'numeric' })
}

function successRateDetails(row: {
  usage: { wins: number; trials: number }
  successPct: number
  posterior: { r: number; u: number }
}) {
  const pct = Math.round(row.successPct)
  if (row.usage.trials > 0) {
    return `Observed success from feedback: ${row.usage.wins}/${row.usage.trials} = ${pct}%.`
  }
  const estPct = Math.round((row.posterior.r ?? 0.5) * 100)
  return `Estimated success from posterior (no explicit trials yet): ${estPct}% (shown as estimate).`
}

function statusDetails(row: {
  status: RowStatus
  confidence: ConfidenceLevel
  item: { enabled: boolean }
  successPct: number
}) {
  const pct = Math.round(row.successPct)
  if (!row.item.enabled) {
    return `Status: Failing. Reason: strategy is disabled.`
  }
  if (row.status === 'Healthy') {
    return `Status: Healthy. Success ${pct}%, confidence ${row.confidence}.`
  }
  if (row.status === 'Review') {
    return `Status: Review. Success ${pct}%, confidence ${row.confidence}. Consider refining prompt/instruction.`
  }
  return `Status: Failing. Success ${pct}%, confidence ${row.confidence}. Needs immediate correction.`
}

async function loadMe() {
  const token = getAccessToken()
  if (!token) return
  try {
    const r = await fetch(`${API_BASE}/api/me`, { headers: { Authorization: `Bearer ${token}` } })
    const d = await r.json().catch(() => ({}))
    if (r.ok) isAdmin.value = Boolean(d?.is_admin)
  } catch {
    isAdmin.value = false
  }
}

async function loadAll() {
  if (!isAdmin.value) return
  loading.value = true
  try {
    const [resStrategies, resUsage, resState] = await Promise.all([
      fetch(`${API_BASE}/api/strategies`, { headers: authHeaders() }),
      fetch(`${API_BASE}/api/strategies/usage`, { headers: authHeaders() }),
      fetch(`${API_BASE}/api/state`, { headers: authHeaders() }),
    ])
    const dStrategies = await resStrategies.json().catch(() => ({}))
    const dUsage = await resUsage.json().catch(() => ({}))
    const dState = await resState.json().catch(() => ({}))
    if (!resStrategies.ok) throw new Error(dStrategies?.detail || 'Failed to load strategies')
    items.value = Array.isArray(dStrategies?.items) ? dStrategies.items : []
    usageById.value = typeof dUsage?.by_id === 'object' && dUsage?.by_id ? dUsage.by_id : {}
    globalPosterior.value = typeof dState?.global === 'object' && dState?.global ? dState.global : {}
  } catch (e) {
    showToast({ title: 'Failed to load', message: e instanceof Error ? e.message : 'Request failed' })
  } finally {
    loading.value = false
  }
}

async function refreshGlobalStrategyStore() {
  resetStrategies()
  await fetchStrategies(API_BASE)
}

async function createOne() {
  if (!canCreate.value) return
  try {
    const res = await fetch(`${API_BASE}/api/strategies/`, {
      method: 'POST',
      headers: authHeaders(true),
      body: JSON.stringify({
        id: newId.value.trim(),
        label: newLabel.value.trim(),
        instruction: newInstruction.value.trim(),
        enabled: newEnabled.value,
        event_name: newEventName.value,
      }),
    })
    const d = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(d?.detail || 'Create failed')
    newId.value = ''
    newLabel.value = ''
    newInstruction.value = ''
    newEnabled.value = true
    newEventName.value = 'Decision'
    showCreate.value = false
    showToast({ title: 'Strategy created', message: 'Added to strategy roster.' })
    await refreshGlobalStrategyStore()
    await loadAll()
  } catch (e) {
    showToast({ title: 'Create failed', message: e instanceof Error ? e.message : 'Request failed' })
  }
}

function goToDetail(id: string) {
  router.push(`/app/strategies/${encodeURIComponent(id)}`)
}

onMounted(async () => {
  await loadMe()
  if (isAdmin.value) {
    await loadAll()
    window.addEventListener('focus', loadAll)
  } else {
    items.value = []
  }
})
</script>

<template>
  <div class="max-w-[96rem] mx-auto space-y-4">
    <div v-if="isAdmin === false" class="p-4 rounded-xl border glass-panel text-sm text-muted-foreground">
      Strategies are admin-only. Ask an admin to add or edit strategies.
    </div>

    <div v-else class="strategies-pro rounded-2xl border border-border/60 p-4 lg:p-6 space-y-5 text-foreground">
      <div class="flex items-center justify-between gap-3 flex-wrap">
        <div class="space-y-1">
          <h1 class="text-2xl lg:text-3xl font-semibold tracking-tight leading-tight">Strategies</h1>
          <p class="text-sm text-muted-foreground">Operational view for adaptive strategy quality and prompt control.</p>
        </div>
        <div class="flex items-center gap-2">
          <Button type="button" variant="outline" class="h-9 px-4 rounded-full" @click="showCreate = !showCreate">
            {{ showCreate ? 'Hide Create' : 'New Strategy' }}
          </Button>
          <Button type="button" class="h-9 px-4 rounded-full" @click="loadAll" :disabled="loading">
            {{ loading ? 'Refreshing…' : 'Refresh' }}
          </Button>
        </div>
      </div>

      <div class="grid grid-cols-2 xl:grid-cols-5 gap-3">
        <Card class="p-4 kpi-card"><div class="text-[11px] font-medium text-muted-foreground uppercase tracking-wide">Total strategies</div><div class="mt-1 text-3xl font-semibold text-amber-600 dark:text-amber-300 leading-none">{{ items.length }}</div></Card>
        <Card class="p-4 kpi-card"><div class="text-[11px] font-medium text-muted-foreground uppercase tracking-wide">Healthy</div><div class="mt-1 text-3xl font-semibold text-emerald-600 dark:text-emerald-300 leading-none">{{ healthyCount }}</div></Card>
        <Card class="p-4 kpi-card"><div class="text-[11px] font-medium text-muted-foreground uppercase tracking-wide">Needs review</div><div class="mt-1 text-3xl font-semibold text-amber-600 dark:text-amber-300 leading-none">{{ reviewCount }}</div></Card>
        <Card class="p-4 kpi-card"><div class="text-[11px] font-medium text-muted-foreground uppercase tracking-wide">Failing</div><div class="mt-1 text-3xl font-semibold text-rose-600 dark:text-rose-300 leading-none">{{ failingCount }}</div></Card>
        <Card class="p-4 kpi-card"><div class="text-[11px] font-medium text-muted-foreground uppercase tracking-wide">Avg success rate</div><div class="mt-1 text-3xl font-semibold text-amber-600 dark:text-amber-200 leading-none">{{ Math.round(avgSuccessRate) }}%</div></Card>
      </div>

      <Card v-if="showCreate" class="p-4 lg:p-5 space-y-4 panel-card">
        <div class="text-sm font-medium">Create new strategy</div>
        <div class="grid gap-3 md:grid-cols-3">
          <label class="block"><div class="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1.5">Id</div><Input v-model="newId" placeholder="e.g. my_custom_strategy" /></label>
          <label class="block"><div class="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1.5">Label</div><Input v-model="newLabel" placeholder="e.g. My Custom Strategy" /></label>
          <label class="block">
            <div class="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1.5">Event name</div>
            <Input v-model="newEventName" list="event-name-suggestions" placeholder="e.g. Escalation, Cancellation, Billing" />
            <datalist id="event-name-suggestions">
              <option v-for="ev in EVENT_TYPE_SUGGESTIONS" :key="ev" :value="ev" />
            </datalist>
          </label>
        </div>
        <label class="block">
          <div class="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1.5">Instruction</div>
          <textarea v-model="newInstruction" class="w-full min-h-[110px] rounded-lg border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2" />
        </label>
        <div class="flex items-center justify-between">
          <label class="flex items-center gap-2 cursor-pointer select-none text-sm text-muted-foreground"><input v-model="newEnabled" type="checkbox" class="rounded border-input" />Enabled</label>
          <Button type="button" class="rounded-full px-4" :disabled="!canCreate" @click="createOne">Create</Button>
        </div>
      </Card>

      <Card class="p-4 lg:p-5 panel-card">
        <div class="flex items-center gap-2.5 flex-wrap mb-3.5">
          <Input v-model="query" placeholder="Search strategy or prompt text…" class="max-w-sm min-w-[260px]" />
          <select v-model="sortBy" class="h-9 rounded-full border border-input bg-background px-3 text-sm text-foreground">
            <option value="status">Sort: Status (failing first)</option>
            <option value="success">Sort: Success rate</option>
            <option value="label">Sort: Label A-Z</option>
            <option value="last_used">Sort: Last used</option>
          </select>
          <div class="ml-auto text-xs text-muted-foreground">{{ tableRows.length }} shown · {{ items.length }} total · {{ selectedCount }} selected</div>
        </div>

        <div class="space-y-2.5 mb-3.5">
          <div class="flex items-center gap-2 flex-wrap">
            <span class="text-xs text-muted-foreground mr-1">Event:</span>
            <button class="chip" :class="{ active: eventFilter === 'All' }" @click="eventFilter = 'All'">All</button>
            <button v-for="ev in eventFilterOptions" :key="ev" class="chip" :class="{ active: eventFilter === ev }" @click="eventFilter = ev">{{ ev }}</button>
          </div>
          <div class="flex items-center gap-2 flex-wrap">
            <span class="text-xs text-muted-foreground mr-1">Status:</span>
            <button class="chip" :class="{ active: statusFilter === 'All' }" @click="statusFilter = 'All'">All</button>
            <button class="chip" :class="{ active: statusFilter === 'Failing' }" @click="statusFilter = 'Failing'">Failing</button>
            <button class="chip" :class="{ active: statusFilter === 'Review' }" @click="statusFilter = 'Review'">Review</button>
            <button class="chip" :class="{ active: statusFilter === 'Healthy' }" @click="statusFilter = 'Healthy'">Healthy</button>
          </div>
          <div class="text-[11px] text-muted-foreground">Event type is a stable UX category derived from strategy id for visual grouping.</div>
        </div>

        <div class="overflow-x-auto rounded-xl border border-border/70">
          <table class="w-full min-w-[1120px] text-sm">
            <thead class="bg-muted/65 text-muted-foreground sticky top-0 z-[1] backdrop-blur supports-[backdrop-filter]:bg-muted/55">
              <tr>
                <th class="px-3 py-2 text-left w-8"></th>
                <th class="px-3 py-2 text-left">Strategy</th>
                <th class="px-3 py-2 text-left">Event type</th>
                <th class="px-3 py-2 text-left">A / B</th>
                <th class="px-3 py-2 text-left">M (success rate)</th>
                <th class="px-3 py-2 text-left">Confidence</th>
                <th class="px-3 py-2 text-left">Trend</th>
                <th class="px-3 py-2 text-left">Status</th>
                <th class="px-3 py-2 text-left">Last used</th>
                <th class="px-3 py-2 text-left">View</th>
              </tr>
            </thead>
            <tbody>
              <tr v-if="tableRows.length === 0">
                <td colspan="10" class="px-3 py-8 text-center text-muted-foreground">No strategies match the selected filters.</td>
              </tr>
              <tr v-for="row in tableRows" :key="row.item.id" class="table-row border-t border-border/70">
                <td class="px-3 py-2.5 align-top"><input v-model="selectedMap[row.item.id]" type="checkbox" class="rounded border-input bg-background" /></td>
                <td class="px-3 py-2.5 align-top">
                  <div class="font-medium text-foreground">{{ row.item.label }}</div>
                  <div class="font-mono text-[11px] text-muted-foreground">{{ row.item.id }}</div>
                </td>
                <td class="px-3 py-2.5 align-top"><span class="inline-flex rounded-full bg-muted border border-border px-2.5 py-0.5 text-xs">{{ row.eventType }}</span></td>
                <td class="px-3 py-2.5 align-top"><span class="text-emerald-600 dark:text-emerald-300">{{ row.usage.wins }}</span> / <span class="text-rose-600 dark:text-rose-300">{{ row.usage.trials }}</span></td>
                <td class="px-3 py-2.5 align-top">
                  <div
                    class="flex items-center gap-2 cursor-help"
                    :title="successRateDetails(row)"
                  >
                    <div class="h-2.5 w-20 rounded-full bg-muted"><div class="h-full rounded-full bg-gradient-to-r from-amber-400 to-emerald-400 dark:from-amber-300 dark:to-emerald-300" :style="{ width: `${Math.max(4, Math.min(100, row.successPct))}%` }" /></div>
                    <span class="tabular-nums">{{ Math.round(row.successPct) }}%</span>
                    <span v-if="row.usage.trials === 0" class="text-[11px] text-muted-foreground">(est.)</span>
                  </div>
                </td>
                <td class="px-3 py-2.5 align-top">
                  <span
                    class="inline-flex rounded-md px-2 py-0.5 text-xs border cursor-help"
                    :title="statusDetails(row)"
                    :class="
                      row.confidence === 'High'
                        ? 'border-emerald-500/50 text-emerald-300'
                        : row.confidence === 'Medium'
                          ? 'border-cyan-500/50 text-cyan-300'
                          : row.confidence === 'Need review'
                            ? 'border-amber-500/50 text-amber-300'
                            : 'border-rose-500/50 text-rose-300'
                    "
                  >
                    {{ row.confidence }}
                  </span>
                </td>
                <td class="px-3 py-2.5 align-top">
                  <svg viewBox="0 0 110 28" class="h-7 w-[110px]">
                    <path :d="sparklinePath(row.sparkline)" fill="none" stroke="#facc15" stroke-width="2" stroke-linecap="round" />
                  </svg>
                </td>
                <td class="px-3 py-2.5 align-top">
                  <span class="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs border" :class="row.status === 'Healthy' ? 'border-emerald-500/50 text-emerald-300' : row.status === 'Review' ? 'border-amber-500/50 text-amber-300' : 'border-rose-500/50 text-rose-300'">
                    <span class="h-1.5 w-1.5 rounded-full bg-current" />{{ row.status }}
                  </span>
                </td>
                <td class="px-3 py-2.5 align-top text-muted-foreground">{{ formatLastUsed(row.usage.last_ts) }}</td>
                <td class="px-3 py-2.5 align-top">
                  <button
                    class="inline-flex items-center rounded-full border border-amber-400/40 bg-amber-500/10 px-2.5 py-1 text-xs font-medium text-amber-700 hover:bg-amber-500/15 dark:text-amber-200 dark:hover:bg-amber-500/25"
                    @click="goToDetail(row.item.id)"
                  >
                    View
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>

      <div class="text-xs text-muted-foreground px-1">
        Click <span class="font-semibold text-amber-600 dark:text-amber-200">View</span> to open the full strategy detail screen with analytics and editing.
      </div>
    </div>
  </div>
</template>

<style scoped>
.strategies-pro {
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.82), rgba(247, 250, 255, 0.9));
}

:global(.dark) .strategies-pro {
  background: linear-gradient(180deg, rgba(9, 14, 29, 0.96), rgba(11, 19, 35, 0.96));
}

.kpi-card,
.panel-card {
  background: color-mix(in oklab, var(--card) 92%, white 8%);
}

.kpi-card {
  transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
}

.kpi-card:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 20px -14px rgba(2, 132, 199, 0.45);
  border-color: color-mix(in oklab, var(--border) 70%, rgb(34 211 238) 30%);
}

:global(.dark) .kpi-card,
:global(.dark) .panel-card {
  background: rgba(15, 23, 42, 0.52);
}

.panel-card {
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.table-row {
  transition: background-color 150ms ease;
}

.table-row:hover {
  background: color-mix(in oklab, var(--muted) 52%, transparent);
}

.chip {
  height: 2rem;
  border-radius: 999px;
  border: 1px solid color-mix(in oklab, var(--border) 82%, transparent);
  color: color-mix(in oklab, var(--foreground) 84%, transparent);
  background: color-mix(in oklab, var(--card) 92%, transparent);
  font-size: 12px;
  padding: 0 0.8rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-weight: 500;
  transition: all 160ms ease;
}

.chip.active {
  border-color: rgba(251, 191, 36, 0.72);
  background: rgba(245, 158, 11, 0.2);
  color: color-mix(in oklab, var(--foreground) 92%, #b45309 8%);
}

:global(.dark) .chip {
  border: 1px solid rgba(148, 163, 184, 0.5);
  color: rgba(226, 232, 240, 0.94);
  background: rgba(15, 23, 42, 0.62);
}

:global(.dark) .chip.active {
  background: rgba(245, 158, 11, 0.26);
  color: rgba(254, 243, 199, 0.98);
}
</style>

