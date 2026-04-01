import { computed, reactive } from 'vue'
import { getAccessToken } from '@/lib/auth'

export type StrategyItem = {
  id: string
  label: string
  instruction: string
  enabled: boolean
  event_name?: string
}

const STATE = reactive({
  items: [] as StrategyItem[],
  loaded: false,
  loading: false,
  lastError: null as string | null,
})

function authHeaders(): HeadersInit {
  const token = getAccessToken()
  const h: Record<string, string> = {}
  if (token) h.Authorization = `Bearer ${token}`
  return h
}

export const enabledStrategies = computed(() => STATE.items.filter((i) => i.enabled))
export const enabledStrategyIds = computed(() => enabledStrategies.value.map((i) => i.id))

export function getStrategyLabel(id: string): string {
  return STATE.items.find((i) => i.id === id)?.label ?? id
}

export function getStrategyInstruction(id: string): string {
  return STATE.items.find((i) => i.id === id)?.instruction ?? ''
}

export function instructionPreview(instruction: string, maxChars = 110) {
  const s = String(instruction ?? '').trim()
  if (s.length <= maxChars) return s
  return s.slice(0, maxChars).trimEnd() + '…'
}

export async function fetchStrategies(apiBase: string): Promise<void> {
  if (STATE.loaded || STATE.loading) return
  STATE.loading = true
  STATE.lastError = null
  try {
    const r = await fetch(`${apiBase}/api/strategies`, { headers: authHeaders() })
    if (!r.ok) {
      const d = await r.json().catch(() => ({}))
      throw new Error(d?.detail || `Failed to fetch strategies (${r.status})`)
    }
    const d = await r.json()
    STATE.items = Array.isArray(d?.items) ? (d.items as StrategyItem[]) : []
    STATE.loaded = true
  } catch (e) {
    STATE.lastError = e instanceof Error ? e.message : 'Request failed'
    STATE.items = []
    STATE.loaded = false
  } finally {
    STATE.loading = false
  }
}

export function resetStrategies() {
  STATE.items = []
  STATE.loaded = false
  STATE.loading = false
  STATE.lastError = null
}

