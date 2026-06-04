<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { Motion } from '@motionone/vue'
import Card from '@/components/ui/Card.vue'
import Input from '@/components/ui/Input.vue'
import Button from '@/components/ui/Button.vue'
import PreferenceModal from '@/components/PreferenceModal.vue'
import TechPanels from '@/components/TechPanels.vue'
import WidgetSchemaRenderer from '@/components/WidgetRegistryRenderer.vue'
import { getAccessToken, clearAccessToken } from '@/lib/auth'
import { ingestDone, ingestReward } from '@/lib/analyticsStore'
import {
  applyPosteriorPack,
  banditState,
  fetchBanditStateFromApi,
  resetBanditUi,
} from '@/lib/banditState'
import { showToast } from '@/lib/toast'
import { type PosteriorMap } from '@/lib/strategies'
import { getStrategyLabel } from '@/lib/strategiesStore'
import { renderAssistantMarkdown } from '@/lib/renderMarkdown'
import { MOTION_BASE, animatePulse, killAnimationsOf } from '@/lib/motion'
import { downloadTextAsFile, prettifyJsonIfPossible } from '@/lib/downloadFile'
import { ArrowDownTrayIcon } from '@/components/icons'

const router = useRouter()
const API_BASE = import.meta.env.VITE_API_BASE_URL || ((typeof location !== 'undefined' && /^(localhost|127.0.0.1)$/.test(location.hostname)) ? 'http://localhost:5051' : '')

/** FastAPI uses `detail`; our API uses `error` — normalize for user-visible messages. */
function formatApiErrorBody(d: Record<string, unknown>, fallback: string): string {
  const err = d.error
  if (typeof err === 'string' && err.trim()) return err.trim()
  const det = d.detail
  if (typeof det === 'string' && det.trim()) return det.trim()
  if (Array.isArray(det)) {
    const parts = det
      .map((item) => {
        if (item && typeof item === 'object' && 'msg' in item) {
          return String((item as { msg?: string }).msg || '').trim()
        }
        return ''
      })
      .filter(Boolean)
    if (parts.length) return parts.join('; ')
  }
  return fallback
}

async function readJsonWithFallback(res: Response): Promise<Record<string, unknown>> {
  const text = await res.text().catch(() => '')
  if (!text.trim()) return {}
  try {
    return JSON.parse(text) as Record<string, unknown>
  } catch {
    return { error: text.slice(0, 400) }
  }
}

/** Matches backend `_json_layout_is_only_numeric_index_arrays` (bogus tic-tac-toe "lines" as text). */
const NUMERIC_TUPLE_TEXT_RE = /^\s*\[\s*\d+(\s*,\s*\d+)*\s*\]\s*$/

function schemaIsOnlyNumericTupleText(wSch: string): boolean {
  const s = wSch.trim()
  if (!s.startsWith('{')) return false
  try {
    const o = JSON.parse(s) as { layout?: unknown }
    const layout = o.layout
    if (!Array.isArray(layout) || layout.length < 2) return false
    for (const item of layout) {
      // Bare numeric array layout items (raw stream form, e.g. [0,1,2]) count as tuple junk.
      if (Array.isArray(item)) {
        if (!item.every((n) => typeof n === 'number')) return false
        continue
      }
      if (!item || typeof item !== 'object') return false
      const rec = item as Record<string, unknown>
      if (String(rec.type || '').toLowerCase() !== 'text') return false
      if (!NUMERIC_TUPLE_TEXT_RE.test(String(rec.content ?? ''))) return false
    }
    return true
  } catch {
    return false
  }
}

/** Drop useless index-array "widgets" (e.g. bare [0,1,2] tuples) that aren't real content. */
function recoverWidgetFromStreamIfDegenerate(cur: ChatMessage) {
  const sch = String(cur.widgetSchema || '').trim()
  if (schemaIsOnlyNumericTupleText(sch)) cur.widgetSchema = ''
}

const isStreaming = ref(false)
const streamStatus = ref('')
const healthOk = ref<boolean | null>(null)

type PlainMsg = { role: 'user' | 'assistant'; content: string; elapsed?: number; error?: string }

type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
  strategy?: string
  xVec?: number[]
  widgetSchema?: string
  widgetHeight?: number
  widgetStream?: string
  widgetStreaming?: boolean
  widgetMode?: 'json' | ''
  rewardUsedUp?: boolean
  rewardUsedDown?: boolean
}

/** Successful `/api/chat` JSON (after error checks). */
type ChatApiSuccess = {
  response?: string
  strategy?: string
  x_vec?: unknown[]
  widget_schema?: string
  widget_height?: number
  instruction?: string
  scores?: Record<string, number> | null
  auto_detected?: boolean
  auto_r?: number
  auto_reason?: string
  elapsed?: number
  posterior?: PosteriorMap
  global?: PosteriorMap
  userb?: PosteriorMap
  global_n?: number
}

type ChatPlainSuccess = {
  response?: string
  elapsed?: number
}

const plainMessages = ref<PlainMsg[]>([])
const messages = ref<ChatMessage[]>([])
const input = ref('')
const sending = ref(false)

const showBaseline = ref(true)
const showTechPanels = ref(true)
const prefOpen = ref(false)

function closeTechPanels() {
  showTechPanels.value = false
}

const baselineScrollEl = ref<HTMLDivElement | null>(null)
const adaptiveScrollEl = ref<HTMLDivElement | null>(null)
const baselineStickToBottom = ref(true)
const adaptiveStickToBottom = ref(true)
const streamPulseEl = ref<HTMLDivElement | null>(null)
const widgetGenerating = ref(false)
const widgetGeneratingIdx = ref<number | null>(null)
/** True while SSE is in widget phase (answer text may already be visible). */
const widgetStreamPhase = ref(false)

let healthTimer: ReturnType<typeof setInterval> | null = null

/** Grid columns must depend on baseline + insights, or a phantom 3rd track leaves empty space. */
const chatMainGridClass = computed(() => {
  const b = showBaseline.value
  const t = showTechPanels.value
  if (b && t) return 'lg:grid-cols-2 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_380px]'
  if (b && !t) return 'lg:grid-cols-2 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]'
  if (!b && t) return 'lg:grid-cols-1 xl:grid-cols-[minmax(0,1fr)_380px]'
  return 'lg:grid-cols-1 xl:grid-cols-1'
})

function onGlobalKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape' && showTechPanels.value) closeTechPanels()
}

function readBaselineStorage() {
  try {
    showBaseline.value = localStorage.getItem('hideBaseline') !== '1'
  } catch {
    showBaseline.value = true
  }
}

function persistBaseline() {
  try {
    localStorage.setItem('hideBaseline', showBaseline.value ? '0' : '1')
  } catch {
    /* ignore */
  }
}

function toggleBaseline() {
  showBaseline.value = !showBaseline.value
  persistBaseline()
}

function authHeaders(): HeadersInit {
  const token = getAccessToken()
  const h: Record<string, string> = { 'Content-Type': 'application/json' }
  if (token) h.Authorization = `Bearer ${token}`
  return h
}

/**
 * If the server rejects our bearer token, the frontend must clear it and
 * bounce the user to /login — otherwise every request keeps failing silently
 * with a generic "request failed" banner.
 */
function handleAuthFailure(): boolean {
  clearAccessToken()
  showToast({
    title: 'Session expired',
    message: 'Please log in again to continue.',
  })
  router.push('/login')
  return true
}

async function fetchState() {
  const token = getAccessToken()
  if (!token) return
  const meta = await fetchBanditStateFromApi(API_BASE, authHeaders)
  if (meta?.msg_count === 0) prefOpen.value = true
}

async function fetchHealth() {
  try {
    const r = await fetch(`${API_BASE}/api/health`)
    healthOk.value = r.ok
  } catch {
    healthOk.value = false
  }
}

async function onPreferenceSubmit(payload: { strategies: string[]; lock: boolean }) {
  const token = getAccessToken()
  if (!token) return
  const res = await fetch(`${API_BASE}/api/preference`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ strategies: payload.strategies, lock: payload.lock }),
  })
  if (res.ok) {
    const d = await res.json()
    if (d.posterior) banditState.userPosterior = d.posterior
    showToast({ title: 'Preferences applied', message: 'Posterior warm-started from your choices.' })
  } else {
    showToast({ title: 'Preferences failed', message: 'Could not save preferences.' })
  }
}

function openPrefs() {
  prefOpen.value = true
}

async function onClearChat() {
  if (!confirm('Clear all messages and widgets? Your bandit learning and reward history stay.')) return
  const token = getAccessToken()
  if (!token) {
    router.push('/login')
    return
  }
  const res = await fetch(`${API_BASE}/api/conversation/clear`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({}),
  })
  if (!res.ok) {
    showToast({ title: 'Clear failed', message: 'Try again.' })
    return
  }
  plainMessages.value = []
  messages.value = [{ role: 'assistant', content: 'Chat cleared. Ask anything to begin again.' }]
  prefOpen.value = false
  showToast({ title: 'Chat cleared', message: 'Messages and widgets removed; bandit state kept.' })
}

async function onReset() {
  if (!confirm('Reset your bandit session? History and learning for this account will be cleared.')) return
  const token = getAccessToken()
  if (!token) {
    router.push('/login')
    return
  }
  const res = await fetch(`${API_BASE}/api/reset`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({}),
  })
  if (!res.ok) {
    showToast({ title: 'Reset failed', message: 'Try again.' })
    return
  }
  plainMessages.value = []
  messages.value = [{ role: 'assistant', content: 'Session reset. Ask anything to begin again.' }]
  resetBanditUi()
  await fetchState()
  prefOpen.value = true
  showToast({ title: 'Session reset', message: 'Your adaptive state was cleared.' })
}

async function submitReward(m: ChatMessage, reward: 0 | 1) {
  if (m.role !== 'assistant') return
  if (!m.strategy || !Array.isArray(m.xVec) || m.xVec.length === 0) return
  if (reward === 1 && m.rewardUsedUp) return
  if (reward === 0 && m.rewardUsedDown) return

  // Capture posterior prediction BEFORE the rate endpoint updates posteriors.
  const predicted = m.strategy ? banditState.userPosterior[m.strategy] : undefined
  const predictedR = predicted?.r ?? null
  const predictedU = predicted?.u ?? null

  const token = getAccessToken()
  if (!token) {
    router.push('/login')
    return
  }

  const res = await fetch(`${API_BASE}/api/rate`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ strategy: m.strategy, x_vec: m.xVec, reward }),
  })

  if (res.ok) {
    const d = await res.json()
    applyPosteriorPack(d)
    if (reward === 1) m.rewardUsedUp = true
    if (reward === 0) m.rewardUsedDown = true
    ingestReward({ strategy: m.strategy, reward, predictedR, predictedU })
    banditState.rewardLog.unshift({
      strategy: m.strategy,
      reward,
      detail: 'explicit feedback',
      source: 'manual',
    })
    showToast({ title: 'Feedback saved', message: reward === 1 ? 'Marked as helpful.' : 'Marked as not helpful.' })
  } else {
    showToast({ title: 'Feedback failed', message: 'Could not save feedback. Try again.' })
  }
}

async function runChatPlain(text: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat_plain`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ message: text }),
  })
  if (res.status === 401) {
    handleAuthFailure()
    return
  }
  const d = await readJsonWithFallback(res)
  if (!res.ok || d.error || d.detail) {
    plainMessages.value.push({
      role: 'assistant',
      content: '',
      error: formatApiErrorBody(d, 'Baseline request failed'),
    })
    return
  }
  const ok = d as ChatPlainSuccess
  plainMessages.value.push({
    role: 'assistant',
    content: ok.response || '',
    elapsed: typeof ok.elapsed === 'number' ? ok.elapsed : undefined,
  })
  await nextTick()
  if (baselineScrollEl.value && baselineStickToBottom.value) scrollToBottom(baselineScrollEl.value)
}

async function runChatFallback(text: string, idx: number): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ message: text }),
  })
  if (res.status === 401) {
    handleAuthFailure()
    messages.value[idx].content = '⚠ Session expired — redirecting to login.'
    return false
  }
  const d = await readJsonWithFallback(res)
  if (!res.ok || d.error || d.detail) {
    messages.value[idx].content = `⚠ ${formatApiErrorBody(d, `Adaptive request failed (${res.status})`)}`
    return false
  }
  const payload = d as ChatApiSuccess
  const cur = messages.value[idx]
  cur.content = cleanAssistantText(
    typeof payload.response === 'string' ? payload.response : String(payload.response ?? ''),
  )
  cur.strategy = typeof payload.strategy === 'string' ? payload.strategy : String(payload.strategy ?? '')
  cur.xVec = Array.isArray(payload.x_vec) ? payload.x_vec.map((n: unknown) => Number(n)) : []
  cur.widgetSchema = typeof payload.widget_schema === 'string' ? payload.widget_schema : ''
  if (schemaIsOnlyNumericTupleText(cur.widgetSchema)) cur.widgetSchema = ''
  cur.widgetHeight = payload.widget_height ? Number(payload.widget_height) : 420
  cur.rewardUsedUp = false
  cur.rewardUsedDown = false
  banditState.activeStrategy = payload.strategy || ''
  banditState.activeInstruction = typeof payload.instruction === 'string' ? payload.instruction : ''
  banditState.selectedStrategy = payload.strategy || ''
  banditState.scores = payload.scores ?? null
  banditState.lastXVec = cur.xVec ?? null
  applyPosteriorPack(payload)
  if (payload.auto_detected && payload.auto_r != null) {
    const autoReward = Number(payload.auto_r)
    const autoStrategy = String(payload.strategy ?? cur.strategy ?? 'unknown')
    // Prevent double-counting if the user later submits explicit feedback.
    cur.rewardUsedUp = autoReward === 1
    cur.rewardUsedDown = autoReward === 0

    banditState.rewardLog.unshift({
      strategy: String(payload.strategy ?? autoStrategy),
      reward: autoReward,
      detail: String(payload.auto_reason || ''),
      source: 'auto',
    })

    const predicted = banditState.userPosterior[autoStrategy]
    ingestReward({
      strategy: autoStrategy,
      reward: autoReward,
      predictedR: predicted?.r ?? null,
      predictedU: predicted?.u ?? null,
    })
  }
  ingestDone({
    strategy: payload.strategy ?? 'unknown',
    elapsed: payload.elapsed,
    widget_schema: payload.widget_schema ?? '',
  })
  return true
}

function handleSseEvent(evt: Record<string, unknown>, idx: number) {
  const cur = messages.value[idx]
  if (!evt?.type) return

  if (evt.type === 'strategy') {
    widgetStreamPhase.value = false
    streamStatus.value = 'Writing…'
    banditState.activeStrategy = String(evt.strategy ?? '')
    banditState.activeInstruction = String(evt.instruction ?? '')
    banditState.selectedStrategy = String(evt.strategy ?? '')
    banditState.scores = (evt.scores as Record<string, number>) || null
    banditState.lastXVec = Array.isArray(evt.x_vec) ? (evt.x_vec as number[]).map((n) => Number(n)) : null
    if (cur.role === 'assistant') {
      cur.strategy = String(evt.strategy ?? '')
      cur.xVec = Array.isArray(evt.x_vec) ? (evt.x_vec as number[]).map((n) => Number(n)) : []
    }
    applyPosteriorPack({
      posterior: evt.posterior as PosteriorMap,
      global: evt.global as PosteriorMap,
      userb: evt.userb as PosteriorMap,
      global_n: evt.global_n as number,
    })
  }

  if (evt.type === 'response_delta') {
    widgetStreamPhase.value = false
    streamStatus.value = 'Streaming answer…'
    cur.content += String((evt as { delta?: string }).delta ?? '')
    requestAdaptiveAutoScroll()
  }

  if (evt.type === 'widget_start') {
    widgetStreamPhase.value = true
    streamStatus.value = 'Building interactive widget…'
    widgetGenerating.value = true
    widgetGeneratingIdx.value = idx
    cur.widgetStream = ''
    cur.widgetStreaming = true
    cur.widgetMode = ''
    requestAdaptiveAutoScroll()
  }

  if (evt.type === 'widget_delta') {
    widgetStreamPhase.value = true
    streamStatus.value = 'Building interactive widget…'
    widgetGenerating.value = true
    widgetGeneratingIdx.value = idx
    const delta = String((evt as { delta?: string }).delta ?? '')
    if (delta) {
      cur.widgetStream = (cur.widgetStream || '') + delta
      cur.widgetStreaming = true
      if (!cur.widgetMode) {
        const preview = cur.widgetStream.slice(0, 400).trim().toLowerCase()
        if (preview.startsWith('{') || preview.startsWith('[') || preview.startsWith('```json')) cur.widgetMode = 'json'
      }
      requestAdaptiveAutoScroll()
    }
  }

  if (evt.type === 'done') {
    streamStatus.value = ''
    widgetStreamPhase.value = false
    widgetGenerating.value = false
    widgetGeneratingIdx.value = null
    const e = evt as {
      response?: string
      strategy?: string
      widget_schema?: string
      widget_height?: number
      x_vec?: number[]
      error?: string
      elapsed?: number
      auto_detected?: boolean
      auto_r?: number
      auto_reason?: string
    }
    if (e.error) {
      cur.content = `⚠ ${e.error}`
      return
    }
    cur.content = cleanAssistantText(e.response ?? cur.content)
    cur.strategy = e.strategy ?? cur.strategy
    cur.widgetSchema = typeof e.widget_schema === 'string' ? e.widget_schema.trim() : ''
    cur.widgetHeight = e.widget_height ? Number(e.widget_height) : 420
    cur.widgetStreaming = false

    recoverWidgetFromStreamIfDegenerate(cur)

    // If the server cleared the finalized schema but we already streamed JSON into
    // `widgetStream`, promote that stream so the panel still mounts (avoids the
    // "widget disappeared after completion" symptom).
    const stream = String(cur.widgetStream || '').trim()
    if (!cur.widgetSchema && stream && !schemaIsOnlyNumericTupleText(stream)) {
      cur.widgetSchema = stream
    }
    cur.widgetMode = cur.widgetSchema?.trim() ? 'json' : ''
    cur.xVec = Array.isArray(e.x_vec) ? e.x_vec.map((n) => Number(n)) : []
    cur.rewardUsedUp = false
    cur.rewardUsedDown = false

    applyPosteriorPack({
      posterior: evt.posterior as PosteriorMap,
      global: evt.global as PosteriorMap,
      userb: evt.userb as PosteriorMap,
      global_n: evt.global_n as number,
    })

    if (e.auto_detected && e.auto_r != null) {
      const autoReward = Number(e.auto_r)
      const autoStrategy = String(e.strategy ?? cur.strategy ?? 'unknown')
      // Prevent double-counting if the user later submits explicit feedback.
      cur.rewardUsedUp = autoReward === 1
      cur.rewardUsedDown = autoReward === 0

      banditState.rewardLog.unshift({
        strategy: autoStrategy,
        reward: autoReward,
        detail: String(e.auto_reason ?? ''),
        source: 'auto',
      })

      const predicted = banditState.userPosterior[autoStrategy]
      ingestReward({
        strategy: autoStrategy,
        reward: autoReward,
        predictedR: predicted?.r ?? null,
        predictedU: predicted?.u ?? null,
      })
    }

    ingestDone({
      strategy: e.strategy ?? cur.strategy ?? 'unknown',
      elapsed: e.elapsed,
      widget_schema: e.widget_schema ?? '',
    })
  }
}

/** An action_row button was clicked — send its label as the next prompt. */
function onWidgetAction(text: string) {
  const t = String(text || '').trim()
  if (!t || sending.value || isStreaming.value) return
  input.value = t
  void onSend()
}

async function onSend() {
  const token = getAccessToken()
  if (!token) {
    router.push('/login')
    return
  }

  const text = input.value.trim()
  if (!text || sending.value) return

  sending.value = true
  isStreaming.value = true
  widgetStreamPhase.value = false
  streamStatus.value = 'Thinking…'
  // Each new compare turn should anchor both panes to latest messages.
  baselineStickToBottom.value = true
  adaptiveStickToBottom.value = true

  plainMessages.value.push({ role: 'user', content: text })
  messages.value.push({ role: 'user', content: text })
  const idx = messages.value.length
  messages.value.push({ role: 'assistant', content: '' })

  await nextTick()
  if (baselineScrollEl.value && showBaseline.value) scrollToBottom(baselineScrollEl.value)
  if (adaptiveScrollEl.value) scrollToBottom(adaptiveScrollEl.value)

  input.value = ''

  const plainJob = showBaseline.value
    ? runChatPlain(text).catch(() => {
        plainMessages.value.push({ role: 'assistant', content: '', error: 'Baseline connection error' })
      })
    : Promise.resolve()

  let finalized = false
  let fallbackUsed = false

  try {
    const resp = await fetch(`${API_BASE}/api/chat_stream`, {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({ message: text }),
    })

    if (resp.status === 401) {
      handleAuthFailure()
      messages.value[idx].content = '⚠ Session expired — redirecting to login.'
      finalized = true
      return
    }

    if (!resp.ok) {
      if (resp.status === 401) {
        handleAuthFailure()
        messages.value[idx].content = '⚠ Session expired — redirecting to login.'
        finalized = true
        return
      }
      const d = await readJsonWithFallback(resp)
      throw new Error(formatApiErrorBody(d, `Stream request failed (${resp.status})`))
    }

    if (!resp.body) throw new Error('Streaming response missing body')

    const reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''

    while (true) {
      const { value, done } = await reader.read()
      if (done) break

      buf += decoder.decode(value, { stream: true })
      const blocks = buf.split('\n\n')
      buf = blocks.pop() || ''

      for (const block of blocks) {
        const dataLines = block
          .split('\n')
          .filter((ln) => ln.startsWith('data:'))
          .map((ln) => ln.slice(5).trimStart())

        if (!dataLines.length) continue

        let evt: Record<string, unknown> | null = null
        try {
          evt = JSON.parse(dataLines.join('\n'))
        } catch {
          continue
        }
        if (!evt) continue

        handleSseEvent(evt, idx)

        if (evt.type === 'done') {
          finalized = true
          const cur = messages.value[idx]
          // Redundant safety: promote streamed JSON if the done payload left the schema empty.
          const wSch = typeof (evt as { widget_schema?: string }).widget_schema === 'string' ? (evt as { widget_schema: string }).widget_schema.trim() : ''
          const stream = String(cur.widgetStream || '').trim()
          if (!wSch && stream && !schemaIsOnlyNumericTupleText(stream)) {
            cur.widgetSchema = stream
          }
          cur.widgetMode = cur.widgetSchema?.trim() ? 'json' : ''
          recoverWidgetFromStreamIfDegenerate(cur)
        }
      }
    }
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e)
    if (!finalized) {
      try {
        const ok = await runChatFallback(text, idx)
        if (!ok) showToast({ title: 'Request failed', message: msg })
      } catch {
        messages.value[idx].content = `⚠ ${msg}`
        showToast({ title: 'Request failed', message: msg })
      }
      fallbackUsed = true
    }
  } finally {
    await plainJob
    sending.value = false
    isStreaming.value = false
    streamStatus.value = ''
    widgetStreamPhase.value = false
    if (!finalized && !fallbackUsed) {
      try {
        const ok = await runChatFallback(text, idx)
        if (!ok) messages.value[idx].content ||= '⚠ Stream ended without final output.'
      } catch {
        messages.value[idx].content ||= '⚠ Stream ended without final output.'
      }
    }
  }
}

type HistoryPair = {
  user: string
  assistant: string
  widget_schema?: string
  widget_height?: number
}

function assistantFromHistoryPair(p: HistoryPair): ChatMessage {
  const wSch = String(p.widget_schema ?? '').trim()
  const wh = Number(p.widget_height) || 420
  const base: ChatMessage = {
    role: 'assistant',
    content: cleanAssistantText(p.assistant),
    widgetHeight: wh,
  }
  if (wSch && !schemaIsOnlyNumericTupleText(wSch)) {
    base.widgetSchema = wSch
    base.widgetMode = 'json'
  }
  return base
}

async function loadConversationHistory() {
  const token = getAccessToken()
  if (!token) return

  try {
    const res = await fetch(`${API_BASE}/api/conversation?limit=20`, { headers: authHeaders() })
    if (!res.ok) return
    const d = await res.json().catch(() => ({}))

    const baselineHist = (d?.baseline?.history ?? []) as HistoryPair[]
    const adaptiveHist = (d?.adaptive?.history ?? []) as HistoryPair[]

    plainMessages.value = baselineHist.flatMap((p) => [
      { role: 'user' as const, content: p.user },
      { role: 'assistant' as const, content: cleanAssistantText(p.assistant) },
    ])

    messages.value = adaptiveHist.flatMap((p) => [
      { role: 'user' as const, content: p.user },
      assistantFromHistoryPair(p),
    ])
  } catch {
    // Non-blocking: chat still works even if history can't load.
  }

  if (plainMessages.value.length === 0 && showBaseline.value) {
    // Keep empty state for baseline panel.
  }
  if (messages.value.length === 0) {
    messages.value = [{ role: 'assistant', content: 'Ask anything; adaptive history will appear here.' }]
  }
}

onMounted(async () => {
  const token = getAccessToken()
  if (!token) {
    router.push('/login')
    return
  }

  readBaselineStorage()

  plainMessages.value = []
  messages.value = []
  prefOpen.value = false
  await loadConversationHistory()
  await nextTick()
  if (baselineScrollEl.value) scrollToBottom(baselineScrollEl.value)
  if (adaptiveScrollEl.value) scrollToBottom(adaptiveScrollEl.value)
  baselineStickToBottom.value = true
  adaptiveStickToBottom.value = true

  fetchState()
  fetchHealth()
  healthTimer = setInterval(fetchHealth, 30000)

  window.addEventListener('keydown', onGlobalKeydown)
  window.addEventListener('chat:control', onShellControl as EventListener)
})

onUnmounted(() => {
  if (healthTimer) clearInterval(healthTimer)
  killAnimationsOf(streamPulseEl.value)
  window.removeEventListener('keydown', onGlobalKeydown)
  window.removeEventListener('chat:control', onShellControl as EventListener)
})

function onShellControl(e: Event) {
  const action = String((e as CustomEvent).detail?.action ?? '')
  if (action === 'toggle-baseline') {
    toggleBaseline()
    return
  }
  if (action === 'toggle-insights') {
    showTechPanels.value = !showTechPanels.value
    return
  }
  if (action === 'open-preferences') {
    openPrefs()
    return
  }
  if (action === 'clear-chat') {
    void onClearChat()
    return
  }
  if (action === 'reset') {
    void onReset()
  }
}

watch(
  () => [plainMessages.value.length, messages.value.length],
  () => {
    nextTick(() => {
      if (baselineScrollEl.value && baselineStickToBottom.value) scrollToBottom(baselineScrollEl.value)
      if (adaptiveScrollEl.value && adaptiveStickToBottom.value) scrollToBottom(adaptiveScrollEl.value)
    })
  },
)

watch(isStreaming, (v) => {
  if (v) animatePulse(streamPulseEl.value)
  else killAnimationsOf(streamPulseEl.value)
})

/**
 * Strip any widget JSON the model leaked into the prose <RESPONSE>. The response is
 * meant to be human text only — JSON must NEVER reach the UI. Handles fenced blocks and
 * bare objects/arrays (balanced or truncated mid-stream).
 */
function stripLeakedJson(text: string): string {
  let s = text
  // 1) fenced code blocks whose content looks like widget JSON
  s = s.replace(/```[\w-]*\s*([\s\S]*?)```/g, (m, inner) => {
    const t = String(inner).trim()
    return /^[[{]/.test(t) && /"(?:type|layout|items|tone|series|chart|version)"/.test(t) ? '' : m
  })
  // 2) bare JSON widget/block blobs — balance-match (string-aware); truncated → cut to end
  const opener = /[{[]/g
  let match: RegExpExecArray | null
  while ((match = opener.exec(s)) !== null) {
    const start = match.index
    const head = s.slice(start, start + 400)
    if (!/"(?:type|layout|version)"\s*:/.test(head)) continue
    const open = s[start]
    const close = open === '{' ? '}' : ']'
    let depth = 0
    let inStr = false
    let esc = false
    let j = start
    for (; j < s.length; j++) {
      const ch = s[j]
      if (inStr) {
        if (esc) esc = false
        else if (ch === '\\') esc = true
        else if (ch === '"') inStr = false
      } else if (ch === '"') inStr = true
      else if (ch === open) depth++
      else if (ch === close) {
        depth--
        if (depth === 0) {
          j++
          break
        }
      }
    }
    const end = depth !== 0 ? s.length : j // unbalanced (truncated) → remove to end
    s = s.slice(0, start) + s.slice(end)
    opener.lastIndex = start
  }
  return s
}

function cleanAssistantText(raw: string | undefined | null) {
  const s = String(raw ?? '')
  const noWidgetBlock = s.replace(/<WIDGET>[\s\S]*?<\/WIDGET>/gi, '')
  const noTags = noWidgetBlock.replace(/<\/?WIDGET>/gi, '')
  return stripLeakedJson(noTags).trim()
}

/**
 * While the last assistant message is still receiving *answer* tokens, show plain text so half-written
 * markdown/tables do not flash. Once the server signals widget loading (`widgetGenerating`), the answer
 * text is complete — switch to markdown and show the compact widget loader below.
 */
function assistantStreamPlain(idx: number): boolean {
  const m = messages.value[idx]
  const last = idx === messages.value.length - 1
  const inWidgetWait =
    widgetGenerating.value && widgetGeneratingIdx.value === idx
  return Boolean(
    isStreaming.value && last && m?.role === 'assistant' && !inWidgetWait,
  )
}

function stratLabel(s?: string) {
  if (!s) return ''
  return getStrategyLabel(s) || s
}

function isNearBottom(el: HTMLElement, thresholdPx = 140): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight < thresholdPx
}

function scrollToBottom(el: HTMLElement) {
  el.scrollTop = el.scrollHeight
}

let scrollRaf: number | null = null
function requestAdaptiveAutoScroll() {
  if (scrollRaf != null) return
  scrollRaf = window.requestAnimationFrame(() => {
    scrollRaf = null
    if (!adaptiveScrollEl.value) return
    if (!adaptiveStickToBottom.value) return
    scrollToBottom(adaptiveScrollEl.value)
  })
}

function onBaselineScroll() {
  if (!baselineScrollEl.value) return
  baselineStickToBottom.value = isNearBottom(baselineScrollEl.value)
}

function onAdaptiveScroll() {
  if (!adaptiveScrollEl.value) return
  adaptiveStickToBottom.value = isNearBottom(adaptiveScrollEl.value)
}

function widgetDownloadBase(m: ChatMessage, idx: number): string {
  const part = (m.strategy || 'adaptive').replace(/[^a-zA-Z0-9_-]+/g, '-').replace(/^-|-$/g, '').slice(0, 32) || 'widget'
  return `widget-${part}-${idx + 1}`
}

function downloadWidgetJson(jsonStr: string, base: string) {
  const body = prettifyJsonIfPossible(String(jsonStr || '').trim())
  if (!body) {
    showToast({ title: 'Nothing to download', message: 'Widget JSON is empty.' })
    return
  }
  downloadTextAsFile(body, `${base}.json`, 'application/json;charset=utf-8')
}

function downloadLiveWidgetDraft(m: ChatMessage, idx: number) {
  const stream = String(m.widgetStream || '').trim()
  if (!stream) {
    showToast({ title: 'Nothing to download', message: 'Widget is still loading.' })
    return
  }
  downloadWidgetJson(stream, `${widgetDownloadBase(m, idx)}-draft`)
}
</script>

<template>
  <div class="h-full min-h-0 w-full min-w-0 overflow-hidden grid grid-rows-[auto_minmax(0,1fr)_auto] gap-2">
    <div class="flex flex-wrap items-center gap-2 justify-between min-h-0">
      <div class="flex items-center gap-3">
        <h1 class="text-lg font-semibold tracking-tight">Chat</h1>
        <span
          class="inline-flex items-center gap-2 text-[11px] text-muted-foreground rounded-full border bg-background/60 px-2.5 py-1"
          :title="healthOk === false ? 'API unreachable' : 'API health'"
        >
          <span
            class="h-2 w-2 rounded-full shrink-0"
            :class="{
              'bg-emerald-500': healthOk === true,
              'bg-red-500': healthOk === false,
              'bg-cyan-500 animate-pulse': healthOk === null,
            }"
          />
          {{ healthOk === false ? 'offline' : healthOk === true ? 'online' : 'checking…' }}
        </span>
      </div>
      <div />
    </div>

    <div class="grid gap-2 min-h-0 min-w-0 overflow-hidden items-stretch" :class="chatMainGridClass">
      <Card v-if="showBaseline" class="flex flex-col min-h-0 overflow-hidden shadow-sm">
        <div class="px-3 py-2 border-b text-[11px] font-semibold text-muted-foreground uppercase tracking-wide flex items-center justify-between">
          <span>Baseline</span>
          <span class="text-[10px] font-medium px-2 py-0.5 rounded-full border bg-background/60">No bandit</span>
        </div>
        <div
          ref="baselineScrollEl"
          class="chat-messages chat-pane-scroll flex-1 min-h-0 overflow-y-auto p-2.5 space-y-2 overscroll-contain scrollbar-gutter-stable"
          @scroll="onBaselineScroll"
        >
          <template v-if="plainMessages.length === 0">
            <div class="text-center text-sm text-muted-foreground py-14">
              <div class="text-2xl mb-2 opacity-40">◻</div>
              <div class="font-medium text-foreground">Baseline chat</div>
              <p class="text-xs mt-1 max-w-xs mx-auto">Same message, without strategy selection or adaptive formatting.</p>
            </div>
          </template>
          <template v-else>
            <Motion
              v-for="(m, i) in plainMessages"
              :key="'p-' + i"
              tag="div"
              class="space-y-1 premium-reveal"
              :initial="{ opacity: 0, y: 10 }"
              :animate="{ opacity: 1, y: 0 }"
              :transition="{ ...MOTION_BASE, delay: i * 0.03 }"
            >
              <div v-if="m.role === 'user'" class="flex justify-end">
                <div
                  class="max-w-[97%] rounded-2xl border bg-accent/15 px-3 py-2 text-sm shadow-sm transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md hover:shadow-cyan-500/15"
                >
                  <div class="text-[10px] text-muted-foreground mb-0.5">You</div>
                  <div class="whitespace-pre-wrap">{{ m.content }}</div>
                </div>
              </div>
              <div v-else class="flex justify-start">
                <div
                  class="max-w-[97%] rounded-2xl border bg-card px-3 py-2 text-sm shadow-sm transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md hover:shadow-cyan-500/15"
                >
                  <div class="text-[10px] text-muted-foreground mb-0.5">Baseline</div>
                  <div v-if="m.error" class="text-red-400 text-sm">⚠ {{ m.error }}</div>
                  <div v-else class="whitespace-pre-wrap leading-relaxed min-w-0">{{ m.content }}</div>
                  <div v-if="m.elapsed != null" class="text-[10px] text-muted-foreground mt-1">{{ m.elapsed }}s</div>
                </div>
              </div>
            </Motion>
          </template>
        </div>
      </Card>

      <Card class="flex flex-col min-h-0 min-w-0 overflow-hidden shadow-sm">
        <div
          class="px-3 py-2 border-b text-[11px] font-semibold text-muted-foreground uppercase tracking-wide flex items-center justify-between gap-2 flex-wrap"
        >
          <div class="flex items-center gap-2 min-w-0">
            <span>Adaptive</span>
            <span class="text-[10px] font-medium px-2 py-0.5 rounded-full border bg-background/60 shrink-0">Bandit layer</span>
          </div>
          <div
            v-if="isStreaming"
            class="flex items-center gap-2 normal-case font-normal text-[11px] rounded-full px-2.5 py-1.5 border max-w-[min(100%,20rem)] transition-colors"
            :class="
              widgetStreamPhase
                ? 'border-cyan-500/45 bg-cyan-500/12 text-cyan-900 dark:text-cyan-100 shadow-sm shadow-cyan-500/10'
                : 'border-border/80 bg-muted/50 text-foreground/80'
            "
          >
            <span class="relative flex h-2 w-2 shrink-0">
              <span
                v-if="widgetStreamPhase"
                class="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-70"
              />
              <span
                class="relative inline-flex rounded-full h-2 w-2"
                :class="widgetStreamPhase ? 'bg-cyan-500' : 'bg-primary'"
              />
            </span>
            <span class="truncate">{{ streamStatus || 'Working…' }}</span>
          </div>
        </div>
        <div
          ref="adaptiveScrollEl"
          class="chat-messages chat-pane-scroll flex-1 min-h-0 overflow-y-auto p-2.5 space-y-2 overscroll-contain scrollbar-gutter-stable"
          @scroll="onAdaptiveScroll"
        >
          <Motion
            v-for="(m, idx) in messages"
            :key="'a-' + idx"
            tag="div"
            class="space-y-1 premium-reveal"
            :initial="{ opacity: 0, y: 10 }"
            :animate="{ opacity: 1, y: 0 }"
            :transition="{ ...MOTION_BASE, delay: idx * 0.03 }"
          >
            <div class="flex" :class="m.role === 'user' ? 'justify-end' : 'justify-start'">
              <div
                class="max-w-[99%] md:max-w-[96%] rounded-2xl border px-3 py-2 shadow-sm transition-all duration-300 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-cyan-500/15"
                :class="m.role === 'user' ? 'bg-accent/15' : 'bg-card'"
              >
                <div class="text-xs text-muted-foreground mb-1 flex items-center gap-2 flex-wrap">
                  <span class="font-medium text-foreground/80">{{ m.role === 'user' ? 'You' : 'Assistant' }}</span>
                  <span
                    v-if="m.role === 'assistant' && m.strategy"
                    class="px-2 py-0.5 rounded-full border bg-background/70 text-[11px]"
                  >
                    {{ stratLabel(m.strategy) }}
                  </span>
                </div>
                <div class="leading-relaxed min-w-0">
                  <template
                    v-if="m.role === 'assistant' && isStreaming && idx === messages.length - 1 && !m.content.trim()"
                  >
                    <div class="typing-indicator" aria-label="Assistant is typing">
                      <span class="typing-dot" />
                      <span class="typing-dot" />
                      <span class="typing-dot" />
                    </div>
                  </template>
                  <template v-else-if="m.role === 'assistant' && assistantStreamPlain(idx)">
                    <div class="whitespace-pre-wrap">{{ cleanAssistantText(m.content) }}</div>
                  </template>
                  <template v-else-if="m.role === 'assistant'">
                    <div class="assistant-markdown" v-html="renderAssistantMarkdown(cleanAssistantText(m.content))" />
                  </template>
                  <template v-else>
                    <div class="whitespace-pre-wrap">{{ m.content }}</div>
                  </template>
                </div>

              </div>
            </div>

            <!-- LIVE streaming widget (Claude-style): visible the moment <WIDGET> opens. -->
            <Motion
              v-if="
                m.role === 'assistant' &&
                m.widgetStreaming &&
                !m.widgetSchema &&
                (m.widgetStream || (widgetGenerating && widgetGeneratingIdx === idx))
              "
              tag="div"
              class="mt-2"
              :initial="{ opacity: 0, y: 20, scale: 0.98 }"
              :animate="{ opacity: 1, y: 0, scale: 1 }"
              :transition="{ ...MOTION_BASE, delay: 0.02 }"
            >
              <div
                class="rounded-2xl border border-cyan-500/30 bg-card overflow-hidden shadow-sm widget-glow transition-shadow duration-500"
              >
                <div class="px-4 py-2 border-b flex items-center justify-between text-xs gap-2">
                  <div class="flex items-center gap-2 min-w-0">
                    <span class="relative flex h-2 w-2 shrink-0">
                      <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-70" />
                      <span class="relative inline-flex rounded-full h-2 w-2 bg-cyan-500" />
                    </span>
                    <span class="font-medium text-cyan-900 dark:text-cyan-100 truncate">Interactive widget</span>
                    <span class="text-muted-foreground shrink-0 hidden sm:inline">· building live</span>
                  </div>
                  <div class="flex items-center gap-2 shrink-0">
                    <span class="text-[10px] text-muted-foreground font-mono tabular-nums">
                      {{ ((m.widgetStream || '').length) }} chars
                    </span>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      class="h-7 px-2"
                      title="Download current widget draft (.html or .json)"
                      @click="downloadLiveWidgetDraft(m, idx)"
                    >
                      <ArrowDownTrayIcon class="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>
                <div class="p-3">
                  <!-- Live block-by-block build: render complete blocks from the partial stream
                       (salvage parser drops the in-progress block); the chart appears the moment
                       its JSON closes. -->
                  <WidgetSchemaRenderer
                    :json-str="m.widgetStream || ''"
                    :show-download="false"
                    :streaming="true"
                  />
                </div>
              </div>
            </Motion>

            <!-- Final widget (JSON schema mode). -->
            <Motion
              v-if="m.role === 'assistant' && m.widgetSchema && String(m.widgetSchema).trim()"
              tag="div"
              class="mt-2"
              :initial="{ opacity: 0, y: 16 }"
              :animate="{ opacity: 1, y: 0 }"
              :transition="MOTION_BASE"
            >
              <div
                class="rounded-2xl border bg-card overflow-hidden shadow-sm transition-shadow duration-300 hover:shadow-md"
              >
                <div class="px-4 py-2 border-b text-xs text-muted-foreground flex items-center gap-2">
                  <span class="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                  Interactive widget
                </div>
                <div class="p-4">
                  <WidgetSchemaRenderer
                    :json-str="m.widgetSchema"
                    :download-base="widgetDownloadBase(m, idx)"
                    @action="onWidgetAction"
                  />
                </div>
              </div>
            </Motion>

            <div
              v-if="m.role === 'assistant' && m.strategy && Array.isArray(m.xVec) && m.xVec.length"
              class="flex gap-2 mt-2"
            >
              <Button
                type="button"
                variant="outline"
                :disabled="m.rewardUsedUp"
                class="px-3"
                @click="submitReward(m, 1)"
              >
                Helpful
              </Button>
              <Button
                type="button"
                variant="outline"
                :disabled="m.rewardUsedDown"
                class="px-3"
                @click="submitReward(m, 0)"
              >
                Not helpful
              </Button>
            </div>
          </Motion>
        </div>
      </Card>

      <aside v-if="showTechPanels" class="hidden xl:block min-h-0">
        <div class="sticky top-2 max-h-[calc(100svh-1rem)] overflow-y-auto pr-1">
          <div class="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground mb-2">Insights</div>
          <div class="rounded-2xl border bg-card/60 backdrop-blur p-2 shadow-sm">
            <TechPanels
              :active-strategy="banditState.activeStrategy"
              :active-instruction="banditState.activeInstruction"
              :selected-strategy="banditState.selectedStrategy"
              :user-posterior="banditState.userPosterior"
              :global-posterior="banditState.globalPosterior"
              :user-b-posterior="banditState.userBPosterior"
              :global-n="banditState.globalN"
              :n-users="banditState.nUsers"
              :scores="banditState.scores"
              :x-vec="banditState.lastXVec"
              :reward-log="banditState.rewardLog"
            />
          </div>
        </div>
      </aside>

      <!-- Mobile/tablet insights drawer (shown below xl). -->
      <Teleport to="body">
        <Transition
          enter-active-class="transition-opacity duration-200 ease-out"
          enter-from-class="opacity-0"
          enter-to-class="opacity-100"
          leave-active-class="transition-opacity duration-150 ease-in"
          leave-from-class="opacity-100"
          leave-to-class="opacity-0"
        >
          <div
            v-if="showTechPanels"
            class="xl:hidden fixed inset-0 z-[250] bg-black/45 backdrop-blur-sm"
            @click="closeTechPanels"
          />
        </Transition>

        <Transition
          enter-active-class="transition-transform duration-250 ease-out"
          enter-from-class="translate-x-full"
          enter-to-class="translate-x-0"
          leave-active-class="transition-transform duration-200 ease-in"
          leave-from-class="translate-x-0"
          leave-to-class="translate-x-full"
        >
          <div v-if="showTechPanels" class="xl:hidden fixed inset-y-0 right-0 z-[260] w-[92vw] max-w-[440px]">
            <div class="h-full bg-background border-l shadow-2xl flex flex-col" @click.stop>
              <div class="px-3 py-2 border-b flex items-center justify-between">
                <div class="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">Insights</div>
                <Button type="button" variant="outline" class="h-9 text-xs" @click="closeTechPanels">Close</Button>
              </div>
              <div class="p-2 overflow-y-auto min-h-0">
                <div class="rounded-2xl border bg-card/60 backdrop-blur p-2 shadow-sm">
                  <TechPanels
                    :active-strategy="banditState.activeStrategy"
                    :active-instruction="banditState.activeInstruction"
                    :selected-strategy="banditState.selectedStrategy"
                    :user-posterior="banditState.userPosterior"
                    :global-posterior="banditState.globalPosterior"
                    :user-b-posterior="banditState.userBPosterior"
                    :global-n="banditState.globalN"
                    :n-users="banditState.nUsers"
                    :scores="banditState.scores"
                    :x-vec="banditState.lastXVec"
                    :reward-log="banditState.rewardLog"
                  />
                </div>
              </div>
            </div>
          </div>
        </Transition>
      </Teleport>
    </div>

    <form class="chat-input-shell flex gap-2 items-end shrink-0 pt-0 pb-0" @submit.prevent="onSend">
      <div class="flex-1 min-w-0">
        <div class="text-[10px] text-muted-foreground mb-0">Message (sent to both panes when baseline is visible)</div>
        <Input v-model="input" class="w-full" placeholder="Type a message…" />
      </div>
      <Button type="submit" :disabled="sending || !input.trim()" class="h-10 px-5">
        {{ sending ? 'Sending…' : 'Send →' }}
      </Button>
    </form>

    <PreferenceModal
      :open="prefOpen"
      @update:open="(v: boolean) => (prefOpen = v)"
      @submit="onPreferenceSubmit"
      @skip="prefOpen = false"
    />
  </div>
</template>

<style scoped>
.chat-pane-scroll {
  /* Explicit fallback height for browsers that mis-handle nested flex min-height. */
  height: var(--chat-pane-height, calc(100dvh - 14rem));
}

@media (min-width: 1024px) {
  .chat-pane-scroll {
    height: var(--chat-pane-height-lg, calc(100dvh - 15rem));
  }
}

.chat-input-shell {
  position: relative;
  background: color-mix(in oklab, var(--background) 84%, transparent);
  backdrop-filter: blur(8px);
  z-index: 10;
}

.typing-indicator {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 2px 0;
}

.typing-dot {
  width: 6px;
  height: 6px;
  border-radius: 9999px;
  background: linear-gradient(90deg, rgb(167 139 250), rgb(6 182 212));
  opacity: 0.55;
  animation: typingPulse 1.1s infinite ease-in-out;
}

.typing-dot:nth-child(2) {
  animation-delay: 0.15s;
}

.typing-dot:nth-child(3) {
  animation-delay: 0.3s;
}

@keyframes typingPulse {
  0%,
  80%,
  100% {
    transform: translateY(0);
    opacity: 0.45;
  }
  40% {
    transform: translateY(-4px);
    opacity: 1;
  }
}

.widget-glow {
  animation: widgetPanelBreathe 1.55s ease-in-out infinite;
}

@keyframes widgetPanelBreathe {
  0%,
  100% {
    box-shadow: 0 1px 0 0 rgba(6, 182, 212, 0.14);
  }
  50% {
    box-shadow: 0 10px 36px -4px rgba(6, 182, 212, 0.38);
  }
}

/* v-html markdown: tables, lists, emphasis */
.assistant-markdown :deep(p) {
  margin: 0 0 0.65em;
}
.assistant-markdown :deep(p:last-child) {
  margin-bottom: 0;
}
.assistant-markdown :deep(ul),
.assistant-markdown :deep(ol) {
  margin: 0.35em 0 0.65em;
  padding-left: 1.25rem;
}
.assistant-markdown :deep(ul) {
  list-style: disc;
}
.assistant-markdown :deep(ol) {
  list-style: decimal;
}
.assistant-markdown :deep(li) {
  margin: 0.15em 0;
}
.assistant-markdown :deep(table) {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8125rem;
  margin: 0.5rem 0 0.75rem;
}
.assistant-markdown :deep(th),
.assistant-markdown :deep(td) {
  border: 1px solid hsl(var(--border));
  padding: 0.4rem 0.55rem;
  text-align: left;
  vertical-align: top;
}
.assistant-markdown :deep(th) {
  background: hsl(var(--muted) / 0.45);
  font-weight: 600;
}
.assistant-markdown :deep(tr:nth-child(even) td) {
  background: hsl(var(--muted) / 0.12);
}
.assistant-markdown :deep(a) {
  color: hsl(var(--primary));
  text-decoration: underline;
  text-underline-offset: 2px;
}
.assistant-markdown :deep(code) {
  font-size: 0.85em;
  padding: 0.1em 0.35em;
  border-radius: 0.25rem;
  background: hsl(var(--muted) / 0.5);
}
.assistant-markdown :deep(pre) {
  margin: 0.5rem 0;
  padding: 0.65rem 0.75rem;
  border-radius: 0.375rem;
  background: hsl(var(--muted) / 0.35);
  overflow-x: auto;
  font-size: 0.8125rem;
}
.assistant-markdown :deep(pre code) {
  padding: 0;
  background: transparent;
}
.assistant-markdown :deep(blockquote) {
  margin: 0.5rem 0;
  padding-left: 0.75rem;
  border-left: 3px solid hsl(var(--border));
  color: hsl(var(--muted-foreground));
}
</style>
