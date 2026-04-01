<script setup lang="ts">
import { nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { Motion } from '@motionone/vue'
import Card from '@/components/ui/Card.vue'
import Input from '@/components/ui/Input.vue'
import Button from '@/components/ui/Button.vue'
import PreferenceModal from '@/components/PreferenceModal.vue'
import TechPanels from '@/components/TechPanels.vue'
import WidgetSchemaRenderer from '@/components/WidgetSchemaRenderer.vue'
import { getAccessToken } from '@/lib/auth'
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
import { MOTION_BASE, animatePulse, killAnimationsOf } from '@/lib/motion'
import { ArrowPathIcon, Cog6ToothIcon, EyeIcon, EyeSlashIcon } from '@/components/icons'

const router = useRouter()
const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5051'

const isStreaming = ref(false)
const streamStatus = ref('')
const healthOk = ref<boolean | null>(null)

type PlainMsg = { role: 'user' | 'assistant'; content: string; elapsed?: number; error?: string }

type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
  strategy?: string
  xVec?: number[]
  widgetHtml?: string
  widgetSchema?: string
  widgetHeight?: number
  rewardUsedUp?: boolean
  rewardUsedDown?: boolean
}

const plainMessages = ref<PlainMsg[]>([])
const messages = ref<ChatMessage[]>([])
const input = ref('')
const sending = ref(false)

const showBaseline = ref(true)
const showTechPanels = ref(true)
const prefOpen = ref(false)

const baselineScrollEl = ref<HTMLDivElement | null>(null)
const adaptiveScrollEl = ref<HTMLDivElement | null>(null)
const baselineStickToBottom = ref(true)
const adaptiveStickToBottom = ref(true)
const streamPulseEl = ref<HTMLDivElement | null>(null)
const widgetGenerating = ref(false)
const widgetGeneratingIdx = ref<number | null>(null)

let healthTimer: ReturnType<typeof setInterval> | null = null

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

async function onReset() {
  if (!confirm('Reset your bandit session? History for this account will be cleared.')) return
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
  const d = await res.json().catch(() => ({}))
  if (!res.ok || d.error) {
    plainMessages.value.push({
      role: 'assistant',
      content: '',
      error: typeof d.error === 'string' ? d.error : 'Baseline request failed',
    })
    return
  }
  plainMessages.value.push({
    role: 'assistant',
    content: d.response || '',
    elapsed: d.elapsed,
  })
}

async function runChatFallback(text: string, idx: number): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ message: text }),
  })
  const d = await res.json().catch(() => ({}))
  if (!res.ok || d.error) {
    messages.value[idx].content = `⚠ ${typeof d.error === 'string' ? d.error : 'Adaptive request failed'}`
    return false
  }
  const cur = messages.value[idx]
  cur.content = cleanAssistantText(d.response)
  cur.strategy = d.strategy
  cur.xVec = Array.isArray(d.x_vec) ? d.x_vec.map((n: unknown) => Number(n)) : []
  cur.widgetHtml = typeof d.widget_html === 'string' ? d.widget_html : ''
  cur.widgetSchema = typeof d.widget_schema === 'string' ? d.widget_schema : ''
  cur.widgetHeight = d.widget_height ? Number(d.widget_height) : 420
  cur.rewardUsedUp = false
  cur.rewardUsedDown = false
  banditState.activeStrategy = d.strategy || ''
  banditState.activeInstruction = d.instruction || ''
  banditState.selectedStrategy = d.strategy || ''
  banditState.scores = d.scores || null
  banditState.lastXVec = cur.xVec ?? null
  applyPosteriorPack(d)
  if (d.auto_detected && d.auto_r != null) {
    const autoReward = Number(d.auto_r)
    const autoStrategy = String(d.strategy ?? cur.strategy ?? 'unknown')
    // Prevent double-counting if the user later submits explicit feedback.
    cur.rewardUsedUp = autoReward === 1
    cur.rewardUsedDown = autoReward === 0

    banditState.rewardLog.unshift({
      strategy: d.strategy,
      reward: autoReward,
      detail: String(d.auto_reason || ''),
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
    strategy: d.strategy ?? 'unknown',
    elapsed: d.elapsed,
    widget_html: d.widget_html ?? '',
  })
  return true
}

function handleSseEvent(evt: Record<string, unknown>, idx: number) {
  const cur = messages.value[idx]
  if (!evt?.type) return

  if (evt.type === 'strategy') {
    streamStatus.value = 'Writing…'
    banditState.activeStrategy = String(evt.strategy ?? '')
    banditState.activeInstruction = String(evt.instruction ?? '')
    banditState.selectedStrategy = String(evt.strategy ?? '')
    banditState.scores = (evt.scores as Record<string, number>) || null
    banditState.lastXVec = Array.isArray(evt.x_vec) ? (evt.x_vec as number[]).map((n) => Number(n)) : null
    applyPosteriorPack({
      posterior: evt.posterior as PosteriorMap,
      global: evt.global as PosteriorMap,
      userb: evt.userb as PosteriorMap,
      global_n: evt.global_n as number,
    })
  }

  if (evt.type === 'response_delta') {
    streamStatus.value = 'Streaming…'
    cur.content += String((evt as { delta?: string }).delta ?? '')
    requestAdaptiveAutoScroll()
  }

  if (evt.type === 'widget_delta') {
    streamStatus.value = 'Generating widget…'
    widgetGenerating.value = true
    widgetGeneratingIdx.value = idx
  }

  if (evt.type === 'widget_start') {
    streamStatus.value = 'Generating widget…'
    widgetGenerating.value = true
    widgetGeneratingIdx.value = idx
  }

  if (evt.type === 'done') {
    streamStatus.value = ''
    widgetGenerating.value = false
    widgetGeneratingIdx.value = null
    const e = evt as {
      response?: string
      strategy?: string
      widget_html?: string
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
    const wHtml = typeof e.widget_html === 'string' ? e.widget_html.trim() : ''
    const wSch = typeof e.widget_schema === 'string' ? e.widget_schema.trim() : ''
    cur.widgetHtml = wHtml
    cur.widgetSchema = wSch
    cur.widgetHeight = e.widget_height ? Number(e.widget_height) : 420
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
      widget_html: e.widget_html ?? '',
    })
  }
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
  streamStatus.value = 'Thinking…'

  plainMessages.value.push({ role: 'user', content: text })
  messages.value.push({ role: 'user', content: text })
  const idx = messages.value.length
  messages.value.push({ role: 'assistant', content: '' })

  input.value = ''

  const plainJob = showBaseline.value
    ? runChatPlain(text).catch(() => {
        plainMessages.value.push({ role: 'assistant', content: '', error: 'Baseline connection error' })
      })
    : Promise.resolve()

  let widgetStream = ''
  let finalized = false
  let fallbackUsed = false

  try {
    const resp = await fetch(`${API_BASE}/api/chat_stream`, {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({ message: text }),
    })

    if (!resp.ok) {
      const errText = await resp.text().catch(() => '')
      throw new Error(errText || `Request failed (${resp.status})`)
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

        if (evt.type === 'widget_delta' && typeof (evt as { delta?: string }).delta === 'string') {
          widgetStream += (evt as { delta: string }).delta
        }

        handleSseEvent(evt, idx)

        if (evt.type === 'done') {
          finalized = true
          const cur = messages.value[idx]
          const wHtml = typeof (evt as { widget_html?: string }).widget_html === 'string' ? (evt as { widget_html: string }).widget_html.trim() : ''
          if (!wHtml && widgetStream) {
            cur.widgetHtml = widgetStream
          }
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

type HistoryPair = { user: string; assistant: string }

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
      { role: 'assistant' as const, content: cleanAssistantText(p.assistant) },
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
})

onUnmounted(() => {
  if (healthTimer) clearInterval(healthTimer)
  killAnimationsOf(streamPulseEl.value)
})

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

function cleanAssistantText(raw: string | undefined | null) {
  const s = String(raw ?? '')
  const noWidgetBlock = s.replace(/<WIDGET>[\s\S]*?<\/WIDGET>/gi, '')
  return noWidgetBlock.replace(/<\/?WIDGET>/gi, '').trim()
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
</script>

<template>
  <div class="h-full min-h-0 max-w-[1400px] mx-auto overflow-hidden grid grid-rows-[auto_minmax(0,1fr)_auto] gap-4">
    <div class="flex flex-wrap items-center gap-3 justify-between min-h-0">
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
      <div class="flex flex-wrap gap-2">
        <Button type="button" variant="outline" class="h-9 text-xs" @click="toggleBaseline">
          <span class="inline-flex items-center gap-1.5">
            <component :is="showBaseline ? EyeSlashIcon : EyeIcon" class="h-3.5 w-3.5" />
            {{ showBaseline ? 'Hide baseline' : 'Show baseline' }}
          </span>
        </Button>
        <Button type="button" variant="outline" class="h-9 text-xs" @click="showTechPanels = !showTechPanels">
          <span class="inline-flex items-center gap-1.5">
            <component :is="showTechPanels ? EyeSlashIcon : EyeIcon" class="h-3.5 w-3.5" />
            {{ showTechPanels ? 'Hide insights' : 'Show insights' }}
          </span>
        </Button>
        <Button type="button" variant="outline" class="h-9 text-xs inline-flex items-center gap-1.5" @click="openPrefs">
          <Cog6ToothIcon class="h-3.5 w-3.5" />
          Preferences
        </Button>
        <Button type="button" variant="outline" class="h-9 text-xs inline-flex items-center gap-1.5" @click="onReset">
          <ArrowPathIcon class="h-3.5 w-3.5" />
          Reset
        </Button>
        <div v-if="isStreaming" ref="streamPulseEl" class="text-xs text-muted-foreground self-center px-2">
          {{ streamStatus || 'Streaming…' }}
        </div>
      </div>
    </div>

    <div
      class="grid gap-4 min-h-0 overflow-hidden items-stretch"
      :class="showBaseline ? 'lg:grid-cols-2 xl:grid-cols-[360px_1fr_420px]' : 'lg:grid-cols-1 xl:grid-cols-[1fr_420px]'"
    >
      <Card v-if="showBaseline" class="flex flex-col min-h-0 overflow-hidden shadow-sm">
        <div class="px-4 py-3 border-b text-xs font-semibold text-muted-foreground uppercase tracking-wide flex items-center justify-between">
          <span>Baseline</span>
          <span class="text-[10px] font-medium px-2 py-0.5 rounded-full border bg-background/60">No bandit</span>
        </div>
        <div
          ref="baselineScrollEl"
          class="chat-messages chat-pane-scroll flex-1 min-h-0 overflow-y-auto p-4 space-y-3 overscroll-contain scrollbar-gutter-stable"
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
                  class="max-w-[92%] rounded-2xl border bg-accent/15 px-3.5 py-2.5 text-sm shadow-sm transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md hover:shadow-cyan-500/15"
                >
                  <div class="text-[10px] text-muted-foreground mb-0.5">You</div>
                  <div class="whitespace-pre-wrap">{{ m.content }}</div>
                </div>
              </div>
              <div v-else class="flex justify-start">
                <div
                  class="max-w-[92%] rounded-2xl border bg-card px-3.5 py-2.5 text-sm shadow-sm transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md hover:shadow-cyan-500/15"
                >
                  <div class="text-[10px] text-muted-foreground mb-0.5">Baseline</div>
                  <div v-if="m.error" class="text-red-400 text-sm">⚠ {{ m.error }}</div>
                  <div v-else class="whitespace-pre-wrap leading-relaxed">{{ m.content }}</div>
                  <div v-if="m.elapsed != null" class="text-[10px] text-muted-foreground mt-1">{{ m.elapsed }}s</div>
                </div>
              </div>
            </Motion>
          </template>
        </div>
      </Card>

      <Card class="flex flex-col min-h-0 overflow-hidden shadow-sm">
        <div class="px-4 py-3 border-b text-xs font-semibold text-muted-foreground uppercase tracking-wide flex items-center justify-between">
          <span>Adaptive</span>
          <span class="text-[10px] font-medium px-2 py-0.5 rounded-full border bg-background/60">Bandit layer</span>
        </div>
        <div
          ref="adaptiveScrollEl"
          class="chat-messages chat-pane-scroll flex-1 min-h-0 overflow-y-auto p-4 space-y-6 overscroll-contain scrollbar-gutter-stable"
          @scroll="onAdaptiveScroll"
        >
          <Motion
            v-for="(m, idx) in messages"
            :key="'a-' + idx"
            tag="div"
            class="space-y-2 premium-reveal"
            :initial="{ opacity: 0, y: 10 }"
            :animate="{ opacity: 1, y: 0 }"
            :transition="{ ...MOTION_BASE, delay: idx * 0.03 }"
          >
            <div class="flex" :class="m.role === 'user' ? 'justify-end' : 'justify-start'">
              <div
                class="max-w-[92%] md:max-w-[78%] rounded-2xl border px-4 py-3 shadow-sm transition-all duration-300 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-cyan-500/15"
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
                <div class="whitespace-pre-wrap leading-relaxed">
                  <template
                    v-if="m.role === 'assistant' && isStreaming && idx === messages.length - 1 && !m.content.trim()"
                  >
                    <div class="typing-indicator" aria-label="Assistant is typing">
                      <span class="typing-dot" />
                      <span class="typing-dot" />
                      <span class="typing-dot" />
                    </div>
                  </template>
                  <template v-else>
                    {{ m.content }}
                  </template>
                </div>

                <div
                  v-if="
                    m.role === 'assistant' &&
                    widgetGenerating &&
                    widgetGeneratingIdx === idx &&
                    !m.widgetHtml &&
                    !m.widgetSchema
                  "
                  class="mt-2 text-xs text-muted-foreground inline-flex items-center gap-2"
                >
                  <span class="inline-flex h-2 w-2 rounded-full bg-cyan-500 animate-pulse" />
                  Generating widget…
                </div>
              </div>
            </div>

            <div v-if="m.role === 'assistant' && m.widgetHtml && String(m.widgetHtml).trim()" class="mt-2">
              <div class="rounded-2xl border bg-card overflow-hidden">
                <div class="px-4 py-2 border-b text-xs text-muted-foreground">Interactive widget</div>
                <iframe
                  :srcdoc="m.widgetHtml"
                  sandbox="allow-scripts allow-same-origin"
                  class="w-full widget-frame border-0"
                  :style="{ height: `${m.widgetHeight || 420}px` }"
                />
              </div>
            </div>

            <div v-if="m.role === 'assistant' && m.widgetSchema && String(m.widgetSchema).trim() && !m.widgetHtml" class="mt-2">
              <div class="rounded-2xl border bg-card overflow-hidden">
                <div class="px-4 py-2 border-b text-xs text-muted-foreground">Widget (JSON schema)</div>
                <div class="p-4">
                  <WidgetSchemaRenderer :json-str="m.widgetSchema" />
                </div>
              </div>
            </div>

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
        <div class="sticky top-20 max-h-[calc(100svh-7.5rem)] overflow-y-auto pr-1">
          <div class="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-3">Insights</div>
          <div class="rounded-2xl border bg-card/60 backdrop-blur p-3 shadow-sm">
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
    </div>

    <form class="chat-input-shell flex gap-2 items-end shrink-0 pb-1 pt-2" @submit.prevent="onSend">
      <div class="flex-1 min-w-0">
        <div class="text-xs text-muted-foreground mb-1">Message (sent to both panes when baseline is visible)</div>
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
  height: var(--chat-pane-height, calc(100dvh - 18rem));
}

@media (min-width: 1024px) {
  .chat-pane-scroll {
    height: var(--chat-pane-height-lg, calc(100dvh - 19.25rem));
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
</style>
