<script setup lang="ts">
/**
 * Registry-driven renderer (POC).
 *
 * Instead of a hardcoded v-if chain, this parses the widget JSON into blocks
 * and resolves each block's `type` against the widget registry, mounting the
 * real component via <component :is>. Add a component to the registry once and
 * it renders here automatically — no change to this file.
 */
import { computed } from 'vue'
import { Motion } from '@motionone/vue'
import Button from '@/components/ui/Button.vue'
import { ArrowDownTrayIcon } from '@/components/icons'
import { resolveWidget } from '@/lib/widgetRegistry'
import { normalizeWidgetBlock } from '@/lib/progressiveWidget'
import { chartHasRenderableData } from '@/lib/echartsOption'
import { downloadTextAsFile, prettifyJsonIfPossible } from '@/lib/downloadFile'
import { widgetToHtml } from '@/lib/exportWidgetHtml'
import { showToast } from '@/lib/toast'

const props = withDefaults(
  defineProps<{ jsonStr: string; downloadBase?: string; showDownload?: boolean; streaming?: boolean }>(),
  { showDownload: true, streaming: false },
)
// Bubbles up action_row button clicks (the follow-up prompt text) to the chat.
const emit = defineEmits<{ (e: 'action', text: string): void }>()

type Block = Record<string, unknown> & { type?: string }

function stripFences(s: string): string {
  const m = s.match(/```(?:json|html|javascript|js)?\s*([\s\S]*?)```/i)
  if (m) return m[1].trim()
  return s.replace(/```\w*/g, '').trim()
}

/**
 * Salvage complete block objects from a truncated/invalid layout string.
 * Walks the `layout` array brace-by-brace (string-aware) and keeps every fully
 * closed `{...}` object that has a `type`. A cut-off final object is dropped.
 * This guarantees we NEVER fall back to dumping raw JSON to the user.
 */
function salvageBlocks(s: string): Block[] {
  const li = s.search(/"(?:layout|blocks|components)"\s*:\s*\[/)
  let start = li >= 0 ? s.indexOf('[', li) + 1 : s.indexOf('[')
  if (start <= 0) return []
  const out: Block[] = []
  const n = s.length
  let i = start
  while (i < n) {
    while (i < n && s[i] !== '{') {
      if (s[i] === ']') return out
      i++
    }
    if (i >= n) break
    let depth = 0
    let inStr = false
    let esc = false
    let j = i
    for (; j < n; j++) {
      const ch = s[j]
      if (inStr) {
        if (esc) esc = false
        else if (ch === '\\') esc = true
        else if (ch === '"') inStr = false
      } else if (ch === '"') inStr = true
      else if (ch === '{') depth++
      else if (ch === '}') {
        depth--
        if (depth === 0) {
          j++
          break
        }
      }
    }
    if (depth !== 0) break // incomplete (truncated) object → stop salvaging
    try {
      const o = JSON.parse(s.slice(i, j))
      if (o && typeof o === 'object' && typeof o.type === 'string') out.push(o as Block)
    } catch {
      /* skip unparseable element */
    }
    i = j
  }
  return out
}

// A text block's content that is actually JSON (a block/layout the model stuffed into
// content) — must NEVER render as raw text. Detect and drop.
const NUMERIC_ARRAY_RE = /^\s*\[\s*-?\d+(\.\d+)?(\s*,\s*-?\d+(\.\d+)?)*\s*\]\s*$/
function textContentIsJson(content: string): boolean {
  const t = String(content || '').trim()
  if (!(t.startsWith('{') || t.startsWith('['))) return false
  // looks like a widget block/array, or parses as a JSON object/array
  if (/"(?:type|layout|items|tone|label|series|chart|value)"\s*:/.test(t)) return true
  try {
    const o = JSON.parse(t)
    return o && typeof o === 'object'
  } catch {
    return /^[{[]/.test(t) // truncated JSON-ish → still drop
  }
}

function finalizeBlocks(layout: unknown[]): Block[] {
  const normalized = layout.map((b) => normalizeWidgetBlock(b)) as Block[]
  return normalized.filter((b) => {
    const type = String(b.type || '').toLowerCase()
    // Never show an empty chart card — drop charts with no renderable data outright.
    if (type === 'chart' && !chartHasRenderableData((b as { chart?: any }).chart || {})) return false
    if (type === 'text') {
      const c = String((b as { content?: string }).content ?? '')
      // Drop numeric-array junk AND any text block whose content is raw JSON.
      if (NUMERIC_ARRAY_RE.test(c) || textContentIsJson(c)) return false
    }
    return true
  })
}

function parseLayout(raw: string): Block[] | null {
  let s = String(raw || '').trim()
  if (!s) return null
  if (s.includes('```')) s = stripFences(s)

  let parsed: unknown = null
  try {
    parsed = JSON.parse(s)
  } catch {
    const i = s.indexOf('{')
    const j = s.lastIndexOf('}')
    if (i >= 0 && j > i) {
      try {
        parsed = JSON.parse(s.slice(i, j + 1))
      } catch {
        parsed = null
      }
    }
  }

  let layout: unknown
  if (Array.isArray(parsed)) {
    layout = parsed
  } else if (parsed && typeof parsed === 'object') {
    const o = parsed as Record<string, unknown>
    layout = o.layout ?? o.blocks ?? o.components
    if (!Array.isArray(layout) && typeof o.type === 'string') layout = [o]
  }

  if (Array.isArray(layout)) return finalizeBlocks(layout as unknown[])

  // Clean parse failed (likely truncated stream) — salvage whatever complete blocks exist
  // instead of dumping raw JSON to the user.
  const salvaged = salvageBlocks(s)
  if (salvaged.length) return finalizeBlocks(salvaged)
  return null
}

const parsed = computed(() => parseLayout(props.jsonStr))
const blocks = computed(() => parsed.value ?? [])
const hasRaw = computed(() => Boolean(String(props.jsonStr || '').trim()))
// Only a genuine parse failure (null) shows the error; a valid-but-empty layout renders nothing.
const parseFailed = computed(() => hasRaw.value && parsed.value === null)

function resolve(type?: string) {
  return resolveWidget(String(type || ''))
}

function downloadStem(): string {
  const stem = String(props.downloadBase || 'widget-schema')
    .replace(/[^a-zA-Z0-9_-]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 80)
  return stem || 'widget-schema'
}

function downloadJson() {
  const raw = String(props.jsonStr || '').trim()
  if (!raw) {
    showToast({ title: 'Nothing to download', message: 'Widget JSON is empty.' })
    return
  }
  downloadTextAsFile(prettifyJsonIfPossible(raw), `${downloadStem()}.json`, 'application/json;charset=utf-8')
}

function downloadHtml() {
  const raw = String(props.jsonStr || '').trim()
  if (!raw) {
    showToast({ title: 'Nothing to download', message: 'Widget is empty.' })
    return
  }
  // Deterministic, no LLM: same JSON the renderer uses → self-contained interactive HTML.
  const html = widgetToHtml(raw, downloadStem())
  downloadTextAsFile(html, `${downloadStem()}.html`, 'text/html;charset=utf-8')
}
</script>

<template>
  <div class="space-y-2">
    <div v-if="showDownload && hasRaw" class="flex justify-end gap-1.5">
      <Button
        type="button"
        variant="outline"
        size="sm"
        class="h-7 px-2 text-[11px]"
        title="Download as a self-contained interactive HTML file"
        @click="downloadHtml"
      >
        <ArrowDownTrayIcon class="h-3.5 w-3.5" /> HTML
      </Button>
      <Button
        type="button"
        variant="outline"
        size="sm"
        class="h-7 px-2 text-[11px]"
        title="Download widget schema as JSON"
        @click="downloadJson"
      >
        <ArrowDownTrayIcon class="h-3.5 w-3.5" /> JSON
      </Button>
    </div>

    <div v-if="blocks.length" class="ws-root space-y-3 text-sm">
      <template v-for="(block, i) in blocks" :key="i">
        <Motion
          tag="div"
          :initial="{ opacity: 0, y: 14, scale: 0.985 }"
          :animate="{ opacity: 1, y: 0, scale: 1 }"
          :transition="{ duration: 0.38, easing: [0.22, 1, 0.36, 1], delay: i * 0.055 }"
          class="will-change-[transform,opacity]"
        >
          <component
            :is="resolve(block.type)!.component"
            v-if="resolve(block.type)"
            :block="block"
            @action="(t: string) => emit('action', t)"
          />
          <div
            v-else
            class="rounded-lg border border-dashed border-amber-500/40 bg-amber-500/5 px-3 py-2 text-[11px] text-muted-foreground"
          >
            <div class="font-medium text-foreground/90 mb-1">
              Unsupported widget block:
              <span class="font-mono">{{ String(block.type ?? 'unknown') }}</span>
            </div>
            <pre class="max-h-32 overflow-auto whitespace-pre-wrap break-all text-[10px] leading-snug">{{
              JSON.stringify(block, null, 2)
            }}</pre>
          </div>
        </Motion>
      </template>
    </div>

    <!-- While streaming, stay quiet until the first block completes (panel header shows "building live"). -->
    <div
      v-else-if="parseFailed && !streaming"
      class="rounded-lg border border-dashed border-amber-500/40 bg-amber-500/5 px-3 py-3 text-xs text-muted-foreground"
    >
      <!-- Never dump raw JSON to the user — show a friendly note only. -->
      The widget couldn’t be built for this answer. Try rephrasing or ask again.
    </div>
  </div>
</template>
