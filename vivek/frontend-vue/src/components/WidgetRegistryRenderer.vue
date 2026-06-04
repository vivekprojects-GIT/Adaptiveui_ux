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
import { downloadTextAsFile, prettifyJsonIfPossible } from '@/lib/downloadFile'
import { showToast } from '@/lib/toast'

const props = withDefaults(
  defineProps<{ jsonStr: string; downloadBase?: string; showDownload?: boolean }>(),
  { showDownload: true },
)

type Block = Record<string, unknown> & { type?: string }

function stripFences(s: string): string {
  const m = s.match(/```(?:json|html|javascript|js)?\s*([\s\S]*?)```/i)
  if (m) return m[1].trim()
  return s.replace(/```\w*/g, '').trim()
}

function parseLayout(raw: string): Block[] | null {
  let s = String(raw || '').trim()
  if (!s) return null
  if (s.includes('```')) s = stripFences(s)

  let parsed: unknown
  try {
    parsed = JSON.parse(s)
  } catch {
    const i = s.indexOf('{')
    const j = s.lastIndexOf('}')
    if (i < 0 || j <= i) return null
    try {
      parsed = JSON.parse(s.slice(i, j + 1))
    } catch {
      return null
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
  if (!Array.isArray(layout)) return null
  const normalized = (layout as unknown[]).map((b) => normalizeWidgetBlock(b)) as Block[]
  // Drop bare numeric-array text blocks (e.g. tic-tac-toe win lines "[0,1,2]") — not real content.
  const NUMERIC_ARRAY_RE = /^\s*\[\s*-?\d+(\.\d+)?(\s*,\s*-?\d+(\.\d+)?)*\s*\]\s*$/
  return normalized.filter(
    (b) =>
      !(
        String(b.type || '').toLowerCase() === 'text' &&
        NUMERIC_ARRAY_RE.test(String((b as { content?: string }).content ?? ''))
      ),
  )
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
</script>

<template>
  <div class="space-y-2">
    <div v-if="showDownload && hasRaw" class="flex justify-end">
      <Button
        type="button"
        variant="outline"
        size="sm"
        class="h-7 px-2"
        title="Download widget schema as JSON"
        @click="downloadJson"
      >
        <ArrowDownTrayIcon class="h-3.5 w-3.5" />
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

    <div
      v-else-if="parseFailed"
      class="rounded-lg border border-dashed border-amber-500/40 bg-amber-500/5 px-3 py-3 text-xs text-muted-foreground"
    >
      <p class="font-medium text-foreground/90 mb-1">Could not parse widget JSON</p>
      <pre class="max-h-48 overflow-auto whitespace-pre-wrap break-all text-[10px] leading-snug">{{
        props.jsonStr.slice(0, 2000)
      }}</pre>
    </div>
  </div>
</template>
