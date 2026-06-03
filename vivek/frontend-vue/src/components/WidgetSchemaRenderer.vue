<script setup lang="ts">
import { computed } from 'vue'
import { Motion } from '@motionone/vue'
import Button from '@/components/ui/Button.vue'
import WidgetSchemaChart from '@/components/WidgetSchemaChart.vue'
import { ArrowDownTrayIcon } from '@/components/icons'
import { normalizeWidgetBlock } from '@/lib/progressiveWidget'
import { downloadTextAsFile, prettifyJsonIfPossible } from '@/lib/downloadFile'
import { showToast } from '@/lib/toast'

const props = withDefaults(
  defineProps<{
    jsonStr: string
    /** Filename stem for JSON download (no extension). */
    downloadBase?: string
    showDownload?: boolean
  }>(),
  { showDownload: true },
)

type ChartSchema = {
  kind?: string
  x_label?: string
  y_label?: string
  series?: { name?: string; color?: string; values?: [number, number][] }[]
}

type Block =
  | { type: 'text'; content?: string }
  | { type: 'kpi_row'; items?: { label?: string; value?: string; tone?: string }[] }
  | { type: 'chart'; title?: string; chart?: ChartSchema }
  | {
      type: 'image'
      title?: string
      src?: string
      alt?: string
      caption?: string
      fit?: string
    }
  | { type: 'table'; title?: string; columns?: string[]; rows?: (string | number)[][] }
  | { type: 'action_row'; buttons?: { label?: string; intent?: string }[] }

function stripWidgetFences(s: string): string {
  const m = s.match(/```(?:json|html|javascript|js)?\s*([\s\S]*?)```/i)
  if (m) return m[1].trim()
  return s.replace(/```\w*/g, '').trim()
}

/** Tolerate markdown fences, prose before JSON, and blocks vs layout (matches backend finalize). */
function parseWidgetSchemaLoose(raw: string): { layout: Block[] } | null {
  let s = String(raw || '').trim()
  if (!s) return null
  if (s.includes('```')) s = stripWidgetFences(s)

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
      const a = s.indexOf('[')
      const b = s.lastIndexOf(']')
      if (a < 0 || b <= a) return null
      try {
        parsed = JSON.parse(s.slice(a, b + 1))
      } catch {
        return null
      }
    }
  }

  if (Array.isArray(parsed)) {
    return { layout: parsed as Block[] }
  }
  if (!parsed || typeof parsed !== 'object') return null
  let o = parsed as Record<string, unknown>
  if (!Array.isArray(o.layout) && typeof o.widget === 'object' && o.widget !== null) {
    o = { ...o, ...(o.widget as Record<string, unknown>) }
  }
  if (!Array.isArray(o.layout) && typeof o.schema === 'object' && o.schema !== null) {
    o = { ...o, ...(o.schema as Record<string, unknown>) }
  }
  let layout = o.layout
  if (!Array.isArray(layout) && Array.isArray(o.blocks)) layout = o.blocks
  if (!Array.isArray(layout) && Array.isArray(o.components)) layout = o.components
  if (!Array.isArray(layout) && Array.isArray(o.Layout)) layout = o.Layout
  if (!Array.isArray(layout) && layout && typeof layout === 'object' && 'type' in (layout as object)) {
    layout = [layout]
  }
  const known = new Set(['text', 'kpi_row', 'chart', 'table', 'action_row'])
  if (
    !Array.isArray(layout) &&
    typeof o.type === 'string' &&
    known.has(String(o.type))
  ) {
    layout = [o]
  }
  if (!Array.isArray(layout)) return null
  const normalized = (layout as unknown[]).map((item) => normalizeWidgetBlock(item)) as Block[]
  return { layout: normalized }
}

/** Plain numbers in `layout` (e.g. [100, 40]) are valid JSON but not valid blocks — models sometimes emit this by mistake. */
const NUMERIC_TEXT_ONLY = /^\s*-?\d+(\.\d+)?\s*$/

/** Turn multiple numeric-only text lines into one KPI row — no user-facing warning. */
function coalesceNumericTextBlocks(layout: Block[]): Block[] {
  if (layout.length < 2) return layout
  const allText = layout.every((b) => b.type === 'text')
  if (!allText) return layout
  const contents = layout.map((b) => String((b as { content?: string }).content ?? '').trim())
  const allNumeric = contents.every((c) => NUMERIC_TEXT_ONLY.test(c))
  if (!allNumeric) return layout
  return [
    {
      type: 'kpi_row',
      items: contents.map((value, i) => ({
        label: `Metric ${i + 1}`,
        value,
        tone: 'neutral',
      })),
    },
  ]
}

const schema = computed(() => parseWidgetSchemaLoose(props.jsonStr))

const blocks = computed(() => schema.value?.layout ?? [])

const displayBlocks = computed(() => coalesceNumericTextBlocks(blocks.value))

const hasRaw = computed(() => Boolean(String(props.jsonStr || '').trim()))

function downloadFileStem(): string {
  const stem = String(props.downloadBase || 'widget-schema')
    .replace(/[^a-zA-Z0-9_-]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 80)
  return stem || 'widget-schema'
}

function downloadSchemaJson() {
  const raw = String(props.jsonStr || '').trim()
  if (!raw) {
    showToast({ title: 'Nothing to download', message: 'Widget JSON is empty.' })
    return
  }
  const body = prettifyJsonIfPossible(raw)
  downloadTextAsFile(body, `${downloadFileStem()}.json`, 'application/json;charset=utf-8')
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
        @click="downloadSchemaJson"
      >
        <ArrowDownTrayIcon class="h-3.5 w-3.5" />
      </Button>
    </div>

    <div v-if="schema && displayBlocks.length" class="ws-root space-y-3 text-sm">
    <template v-for="(block, i) in displayBlocks" :key="i">
      <Motion
        tag="div"
        :initial="{ opacity: 0, y: 14, scale: 0.985 }"
        :animate="{ opacity: 1, y: 0, scale: 1 }"
        :transition="{ duration: 0.38, easing: [0.22, 1, 0.36, 1], delay: i * 0.055 }"
        class="will-change-[transform,opacity]"
      >
      <div v-if="block.type === 'text'" class="whitespace-pre-wrap leading-relaxed motion-safe:transition-opacity motion-safe:duration-300">
        {{ (block as { content?: string }).content || '' }}
      </div>

      <div v-else-if="block.type === 'kpi_row'" class="grid grid-cols-2 sm:grid-cols-3 gap-2">
        <div
          v-for="(it, j) in (block as { items?: { label?: string; value?: string; tone?: string }[] }).items || []"
          :key="j"
          class="rounded-xl border bg-card px-3 py-2 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md"
        >
          <div class="text-[10px] text-muted-foreground uppercase tracking-wide">{{ it.label }}</div>
          <div
            class="text-lg font-semibold mt-0.5"
            :class="{
              'text-emerald-600 dark:text-emerald-400': it.tone === 'positive',
              'text-red-600 dark:text-red-400': it.tone === 'negative',
            }"
          >
            {{ it.value }}
          </div>
        </div>
      </div>

      <WidgetSchemaChart
        v-else-if="block.type === 'chart'"
        :title="(block as Extract<Block, { type: 'chart' }>).title"
        :chart="(block as Extract<Block, { type: 'chart' }>).chart || {}"
      />

      <div
        v-else-if="block.type === 'image'"
        class="rounded-xl border bg-card overflow-hidden transition-shadow duration-300 hover:shadow-md"
      >
        <div
          v-if="(block as Extract<Block, { type: 'image' }>).title"
          class="px-3 py-2 border-b text-xs font-medium"
        >
          {{ (block as Extract<Block, { type: 'image' }>).title }}
        </div>
        <div v-if="(block as Extract<Block, { type: 'image' }>).src" class="p-2 flex justify-center bg-muted/15">
          <img
            :src="(block as Extract<Block, { type: 'image' }>).src || ''"
            :alt="(block as Extract<Block, { type: 'image' }>).alt || 'Widget image'"
            class="max-w-full rounded-lg"
            :class="
              (block as Extract<Block, { type: 'image' }>).fit === 'cover'
                ? 'object-cover w-full max-h-[min(420px,55vh)]'
                : 'object-contain max-h-[min(420px,55vh)] h-auto'
            "
            loading="lazy"
            referrerpolicy="no-referrer"
          />
        </div>
        <div v-else class="px-3 py-2 text-xs text-muted-foreground">No image URL in block.</div>
        <div
          v-if="(block as Extract<Block, { type: 'image' }>).caption"
          class="px-3 py-2 text-[11px] text-muted-foreground border-t"
        >
          {{ (block as Extract<Block, { type: 'image' }>).caption }}
        </div>
      </div>

      <div v-else-if="block.type === 'table'" class="rounded-xl border bg-card overflow-hidden transition-shadow duration-300 hover:shadow-md">
        <div v-if="(block as { title?: string }).title" class="px-3 py-2 border-b text-xs font-medium">
          {{ (block as { title?: string }).title }}
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-xs">
            <thead v-if="(block as { columns?: string[] }).columns?.length">
              <tr>
                <th
                  v-for="(c, ci) in (block as { columns?: string[] }).columns"
                  :key="ci"
                  class="text-left px-3 py-2 border-b bg-muted/40 font-medium"
                >
                  {{ c }}
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, ri) in (block as { rows?: unknown[][] }).rows || []" :key="ri">
                <td v-for="(cell, ci) in row" :key="ci" class="px-3 py-1.5 border-b border-border/60">
                  {{ cell }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div v-else-if="block.type === 'action_row'" class="flex flex-wrap gap-2">
        <Button
          v-for="(b, bi) in (block as { buttons?: { label?: string }[] }).buttons || []"
          :key="bi"
          type="button"
          variant="outline"
          size="sm"
          disabled
          :title="(b as { intent?: string }).intent ? `Intent: ${(b as { intent?: string }).intent}` : undefined"
        >
          {{ b.label || 'Action' }}
        </Button>
      </div>

      <!-- Model returned a block shape we do not render yet — show something instead of a blank card. -->
      <div
        v-else
        class="rounded-lg border border-dashed border-amber-500/40 bg-amber-500/5 px-3 py-2 text-[11px] text-muted-foreground"
      >
        <div class="font-medium text-foreground/90 mb-1">
          Unsupported widget block:
          <span class="font-mono">{{ String((block as { type?: unknown }).type ?? 'unknown') }}</span>
        </div>
        <pre class="max-h-32 overflow-auto whitespace-pre-wrap break-all text-[10px] leading-snug">{{
          JSON.stringify(block, null, 2)
        }}</pre>
      </div>
      </Motion>
    </template>
    </div>

    <div
      v-else-if="schema && !blocks.length"
    class="rounded-lg border border-dashed border-muted-foreground/30 bg-muted/20 px-3 py-3 text-xs text-muted-foreground"
  >
    <p class="font-medium text-foreground/90 mb-1">Widget schema parsed but is empty</p>
    <p class="mb-2">The model returned a layout array with no blocks, or blocks could not be read.</p>
    <pre class="max-h-40 overflow-auto whitespace-pre-wrap break-all text-[10px] text-foreground/70">{{
      props.jsonStr.slice(0, 1200)
    }}</pre>
    </div>

    <div
      v-else-if="hasRaw"
    class="rounded-lg border border-dashed border-amber-500/40 bg-amber-500/5 px-3 py-3 text-xs text-muted-foreground"
  >
    <p class="font-medium text-foreground/90 mb-1">Could not parse widget JSON</p>
    <p class="mb-2">The assistant returned widget text the UI could not parse into blocks. Raw payload (trimmed):</p>
    <pre class="max-h-48 overflow-auto whitespace-pre-wrap break-all text-[10px] leading-snug">{{
      props.jsonStr.slice(0, 2000)
    }}</pre>
    </div>
  </div>
</template>
