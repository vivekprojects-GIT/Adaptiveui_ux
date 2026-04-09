<script setup lang="ts">
import { computed } from 'vue'
import Button from '@/components/ui/Button.vue'
import WidgetSchemaChart from '@/components/WidgetSchemaChart.vue'

const props = defineProps<{
  jsonStr: string
}>()

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
  return { layout: layout as Block[] }
}

const schema = computed(() => parseWidgetSchemaLoose(props.jsonStr))

const blocks = computed(() => schema.value?.layout ?? [])
</script>

<template>
  <div v-if="schema" class="ws-root space-y-3 text-sm">
    <template v-for="(block, i) in blocks" :key="i">
      <div v-if="block.type === 'text'" class="whitespace-pre-wrap leading-relaxed">
        {{ (block as { content?: string }).content || '' }}
      </div>

      <div v-else-if="block.type === 'kpi_row'" class="grid grid-cols-2 sm:grid-cols-3 gap-2">
        <div
          v-for="(it, j) in (block as { items?: { label?: string; value?: string; tone?: string }[] }).items || []"
          :key="j"
          class="rounded-xl border bg-card px-3 py-2"
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

      <div v-else-if="block.type === 'table'" class="rounded-xl border bg-card overflow-hidden">
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
    </template>
  </div>
  <div v-else class="text-xs text-muted-foreground">Invalid widget schema JSON.</div>
</template>
