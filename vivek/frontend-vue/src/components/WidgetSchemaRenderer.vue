<script setup lang="ts">
import { computed } from 'vue'
import Button from '@/components/ui/Button.vue'

const props = defineProps<{
  jsonStr: string
}>()

type Block =
  | { type: 'text'; content?: string }
  | { type: 'kpi_row'; items?: { label?: string; value?: string; tone?: string }[] }
  | { type: 'chart'; title?: string }
  | { type: 'table'; title?: string; columns?: string[]; rows?: (string | number)[][] }
  | { type: 'action_row'; buttons?: { label?: string; intent?: string }[] }

const schema = computed(() => {
  try {
    const o = JSON.parse(String(props.jsonStr || ''))
    if (!o || !Array.isArray(o.layout)) return null
    return o as { layout: Block[] }
  } catch {
    return null
  }
})

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

      <div v-else-if="block.type === 'chart'" class="rounded-xl border bg-card p-3">
        <div v-if="(block as { title?: string }).title" class="text-xs font-medium mb-2">
          {{ (block as { title?: string }).title }}
        </div>
        <pre class="text-[11px] text-muted-foreground whitespace-pre-wrap">Chart schema received. (Wire ECharts here.)</pre>
      </div>

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
