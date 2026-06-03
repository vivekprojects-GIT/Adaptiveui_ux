<script setup lang="ts">
import { computed } from 'vue'
import { Motion } from '@motionone/vue'
import Button from '@/components/ui/Button.vue'
import WidgetSchemaChart from '@/components/WidgetSchemaChart.vue'
import { parseProgressiveWidgetSchema, type WidgetBlock } from '@/lib/progressiveWidget'

/**
 * Renders widget blocks progressively as JSON streams in. Each block fades up
 * independently (Claude-artifact style) and we show a soft "pending" block for
 * whichever block the model is currently generating.
 */
const props = defineProps<{
  rawStream: string
  finalized?: boolean
}>()

const state = computed(() => parseProgressiveWidgetSchema(props.rawStream))
const blocks = computed<WidgetBlock[]>(() => state.value.blocks)
const pendingType = computed<string | null>(() => (props.finalized ? null : state.value.pendingBlockHint))

function pendingLabel(t: string | null): string {
  if (!t) return 'Building block'
  const labels: Record<string, string> = {
    text: 'Writing narrative',
    kpi_row: 'Preparing KPIs',
    chart: 'Rendering chart',
    image: 'Loading image',
    table: 'Composing table',
    action_row: 'Wiring actions',
  }
  return labels[t] || `Building ${t}`
}
</script>

<template>
  <div class="live-widget space-y-3 text-sm">
    <template v-for="(block, i) in blocks" :key="`lw-${i}`">
      <Motion
        tag="div"
        :initial="{ opacity: 0, y: 8, scale: 0.985 }"
        :animate="{ opacity: 1, y: 0, scale: 1 }"
        :transition="{ duration: 0.32, easing: [0.16, 1, 0.3, 1] }"
      >
        <div v-if="block.type === 'text'" class="whitespace-pre-wrap leading-relaxed">
          {{ (block as { content?: string }).content || '' }}
        </div>

        <div v-else-if="block.type === 'kpi_row'" class="grid grid-cols-2 sm:grid-cols-3 gap-2">
          <div
            v-for="(it, j) in (block as { items?: { label?: string; value?: string; tone?: string }[] }).items || []"
            :key="j"
            class="rounded-xl border bg-card px-3 py-2 shadow-sm"
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
          :title="(block as { title?: string }).title"
          :chart="(block as { chart?: Record<string, unknown> }).chart as any || {}"
        />

        <div v-else-if="block.type === 'image'" class="rounded-xl border bg-card overflow-hidden shadow-sm">
          <div
            v-if="(block as { title?: string }).title"
            class="px-3 py-2 border-b text-xs font-medium"
          >
            {{ (block as { title?: string }).title }}
          </div>
          <div class="p-2 flex justify-center bg-muted/15">
            <img
              :src="(block as { src?: string }).src || ''"
              :alt="(block as { alt?: string }).alt || 'Widget image'"
              class="max-w-full max-h-[min(420px,55vh)] rounded-lg object-contain"
              loading="lazy"
              referrerpolicy="no-referrer"
            />
          </div>
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
            v-for="(b, bi) in (block as { buttons?: { label?: string; intent?: string }[] }).buttons || []"
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

        <div v-else class="rounded-lg border border-dashed border-muted-foreground/30 bg-muted/20 px-3 py-2 text-xs whitespace-pre-wrap">
          {{ JSON.stringify(block, null, 2) }}
        </div>
      </Motion>
    </template>

    <Transition
      enter-active-class="transition-all duration-300 ease-out"
      enter-from-class="opacity-0 translate-y-1"
      enter-to-class="opacity-100 translate-y-0"
      leave-active-class="transition-all duration-200 ease-in"
      leave-from-class="opacity-100 translate-y-0"
      leave-to-class="opacity-0 -translate-y-1"
    >
      <div
        v-if="pendingType"
        class="pending-block rounded-xl border border-cyan-500/30 bg-cyan-500/5 dark:bg-cyan-950/25 px-3 py-2 flex items-center gap-3"
      >
        <span class="relative flex h-2.5 w-2.5 shrink-0">
          <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-60" />
          <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-cyan-500" />
        </span>
        <span class="text-[12px] font-medium text-cyan-900 dark:text-cyan-100">
          {{ pendingLabel(pendingType) }}…
        </span>
        <div class="shimmer flex-1 h-1.5 rounded-full bg-muted/40" />
      </div>
    </Transition>

    <div v-if="!blocks.length && !pendingType" class="skeleton-stack space-y-2">
      <div class="skeleton h-6 w-2/3 rounded-lg" />
      <div class="skeleton h-20 w-full rounded-xl" />
      <div class="skeleton h-4 w-1/2 rounded" />
    </div>
  </div>
</template>

<style scoped>
.shimmer {
  position: relative;
  overflow: hidden;
}
.shimmer::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(6, 182, 212, 0.35) 50%,
    transparent 100%
  );
  animation: shimmerSweep 1.3s linear infinite;
}
.skeleton {
  position: relative;
  overflow: hidden;
  background: hsl(var(--muted) / 0.45);
}
.skeleton::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(
    90deg,
    transparent 0%,
    hsl(var(--muted) / 0.75) 50%,
    transparent 100%
  );
  animation: shimmerSweep 1.4s linear infinite;
}
@keyframes shimmerSweep {
  0% {
    transform: translateX(-100%);
  }
  100% {
    transform: translateX(100%);
  }
}
</style>
