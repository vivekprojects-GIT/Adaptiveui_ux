<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { buildProgressiveHtmlDoc } from '@/lib/progressiveWidget'

/**
 * Progressive iframe renderer for HTML-mode widgets. While content streams,
 * srcdoc is updated at a throttled rate (~120ms) so the browser re-renders
 * without flickering. When `finalized` goes true, we swap to the polished
 * final HTML so design-system injection + truncation fixes take effect.
 */
const props = defineProps<{
  rawStream: string
  finalHtml?: string
  finalized: boolean
  height?: number
}>()

const throttledDoc = ref('')
let lastWrite = 0
let pendingTimer: ReturnType<typeof setTimeout> | null = null

function scheduleWrite(next: string) {
  const now = Date.now()
  const THROTTLE_MS = 120
  if (now - lastWrite >= THROTTLE_MS) {
    throttledDoc.value = next
    lastWrite = now
    return
  }
  if (pendingTimer) clearTimeout(pendingTimer)
  pendingTimer = setTimeout(() => {
    throttledDoc.value = next
    lastWrite = Date.now()
    pendingTimer = null
  }, THROTTLE_MS - (now - lastWrite))
}

watch(
  () => [props.rawStream, props.finalized, props.finalHtml],
  () => {
    if (props.finalized && props.finalHtml) {
      throttledDoc.value = props.finalHtml
      return
    }
    const doc = buildProgressiveHtmlDoc(props.rawStream)
    if (doc) scheduleWrite(doc)
  },
  { immediate: true },
)

const frameHeight = computed(() => {
  const raw = Number(props.height || 420)
  if (!Number.isFinite(raw)) return 420
  return Math.min(Math.max(raw, 300), 520)
})
</script>

<template>
  <div class="relative w-full">
    <iframe
      v-if="throttledDoc"
      :srcdoc="throttledDoc"
      sandbox="allow-scripts allow-same-origin"
      class="w-full widget-frame border-0"
      :style="{ height: `${frameHeight}px`, maxHeight: '56vh' }"
    />
    <div
      v-else
      class="w-full rounded-xl border bg-muted/20 skeleton"
      :style="{ height: `${frameHeight}px`, maxHeight: '56vh' }"
    />
    <div
      v-if="!finalized"
      class="absolute top-2 right-2 text-[10px] px-2 py-1 rounded-full bg-cyan-500/90 text-white shadow-md flex items-center gap-1.5"
    >
      <span class="relative flex h-1.5 w-1.5">
        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75" />
        <span class="relative inline-flex rounded-full h-1.5 w-1.5 bg-white" />
      </span>
      streaming
    </div>
  </div>
</template>

<style scoped>
.skeleton {
  position: relative;
  overflow: hidden;
  background: hsl(var(--muted) / 0.3);
}
.skeleton::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(
    90deg,
    transparent 0%,
    hsl(var(--muted) / 0.6) 50%,
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
