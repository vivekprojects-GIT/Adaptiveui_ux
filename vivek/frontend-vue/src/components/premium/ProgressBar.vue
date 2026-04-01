<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    value: number
    max?: number
    tone?: 'emerald' | 'amber' | 'red' | 'cyan' | 'indigo'
  }>(),
  { max: 1, tone: 'indigo' },
)

const percent = Math.max(0, Math.min(100, (props.value / props.max) * 100))

const toneGrad: Record<string, string> = {
  emerald: 'from-emerald-500 to-emerald-300',
  amber: 'from-amber-500 to-amber-300',
  red: 'from-red-500 to-rose-300',
  cyan: 'from-cyan-500 to-cyan-300',
  indigo: 'from-indigo-500 to-cyan-400',
}
</script>

<template>
  <div class="w-full h-2 rounded-full bg-white/60 dark:bg-background/60 border border-border/60 overflow-hidden">
    <div
      class="h-full rounded-full bg-gradient-to-r opacity-95 transition-all duration-600"
      :class="`bg-gradient-to-r ${toneGrad[tone]}`"
      :style="{ width: `${percent}%` }"
    />
  </div>
</template>
