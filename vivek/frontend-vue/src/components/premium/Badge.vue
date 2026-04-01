<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    tone?: 'default' | 'success' | 'warning' | 'danger' | 'info'
  }>(),
  {
    tone: 'default',
  },
)

const toneClass: Record<string, string> = {
  default: 'text-foreground/90 border-border/70 bg-white/70 dark:bg-background/50',
  success: 'text-emerald-700 dark:text-emerald-300 border-emerald-400/35 bg-emerald-500/10',
  warning: 'text-amber-700 dark:text-amber-300 border-amber-400/35 bg-amber-500/10',
  danger: 'text-rose-700 dark:text-rose-300 border-rose-400/35 bg-rose-500/10',
  info: 'text-cyan-700 dark:text-cyan-300 border-cyan-400/35 bg-cyan-500/10',
}
</script>

<template>
  <span
    class="inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium tracking-wide uppercase"
    :class="toneClass[props.tone]"
  >
    <slot />
  </span>
</template>
