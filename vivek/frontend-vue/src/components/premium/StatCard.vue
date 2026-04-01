<script setup lang="ts">
defineProps<{
  label: string
  value: string | number
  subtext?: string
  tone?: 'success' | 'warning' | 'danger' | 'info' | 'default'
}>()

const toneToText: Record<string, string> = {
  success: 'text-emerald-600 dark:text-emerald-300',
  warning: 'text-amber-600 dark:text-amber-300',
  danger: 'text-rose-600 dark:text-rose-300',
  info: 'text-cyan-700 dark:text-cyan-300',
  default: 'text-foreground',
}
</script>

<template>
  <div
    class="rounded-xl border bg-white/70 dark:bg-background/50 backdrop-blur p-4 shadow-sm transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md hover:border-emerald-200/70 dark:hover:border-border/80"
  >
    <div class="text-xs text-muted-foreground">{{ label }}</div>
    <div class="text-3xl font-semibold mt-2" :class="toneToText[tone || 'default']">{{ value }}</div>
    <div v-if="subtext" class="text-[11px] text-muted-foreground mt-2">{{ subtext }}</div>
    <div
      v-if="tone && tone !== 'default'"
      class="mt-3 h-1 rounded-full bg-background/60 overflow-hidden border border-border/60"
    >
      <div
        class="h-full bg-gradient-to-r opacity-90 transition-all duration-600"
        :class="{
          'from-emerald-500/90 to-emerald-300/70': tone === 'success',
          'from-amber-500/90 to-amber-300/70': tone === 'warning',
          'from-rose-500/90 to-rose-300/70': tone === 'danger',
          'from-cyan-500/90 to-cyan-300/70': tone === 'info',
        }"
        style="width: 70%"
      />
    </div>
  </div>
</template>

