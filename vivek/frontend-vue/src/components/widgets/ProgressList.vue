<script setup lang="ts">
import ProgressBar from '@/components/premium/ProgressBar.vue'

type Tone = 'emerald' | 'amber' | 'red' | 'cyan' | 'indigo'
defineProps<{
  block: { items?: { label?: string; value?: number; max?: number; tone?: Tone }[] }
}>()
</script>

<template>
  <div class="space-y-3">
    <div v-for="(it, i) in block.items || []" :key="i">
      <div class="flex items-center justify-between text-xs mb-1">
        <span class="text-muted-foreground">{{ it.label }}</span>
        <span class="font-medium">{{ it.value }}{{ !it.max || it.max === 100 ? '%' : '' }}</span>
      </div>
      <ProgressBar :value="Number(it.value) || 0" :max="it.max ?? 100" :tone="it.tone || 'indigo'" />
    </div>
  </div>
</template>
