<script setup lang="ts">
import StatCard from '@/components/premium/StatCard.vue'

type Tone = 'success' | 'warning' | 'danger' | 'info' | 'default'
defineProps<{
  block: { items?: { label?: string; value?: string | number; subtext?: string; tone?: Tone }[] }
}>()
</script>

<template>
  <div class="grid grid-cols-2 sm:grid-cols-3 gap-3">
    <StatCard
      v-for="(it, i) in block.items || []"
      :key="i"
      :label="it.label || ''"
      :value="it.value ?? ''"
      :subtext="it.subtext"
      :tone="it.tone || 'default'"
    />
  </div>
</template>
