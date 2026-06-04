<script setup lang="ts">
import Badge from '@/components/premium/Badge.vue'

type Tone = 'default' | 'success' | 'warning' | 'danger' | 'info'
defineProps<{ block: { items?: { label?: string; tone?: Tone }[] } }>()
</script>

<template>
  <div class="flex flex-wrap gap-2">
    <Badge v-for="(it, i) in block.items || []" :key="i" :tone="it.tone || 'default'">
      {{ it.label }}
    </Badge>
  </div>
</template>
