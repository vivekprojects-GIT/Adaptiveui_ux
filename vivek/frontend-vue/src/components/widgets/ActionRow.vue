<script setup lang="ts">
import Button from '@/components/ui/Button.vue'

defineProps<{ block: { buttons?: { label?: string; intent?: string }[] } }>()
</script>

<template>
  <div class="flex flex-wrap gap-2">
    <Button
      v-for="(b, bi) in block.buttons || []"
      :key="bi"
      type="button"
      variant="outline"
      size="sm"
      disabled
      :title="b.intent ? `Intent: ${b.intent}` : undefined"
    >
      {{ b.label || 'Action' }}
    </Button>
  </div>
</template>
