<script setup lang="ts">
import Button from '@/components/ui/Button.vue'

defineProps<{ block: { buttons?: { label?: string; intent?: string }[] } }>()
const emit = defineEmits<{ (e: 'action', text: string): void }>()

function click(b: { label?: string; intent?: string }) {
  // Send the human-readable label as the follow-up prompt (fall back to the intent code).
  const text = String(b.label || b.intent || '').trim()
  if (text) emit('action', text)
}
</script>

<template>
  <div class="flex flex-wrap gap-2">
    <Button
      v-for="(b, bi) in block.buttons || []"
      :key="bi"
      type="button"
      variant="outline"
      size="sm"
      :title="b.intent ? `Ask: ${b.label || b.intent}` : undefined"
      @click="click(b)"
    >
      {{ b.label || 'Action' }}
    </Button>
  </div>
</template>
