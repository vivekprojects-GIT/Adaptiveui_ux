<script setup lang="ts">
defineProps<{ block: { content?: string } }>()
</script>

<template>
  <div class="whitespace-pre-wrap leading-relaxed motion-safe:transition-opacity motion-safe:duration-300">
    {{ block.content || '' }}
  </div>
</template>
