<script setup lang="ts">
defineProps<{ block: { items?: { label?: string; value?: string; tone?: string }[] } }>()
</script>

<template>
  <div class="grid grid-cols-2 sm:grid-cols-3 gap-2">
    <div
      v-for="(it, j) in block.items || []"
      :key="j"
      class="rounded-xl border bg-card px-3 py-2 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md"
    >
      <div class="text-[10px] text-muted-foreground uppercase tracking-wide">{{ it.label }}</div>
      <div
        class="text-lg font-semibold mt-0.5"
        :class="{
          'text-emerald-600 dark:text-emerald-400': it.tone === 'positive',
          'text-red-600 dark:text-red-400': it.tone === 'negative',
        }"
      >
        {{ it.value }}
      </div>
    </div>
  </div>
</template>
