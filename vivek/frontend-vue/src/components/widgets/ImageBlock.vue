<script setup lang="ts">
defineProps<{ block: { title?: string; src?: string; alt?: string; caption?: string; fit?: string } }>()
</script>

<template>
  <div class="rounded-xl border bg-card overflow-hidden transition-shadow duration-300 hover:shadow-md">
    <div v-if="block.title" class="px-3 py-2 border-b text-xs font-medium">{{ block.title }}</div>
    <div v-if="block.src" class="p-2 flex justify-center bg-muted/15">
      <img
        :src="block.src"
        :alt="block.alt || 'Widget image'"
        class="max-w-full rounded-lg"
        :class="
          block.fit === 'cover'
            ? 'object-cover w-full max-h-[min(420px,55vh)]'
            : 'object-contain max-h-[min(420px,55vh)] h-auto'
        "
        loading="lazy"
        referrerpolicy="no-referrer"
      />
    </div>
    <div v-else class="px-3 py-2 text-xs text-muted-foreground">No image URL in block.</div>
    <div v-if="block.caption" class="px-3 py-2 text-[11px] text-muted-foreground border-t">
      {{ block.caption }}
    </div>
  </div>
</template>
