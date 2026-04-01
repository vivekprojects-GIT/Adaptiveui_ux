<script setup lang="ts">
import { computed } from 'vue'
import { toasts, dismissToast } from '@/lib/toast'

const items = computed(() => toasts.items)
</script>

<template>
  <div class="fixed top-16 right-4 z-[100] w-[360px] max-w-[calc(100vw-2rem)] space-y-2">
    <div
      v-for="t in items"
      :key="t.id"
      class="rounded-xl border bg-card px-4 py-3 shadow-sm"
    >
      <div class="flex items-start justify-between gap-3">
        <div class="min-w-0">
          <div class="text-sm font-medium">
            {{ t.title }}
          </div>
          <div v-if="t.message" class="text-sm text-muted-foreground mt-0.5 whitespace-pre-wrap break-words">
            {{ t.message }}
          </div>
        </div>
        <button
          class="h-7 w-7 rounded-lg border hover:bg-accent/10 text-muted-foreground"
          @click="dismissToast(t.id)"
          aria-label="Dismiss"
        >
          ✕
        </button>
      </div>
    </div>
  </div>
</template>

