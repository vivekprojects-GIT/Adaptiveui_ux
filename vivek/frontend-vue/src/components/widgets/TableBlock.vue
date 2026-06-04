<script setup lang="ts">
defineProps<{ block: { title?: string; columns?: string[]; rows?: unknown[][] } }>()
</script>

<template>
  <div class="rounded-xl border bg-card overflow-hidden transition-shadow duration-300 hover:shadow-md">
    <div v-if="block.title" class="px-3 py-2 border-b text-xs font-medium">{{ block.title }}</div>
    <div class="overflow-x-auto">
      <table class="w-full text-xs">
        <thead v-if="block.columns?.length">
          <tr>
            <th
              v-for="(c, ci) in block.columns"
              :key="ci"
              class="text-left px-3 py-2 border-b bg-muted/40 font-medium"
            >
              {{ c }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, ri) in block.rows || []" :key="ri">
            <td v-for="(cell, ci) in row" :key="ci" class="px-3 py-1.5 border-b border-border/60">
              {{ cell }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
