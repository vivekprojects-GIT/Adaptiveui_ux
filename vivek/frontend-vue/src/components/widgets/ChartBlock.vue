<script setup lang="ts">
import WidgetSchemaChart from '@/components/WidgetSchemaChart.vue'

const props = defineProps<{ block: { title?: string; chart?: Record<string, unknown> } }>()
</script>

<template>
  <WidgetSchemaChart :title="props.block.title" :chart="(props.block.chart as any) || {}" />
</template>
