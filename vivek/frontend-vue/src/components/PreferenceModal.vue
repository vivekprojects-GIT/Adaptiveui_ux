<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import Button from '@/components/ui/Button.vue'
import { enabledStrategies, instructionPreview } from '@/lib/strategiesStore'

const props = defineProps<{
  open: boolean
}>()

const emit = defineEmits<{
  (e: 'update:open', v: boolean): void
  (e: 'submit', payload: { strategies: string[]; lock: boolean }): void
  (e: 'skip'): void
}>()

const lockOne = ref(false)
const selected = ref<Set<string>>(new Set())

watch(
  () => props.open,
  (v) => {
    if (v) {
      selected.value = new Set()
      lockOne.value = false
    }
  },
)

const items = computed(() =>
  enabledStrategies.value.map((s) => ({
    id: s.id,
    label: s.label,
    hint: instructionPreview(s.instruction, 92) || '—',
    wide: s.id === 'step_by_step' || s.id === 'comparison_table',
  })),
)

function toggle(id: string) {
  if (lockOne.value) {
    selected.value = new Set([id])
  } else {
    const next = new Set(selected.value)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    selected.value = next
  }
}

function isSel(id: string) {
  return selected.value.has(id)
}

function onSkip() {
  emit('update:open', false)
  emit('skip')
}

function onApply() {
  emit('update:open', false)
  emit('submit', { strategies: [...selected.value], lock: lockOne.value })
}
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="fixed inset-0 z-[200] flex items-center justify-center bg-background/80 backdrop-blur-sm px-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="pref-title"
    >
      <div
        class="w-full max-w-lg rounded-2xl border bg-card p-6 shadow-xl space-y-4 max-h-[90vh] overflow-y-auto"
        @click.stop
      >
        <h2 id="pref-title" class="text-base font-semibold tracking-tight">
          How do you like responses? <span class="opacity-40">✦</span>
        </h2>
        <p class="text-sm text-muted-foreground leading-relaxed">
          Pick one or more styles — the engine will warm-start your posterior from these before your first message. You can
          skip and let it learn from scratch instead.
        </p>

        <div class="grid grid-cols-2 gap-2">
          <button
            v-for="it in items"
            :key="it.id"
            type="button"
            class="rounded-xl border px-3 py-2.5 text-left transition-colors hover:border-primary/50 hover:bg-accent/30"
            :class="[
              isSel(it.id) ? 'border-primary bg-accent/40' : 'border-border bg-transparent',
              it.wide ? 'col-span-2' : '',
            ]"
            @click="toggle(it.id)"
          >
            <div class="text-[11px] font-semibold uppercase tracking-wide">{{ it.label }}</div>
            <div class="text-[11px] text-muted-foreground mt-0.5 leading-snug">{{ it.hint }}</div>
          </button>
        </div>

        <label class="flex items-center gap-2 cursor-pointer select-none">
          <input v-model="lockOne" type="checkbox" class="rounded border-input" />
          <span class="text-[10px] text-muted-foreground">Lock to ONE style (turns off exploration)</span>
        </label>

        <div class="flex justify-end gap-2 pt-2">
          <Button type="button" variant="outline" class="text-xs h-9" @click="onSkip">Skip — learn from scratch</Button>
          <Button type="button" class="h-9 text-xs" :disabled="selected.size === 0" @click="onApply">Apply preferences →</Button>
        </div>
      </div>
    </div>
  </Teleport>
</template>
