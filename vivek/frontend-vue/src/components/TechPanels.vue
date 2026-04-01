<script setup lang="ts">
import { computed } from 'vue'
import { FEATURE_NAMES, type PosteriorMap } from '@/lib/strategies'
import { enabledStrategyIds, getStrategyLabel } from '@/lib/strategiesStore'

const props = defineProps<{
  activeStrategy: string
  activeInstruction: string
  selectedStrategy: string
  userPosterior: PosteriorMap
  globalPosterior: PosteriorMap
  userBPosterior: PosteriorMap
  globalN: number
  nUsers: number
  scores: Record<string, number> | null
  xVec: number[] | null
  rewardLog: { strategy: string; reward: number; detail: string; source: string }[]
}>()

const hasFeatures = computed(() => (props.xVec?.length ?? 0) > 0)
const hasScores = computed(() => props.scores && Object.keys(props.scores).length > 0)
const displayStrategyIds = computed(() => {
  if (enabledStrategyIds.value.length > 0) return enabledStrategyIds.value
  const keys = new Set<string>([
    ...Object.keys(props.userPosterior || {}),
    ...Object.keys(props.globalPosterior || {}),
    ...Object.keys(props.userBPosterior || {}),
  ])
  return Array.from(keys)
})

function barPct(s: string, data: PosteriorMap) {
  const d = data[s]
  if (!d) return 0
  return Math.round((d.r ?? 0) * 100)
}

function uncPct(s: string, data: PosteriorMap) {
  const d = data[s]
  if (!d) return 50
  return Math.min((d.u ?? 0) / 8, 1) * 100
}

function dimRow(s: string) {
  if (!props.selectedStrategy) return false
  return s !== props.selectedStrategy
}
</script>

<template>
  <div class="space-y-4">
    <details class="rounded-xl border bg-card open:shadow-sm" open>
      <summary class="cursor-pointer px-4 py-3 text-xs font-semibold uppercase tracking-wide">Active strategy</summary>
      <div class="px-4 pb-4 space-y-2 border-t">
        <div class="text-sm font-medium mt-3">{{ activeStrategy ? getStrategyLabel(activeStrategy) : '—' }}</div>
        <p class="text-[11px] text-muted-foreground leading-relaxed">{{ activeInstruction || 'Waiting for first message…' }}</p>
      </div>
    </details>

    <details v-if="hasScores" class="rounded-xl border bg-card open:shadow-sm" open>
      <summary class="cursor-pointer px-4 py-3 text-xs font-semibold uppercase tracking-wide">Thompson sampling — this turn</summary>
      <div class="px-4 pb-4 border-t space-y-1.5 mt-2">
        <div v-for="s in displayStrategyIds" :key="s" class="flex justify-between text-[11px]">
          <span class="text-muted-foreground">{{ getStrategyLabel(s) }}</span>
          <span class="font-mono">{{ scores?.[s] != null ? scores![s].toFixed(3) : '—' }}</span>
        </div>
      </div>
    </details>

    <details class="rounded-xl border bg-card open:shadow-sm" open>
      <summary class="cursor-pointer px-4 py-3 text-xs font-semibold uppercase tracking-wide">Your posterior</summary>
      <div class="px-4 pb-4 border-t space-y-2 mt-3">
        <p v-if="displayStrategyIds.length === 0" class="text-[11px] text-muted-foreground">No strategy bars yet.</p>
        <div v-for="s in displayStrategyIds" :key="s" class="flex flex-col gap-0.5">
          <div class="flex justify-between text-[10px]">
            <span :class="dimRow(s) ? 'opacity-45' : ''">{{ getStrategyLabel(s) }}</span>
            <span class="font-mono">{{ barPct(s, userPosterior) }}%</span>
          </div>
          <div class="h-1.5 rounded-full bg-muted overflow-hidden">
            <div class="h-full rounded-full bg-primary transition-all" :style="{ width: barPct(s, userPosterior) + '%' }" />
          </div>
          <div class="flex items-center gap-1 text-[9px] text-muted-foreground">
            <span>uncertainty</span>
            <div class="flex-1 h-1 rounded bg-muted overflow-hidden">
              <div class="h-full bg-amber-500/70" :style="{ width: uncPct(s, userPosterior) + '%' }" />
            </div>
          </div>
        </div>
      </div>
    </details>

    <details class="rounded-xl border bg-card open:shadow-sm" open>
      <summary class="cursor-pointer px-4 py-3 text-xs font-semibold uppercase tracking-wide">Global prior — all users</summary>
      <div class="px-4 pb-4 border-t space-y-3 mt-3">
        <p class="text-[11px] text-muted-foreground leading-relaxed">
          Shared knowledge. Every reward feeds back here at α=0.05.
        </p>
        <div class="flex gap-4">
          <div>
            <div class="text-xl font-semibold">{{ globalN }}</div>
            <div class="text-[10px] text-muted-foreground uppercase">updates</div>
          </div>
          <div>
            <div class="text-xl font-semibold">{{ nUsers }}</div>
            <div class="text-[10px] text-muted-foreground uppercase">users</div>
          </div>
        </div>
        <p v-if="displayStrategyIds.length === 0" class="text-[11px] text-muted-foreground">No strategy bars yet.</p>
        <div v-for="s in displayStrategyIds" :key="'g-' + s" class="flex flex-col gap-0.5">
          <div class="flex justify-between text-[10px]">
            <span>{{ getStrategyLabel(s) }}</span>
            <span class="font-mono">{{ barPct(s, globalPosterior) }}%</span>
          </div>
          <div class="h-1.5 rounded-full bg-muted overflow-hidden">
            <div class="h-full rounded-full bg-emerald-600/80" :style="{ width: barPct(s, globalPosterior) + '%' }" />
          </div>
        </div>
      </div>
    </details>

    <details class="rounded-xl border bg-card open:shadow-sm">
      <summary class="cursor-pointer px-4 py-3 text-xs font-semibold uppercase tracking-wide">User B — inheriting prior</summary>
      <div class="px-4 pb-4 border-t space-y-2 mt-3">
        <p class="text-[11px] text-muted-foreground leading-relaxed">
          Fresh session baseline. Bars move when the global prior changes after reward updates.
        </p>
        <p v-if="displayStrategyIds.length === 0" class="text-[11px] text-muted-foreground">No strategy bars yet.</p>
        <div v-for="s in displayStrategyIds" :key="'b-' + s" class="flex flex-col gap-0.5">
          <div class="flex justify-between text-[10px]">
            <span>{{ getStrategyLabel(s) }}</span>
            <span class="font-mono">{{ barPct(s, userBPosterior) }}%</span>
          </div>
          <div class="h-1.5 rounded-full bg-muted overflow-hidden">
            <div class="h-full rounded-full bg-violet-600/70" :style="{ width: barPct(s, userBPosterior) + '%' }" />
          </div>
        </div>
      </div>
    </details>

    <details v-if="hasFeatures" class="rounded-xl border bg-card open:shadow-sm" open>
      <summary class="cursor-pointer px-4 py-3 text-xs font-semibold uppercase tracking-wide">Feature vector x ∈ ℝ¹⁰</summary>
      <div class="px-4 pb-4 border-t grid grid-cols-2 gap-x-4 gap-y-1 mt-3 text-[11px]">
        <template v-for="(name, i) in FEATURE_NAMES" :key="name">
          <span class="text-muted-foreground">{{ name }}</span>
          <span class="font-mono text-right">{{ xVec?.[i] != null ? (xVec![i] as number).toFixed(3) : '—' }}</span>
        </template>
      </div>
    </details>

    <details class="rounded-xl border bg-card open:shadow-sm">
      <summary class="cursor-pointer px-4 py-3 text-xs font-semibold uppercase tracking-wide">Reward log</summary>
      <div class="px-4 pb-4 border-t mt-2 max-h-48 overflow-y-auto">
        <p v-if="!rewardLog.length" class="text-[11px] text-muted-foreground text-center py-6">No interactions yet</p>
        <div v-for="(e, i) in rewardLog" :key="i" class="text-[11px] border-b border-border/50 py-2 last:border-0 flex flex-wrap gap-2">
          <span class="font-medium">{{ getStrategyLabel(e.strategy) }}</span>
          <span :class="e.reward < 0.5 ? 'text-red-600' : 'text-emerald-600'">r={{ e.reward.toFixed(2) }}</span>
          <span class="text-muted-foreground">{{ e.detail }}</span>
          <span class="text-muted-foreground/80 text-[10px]">({{ e.source }})</span>
        </div>
      </div>
    </details>
  </div>
</template>
