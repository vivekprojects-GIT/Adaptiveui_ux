<script setup lang="ts">
import { computed } from 'vue'
import { banditState } from '@/lib/banditState'
import { sessionUserState } from '@/lib/sessionUser'
import { FEATURE_NAMES, type PosteriorMap } from '@/lib/strategies'
import { enabledStrategyIds, getStrategyLabel } from '@/lib/strategiesStore'

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
  if (!banditState.selectedStrategy) return false
  return s !== banditState.selectedStrategy
}

const activeLabel = computed(() => {
  const s = banditState.activeStrategy
  if (!s) return '—'
  return getStrategyLabel(s) || s
})
const currentUserLabel = computed(() => {
  return sessionUserState.username ? `${sessionUserState.username} posterior` : 'Your posterior'
})

const hasScores = computed(() => banditState.scores && Object.keys(banditState.scores).length > 0)
const hasFeatures = computed(() => (banditState.lastXVec?.length ?? 0) > 0)
</script>

<template>
  <section class="space-y-4">
    <div class="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-2">
      <div>
        <h2 class="text-lg font-semibold tracking-tight text-slate-900 dark:text-foreground">Bandit intelligence</h2>
        <p class="text-sm text-slate-500 dark:text-muted-foreground mt-0.5">
          Live posteriors, global prior, and reward history — same signals as the chat insights panel.
        </p>
      </div>
      <div
        class="text-xs text-slate-500 dark:text-muted-foreground rounded-full border border-violet-200/80 dark:border-violet-900/50 bg-violet-50/80 dark:bg-violet-950/30 px-3 py-1.5"
      >
        α = 0.05 global blend · synced from chat + API
      </div>
    </div>

    <!-- KPI strip (Apexify-style) -->
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-3">
      <div
        class="rounded-2xl border border-slate-200/90 dark:border-border bg-white dark:bg-card p-4 shadow-sm"
        :class="banditState.activeStrategy ? 'ring-1 ring-violet-300/70 dark:ring-violet-800/60' : ''"
      >
        <div class="text-[11px] font-medium uppercase tracking-wide text-slate-500 dark:text-muted-foreground">Active strategy</div>
        <div class="text-xl font-semibold mt-2 text-slate-900 dark:text-foreground truncate">{{ activeLabel }}</div>
        <div class="text-xs text-slate-500 dark:text-muted-foreground mt-1 line-clamp-2">
          {{ banditState.activeInstruction || 'Waiting for first message…' }}
        </div>
      </div>
      <div class="rounded-2xl border border-slate-200/90 dark:border-border bg-white dark:bg-card p-4 shadow-sm">
        <div class="text-[11px] font-medium uppercase tracking-wide text-slate-500 dark:text-muted-foreground">Global updates</div>
        <div class="text-2xl font-semibold mt-2 text-slate-900 dark:text-foreground">{{ banditState.globalN }}</div>
        <div class="text-xs text-slate-500 dark:text-muted-foreground mt-1">Reward-driven prior updates</div>
      </div>
      <div class="rounded-2xl border border-slate-200/90 dark:border-border bg-white dark:bg-card p-4 shadow-sm">
        <div class="text-[11px] font-medium uppercase tracking-wide text-slate-500 dark:text-muted-foreground">Users</div>
        <div class="text-2xl font-semibold mt-2 text-slate-900 dark:text-foreground">{{ banditState.nUsers }}</div>
        <div class="text-xs text-slate-500 dark:text-muted-foreground mt-1">Bandit users in engine</div>
      </div>
      <div class="rounded-2xl border border-slate-200/90 dark:border-border bg-white dark:bg-card p-4 shadow-sm">
        <div class="text-[11px] font-medium uppercase tracking-wide text-slate-500 dark:text-muted-foreground">Messages</div>
        <div class="text-2xl font-semibold mt-2 text-slate-900 dark:text-foreground">
          {{ banditState.msgCount == null ? '—' : banditState.msgCount }}
        </div>
        <div class="text-xs text-slate-500 dark:text-muted-foreground mt-1">Adaptive turns (this account)</div>
      </div>
    </div>

    <!-- Thompson scores -->
    <div
      v-if="hasScores"
      class="rounded-2xl border border-slate-200/90 dark:border-border bg-white dark:bg-card p-5 shadow-sm"
    >
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-sm font-semibold text-slate-900 dark:text-foreground">Thompson sampling — this turn</h3>
      </div>
      <div class="grid sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-2">
        <div v-for="s in enabledStrategyIds" :key="'ts-' + s" class="flex justify-between text-sm gap-3">
          <span class="text-slate-600 dark:text-muted-foreground truncate">{{ getStrategyLabel(s) }}</span>
          <span class="font-mono text-slate-900 dark:text-foreground tabular-nums">
            {{ banditState.scores?.[s] != null ? banditState.scores![s].toFixed(3) : '—' }}
          </span>
        </div>
      </div>
    </div>

    <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
      <!-- Your posterior -->
      <div class="rounded-2xl border border-slate-200/90 dark:border-border bg-white dark:bg-card p-5 shadow-sm">
        <h3 class="text-sm font-semibold text-slate-900 dark:text-foreground mb-1">{{ currentUserLabel }}</h3>
        <p class="text-xs text-slate-500 dark:text-muted-foreground mb-4">P(r=1 | x, a=k) — mean and uncertainty</p>
        <div class="space-y-3">
          <div v-for="s in enabledStrategyIds" :key="'u-' + s" class="group">
            <div class="flex justify-between text-xs mb-1">
              <span class="text-slate-700 dark:text-foreground/90" :class="dimRow(s) ? 'opacity-50' : ''">
                {{ getStrategyLabel(s) }}
              </span>
              <span class="font-mono font-medium text-slate-900 dark:text-foreground tabular-nums">{{ barPct(s, banditState.userPosterior) }}%</span>
            </div>
            <div class="h-2 rounded-full bg-slate-100 dark:bg-muted overflow-hidden">
              <div
                class="h-full rounded-full bg-gradient-to-r from-violet-500 to-violet-600 transition-all"
                :style="{ width: barPct(s, banditState.userPosterior) + '%' }"
              />
            </div>
            <div class="flex items-center gap-2 mt-1 text-[10px] text-slate-500">
              <span class="w-16 shrink-0">Uncertainty</span>
              <div class="flex-1 h-1.5 rounded-full bg-slate-100 dark:bg-muted overflow-hidden">
                <div class="h-full rounded-full bg-amber-400/90" :style="{ width: uncPct(s, banditState.userPosterior) + '%' }" />
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Global prior -->
      <div class="rounded-2xl border border-slate-200/90 dark:border-border bg-white dark:bg-card p-5 shadow-sm">
        <h3 class="text-sm font-semibold text-slate-900 dark:text-foreground mb-1">Global prior — all users</h3>
        <p class="text-xs text-slate-500 dark:text-muted-foreground leading-relaxed mb-4">
          Shared knowledge. Every reward feeds back here at α=0.05.
        </p>
        <div class="flex gap-8 mb-4">
          <div>
            <div class="text-2xl font-semibold text-slate-900 dark:text-foreground">{{ banditState.globalN }}</div>
            <div class="text-[10px] uppercase tracking-wide text-slate-500">updates</div>
          </div>
          <div>
            <div class="text-2xl font-semibold text-slate-900 dark:text-foreground">{{ banditState.nUsers }}</div>
            <div class="text-[10px] uppercase tracking-wide text-slate-500">users</div>
          </div>
        </div>
        <div class="space-y-3">
          <div v-for="s in enabledStrategyIds" :key="'g-' + s">
            <div class="flex justify-between text-xs mb-1">
              <span class="text-slate-700 dark:text-foreground/90">{{ getStrategyLabel(s) }}</span>
              <span class="font-mono tabular-nums text-slate-900 dark:text-foreground">{{ barPct(s, banditState.globalPosterior) }}%</span>
            </div>
            <div class="h-2 rounded-full bg-slate-100 dark:bg-muted overflow-hidden">
              <div
                class="h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-500"
                :style="{ width: barPct(s, banditState.globalPosterior) + '%' }"
              />
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Baseline synthetic user -->
    <div class="rounded-2xl border border-slate-200/90 dark:border-border bg-white dark:bg-card p-5 shadow-sm">
      <h3 class="text-sm font-semibold text-slate-900 dark:text-foreground mb-1">Baseline user — inheriting prior</h3>
      <p class="text-xs text-slate-500 dark:text-muted-foreground mb-4 max-w-3xl">
        Synthetic fresh-session baseline that tracks the shared global posterior. Bars move when the global prior changes after real reward updates.
      </p>
      <div class="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        <div v-for="s in enabledStrategyIds" :key="'b-' + s">
          <div class="flex justify-between text-xs mb-1">
            <span class="text-slate-700 dark:text-foreground/90 truncate pr-2">{{ getStrategyLabel(s) }}</span>
            <span class="font-mono tabular-nums shrink-0">{{ barPct(s, banditState.userBPosterior) }}%</span>
          </div>
          <div class="h-2 rounded-full bg-slate-100 dark:bg-muted overflow-hidden">
            <div
              class="h-full rounded-full bg-gradient-to-r from-fuchsia-500 to-violet-600"
              :style="{ width: barPct(s, banditState.userBPosterior) + '%' }"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- Feature vector -->
    <div
      v-if="hasFeatures"
      class="rounded-2xl border border-slate-200/90 dark:border-border bg-white dark:bg-card p-5 shadow-sm"
    >
      <h3 class="text-sm font-semibold text-slate-900 dark:text-foreground mb-4">Feature vector x ∈ ℝ¹⁰</h3>
      <div class="grid sm:grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-2 text-sm">
        <template v-for="(name, i) in FEATURE_NAMES" :key="name">
          <div class="flex justify-between gap-4 border-b border-slate-100 dark:border-border/60 py-1.5 last:border-0">
            <span class="text-slate-600 dark:text-muted-foreground font-mono text-xs">{{ name }}</span>
            <span class="font-mono text-slate-900 dark:text-foreground tabular-nums">
              {{ banditState.lastXVec?.[i] != null ? (banditState.lastXVec![i] as number).toFixed(3) : '—' }}
            </span>
          </div>
        </template>
      </div>
    </div>

    <!-- Reward log table -->
    <div class="rounded-2xl border border-slate-200/90 dark:border-border bg-white dark:bg-card shadow-sm overflow-hidden">
      <div class="px-5 py-4 border-b border-slate-100 dark:border-border flex items-center justify-between">
        <div>
          <h3 class="text-sm font-semibold text-slate-900 dark:text-foreground">Reward log</h3>
          <p class="text-xs text-slate-500 dark:text-muted-foreground mt-0.5">Auto + manual feedback from chat</p>
        </div>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left border-b border-slate-100 dark:border-border bg-slate-50/80 dark:bg-muted/30">
              <th class="py-3 px-5 font-medium text-slate-500 dark:text-muted-foreground">Strategy</th>
              <th class="py-3 px-5 font-medium text-slate-500 dark:text-muted-foreground">Reward</th>
              <th class="py-3 px-5 font-medium text-slate-500 dark:text-muted-foreground">Detail</th>
              <th class="py-3 px-5 font-medium text-slate-500 dark:text-muted-foreground">Source</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(e, i) in banditState.rewardLog"
              :key="i"
              class="border-b border-slate-100 dark:border-border/60 last:border-0 hover:bg-slate-50/50 dark:hover:bg-muted/20"
            >
              <td class="py-3 px-5 font-medium text-slate-900 dark:text-foreground">
                {{ getStrategyLabel(e.strategy) }}
              </td>
              <td class="py-3 px-5">
                <span
                  class="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium"
                  :class="e.reward >= 0.5 ? 'bg-emerald-50 text-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300' : 'bg-red-50 text-red-800 dark:bg-red-950/40 dark:text-red-300'"
                >
                  <span class="h-1.5 w-1.5 rounded-full" :class="e.reward >= 0.5 ? 'bg-emerald-500' : 'bg-red-500'" />
                  {{ e.reward.toFixed(2) }}
                </span>
              </td>
              <td class="py-3 px-5 text-slate-600 dark:text-muted-foreground max-w-md truncate">{{ e.detail }}</td>
              <td class="py-3 px-5 text-slate-500 dark:text-muted-foreground text-xs capitalize">{{ e.source }}</td>
            </tr>
            <tr v-if="banditState.rewardLog.length === 0">
              <td class="py-10 px-5 text-center text-slate-500 dark:text-muted-foreground" colspan="4">
                No reward events yet. Send messages and use helpful / not helpful in chat.
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>
</template>
