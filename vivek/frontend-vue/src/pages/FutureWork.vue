<script setup lang="ts">
import { computed, ref } from 'vue'

const open = ref<Record<number, boolean>>({ 0: true })

function toggle(i: number) {
  open.value[i] = !open.value[i]
}

const roadmap = [
  { label: 'Adaptive memory graph', progress: 82 },
  { label: 'Causal strategy debugger', progress: 68 },
  { label: 'Confidence-calibrated widgets', progress: 74 },
] as const

const maxProgress = computed(() => Math.max(...roadmap.map((x) => x.progress)))
</script>

<template>
  <div class="max-w-5xl mx-auto pb-10 page-shell page-shell-future">
    <div class="space-y-8 p-5 lg:p-8 page-content">
    <div class="space-y-2">
      <div class="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">Adaptive Presentation Engine</div>
      <h1 class="text-2xl font-semibold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500">Cognitive Mental Model</h1>
      <p class="text-sm text-muted-foreground leading-relaxed">
        A living portrait — not a static persona — constructed through revealed preferences and continuously refined by
        Bayesian inference.
      </p>
    </div>

    <div class="rounded-2xl border premium-card p-5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 hover-lift">
      <div>
        <div class="font-semibold">Margaret Chen</div>
        <div class="text-xs text-muted-foreground mt-0.5">VG-0847291 · 4 months · 47 interactions</div>
      </div>
      <div class="flex gap-6">
        <div>
          <div class="text-xl font-semibold text-blue-600 dark:text-blue-400">75%</div>
          <div class="text-[10px] text-muted-foreground uppercase">Confidence</div>
        </div>
        <div>
          <div class="text-xl font-semibold text-emerald-600 dark:text-emerald-400">5</div>
          <div class="text-[10px] text-muted-foreground uppercase">Facets</div>
        </div>
        <div>
          <div class="text-xl font-semibold text-amber-600 dark:text-amber-400">3</div>
          <div class="text-[10px] text-muted-foreground uppercase">High fidelity</div>
        </div>
      </div>
    </div>

    <section class="rounded-2xl border premium-card p-5 space-y-3 hover-lift">
      <div class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Portrait summary</div>
      <p class="text-sm leading-relaxed text-muted-foreground">
        Margaret processes financial decisions through
        <span class="text-foreground font-medium">causal narratives</span>
        when stakes involve action (rebalancing, volatility response) but switches to
        <span class="text-foreground font-medium">structured elimination tools</span>
        when exploring options. She seeks
        <span class="text-foreground font-medium">historical anchoring</span>
        during market stress rather than forward projections. Tax-related and retirement planning facets are still
        developing — the system is actively exploring presentation strategies in these contexts.
      </p>
    </section>

    <section class="rounded-2xl border premium-card p-5 space-y-3">
      <div class="flex items-center justify-between">
        <h2 class="text-sm font-semibold">Roadmap momentum</h2>
        <span class="text-[10px] uppercase tracking-wide text-muted-foreground">animated signal</span>
      </div>
      <div class="space-y-3">
        <div v-for="item in roadmap" :key="item.label" class="space-y-1">
          <div class="flex items-center justify-between text-xs">
            <span class="text-foreground/90">{{ item.label }}</span>
            <span class="font-mono text-muted-foreground">{{ item.progress }}%</span>
          </div>
          <div class="h-2 rounded-full bg-muted/60 overflow-hidden">
            <div
              class="h-full rounded-full progress-bar"
              :style="{ width: `${item.progress}%`, opacity: item.progress / maxProgress }"
            />
          </div>
        </div>
      </div>
      <div class="future-line mt-2" />
    </section>

    <div class="flex items-baseline justify-between gap-4">
      <h2 class="text-sm font-semibold">Cognitive facets</h2>
      <span class="text-[10px] text-muted-foreground">sorted by confidence</span>
    </div>

    <div class="space-y-3">
      <div
        class="rounded-xl border overflow-hidden cursor-pointer transition-colors hover:bg-accent/20"
        :class="open[0] ? 'ring-1 ring-primary/30' : ''"
        @click="toggle(0)"
      >
        <div class="px-4 py-3 flex items-center gap-3">
          <div class="h-12 w-12 rounded-full border-2 border-blue-500/50 flex items-center justify-center text-sm font-bold text-blue-600">
            88
          </div>
          <div class="flex-1 min-w-0">
            <div class="font-medium">Rebalancing Decisions</div>
            <div class="text-[11px] text-muted-foreground">14 interactions · <span class="text-emerald-600">High</span></div>
          </div>
          <div class="text-right text-[11px] hidden sm:block">
            <div>Scenario Narrative</div>
            <div class="text-muted-foreground">μ = 0.85</div>
          </div>
          <span class="text-muted-foreground">{{ open[0] ? '▾' : '▸' }}</span>
        </div>
        <div v-if="open[0]" class="px-4 pb-4 pt-0 border-t text-xs text-muted-foreground space-y-2">
          <p>
            Responds best to scenario-based framing that shows consequences. Engagement drops when presented with abstract
            risk tables alone.
          </p>
          <p class="text-[10px]">Evidence: completed portfolio rebalance after scenario walkthrough on multiple occasions.</p>
        </div>
      </div>

      <div
        class="rounded-xl border overflow-hidden cursor-pointer transition-colors hover:bg-accent/20"
        @click="toggle(1)"
      >
        <div class="px-4 py-3 flex items-center gap-3">
          <div class="h-12 w-12 rounded-full border-2 border-emerald-500/50 flex items-center justify-center text-sm font-bold text-emerald-600">
            79
          </div>
          <div class="flex-1 min-w-0">
            <div class="font-medium">Fund Exploration</div>
            <div class="text-[11px] text-muted-foreground">11 interactions · Moderate</div>
          </div>
          <div class="text-right text-[11px] hidden sm:block">
            <div>Elimination Matrix</div>
            <div class="text-muted-foreground">μ = 0.80</div>
          </div>
          <span class="text-muted-foreground">{{ open[1] ? '▾' : '▸' }}</span>
        </div>
        <div v-if="open[1]" class="px-4 pb-4 pt-0 border-t text-xs text-muted-foreground space-y-2">
          <p>Engages deeply with structured narrowing tools. Prefers to reduce options before evaluating.</p>
        </div>
      </div>

      <div
        class="rounded-xl border overflow-hidden cursor-pointer transition-colors hover:bg-accent/20"
        @click="toggle(2)"
      >
        <div class="px-4 py-3 flex items-center gap-3">
          <div class="h-12 w-12 rounded-full border-2 border-amber-500/50 flex items-center justify-center text-sm font-bold text-amber-600">
            72
          </div>
          <div class="flex-1 min-w-0">
            <div class="font-medium">Market Volatility Response</div>
            <div class="text-[11px] text-muted-foreground">8 interactions · Moderate</div>
          </div>
          <div class="text-right text-[11px] hidden sm:block">
            <div>Historical Precedent</div>
            <div class="text-muted-foreground">μ = 0.75</div>
          </div>
          <span class="text-muted-foreground">{{ open[2] ? '▾' : '▸' }}</span>
        </div>
        <div v-if="open[2]" class="px-4 pb-4 pt-0 border-t text-xs text-muted-foreground space-y-2">
          <p>During volatility, seeks anchoring in past outcomes rather than forward projections.</p>
        </div>
      </div>
    </div>

    <p class="text-[11px] text-muted-foreground leading-relaxed border-t pt-6">
      This model is a posterior, not a profile. High-fidelity facets reflect strong convergence across multiple
      interactions; emerging facets are still being shaped. Every interaction updates the weights; the portrait is never
      frozen.
    </p>
    </div>
  </div>
</template>

<style scoped>
.progress-bar {
  background: linear-gradient(90deg, rgba(34, 211, 238, 0.95), rgba(52, 211, 153, 0.9));
  transform-origin: left center;
  animation: growBar 900ms cubic-bezier(0.22, 1, 0.36, 1) both;
}

.future-line {
  height: 44px;
  border-radius: 999px;
  background:
    radial-gradient(circle at 22% 50%, rgba(34, 211, 238, 0.5), transparent 28%),
    radial-gradient(circle at 58% 50%, rgba(52, 211, 153, 0.45), transparent 30%),
    linear-gradient(90deg, rgba(34, 211, 238, 0.18), rgba(52, 211, 153, 0.2));
  animation: waveGlow 2.8s ease-in-out infinite;
}

@keyframes growBar {
  from {
    transform: scaleX(0.2);
    filter: saturate(0.7);
  }
  to {
    transform: scaleX(1);
    filter: saturate(1);
  }
}

@keyframes waveGlow {
  0%,
  100% {
    box-shadow: 0 0 0 rgba(34, 211, 238, 0);
    background-position: 0% 50%;
  }
  50% {
    box-shadow: 0 0 32px rgba(34, 211, 238, 0.16);
    background-position: 100% 50%;
  }
}
</style>
