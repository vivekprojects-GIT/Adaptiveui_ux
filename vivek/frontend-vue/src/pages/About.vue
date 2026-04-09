<script setup lang="ts">
import Card from '@/components/ui/Card.vue'
</script>

<template>
  <div class="max-w-5xl mx-auto pb-10 page-shell page-shell-about">
    <article class="space-y-8 p-5 lg:p-8 page-content">
      <header class="space-y-3">
        <div class="text-[11px] uppercase tracking-[0.24em] text-muted-foreground">Prism Journal</div>
        <h1 class="text-3xl lg:text-4xl font-semibold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500 text-balance">
          Building AI Answers People Can Read, Trust, and Act On
        </h1>
        <p class="text-base text-muted-foreground leading-relaxed max-w-3xl">
          Prism is an adaptive response engine focused on one outcome: reduce cognitive overload while improving response
          quality. Instead of forcing one output style for every question, Prism learns which presentation works best for
          each context and keeps refining itself through feedback.
        </p>
      </header>

      <Card class="p-5 space-y-3 premium-card hover-lift">
        <div class="text-xs uppercase tracking-wide text-muted-foreground">Why this matters</div>
        <p class="text-sm text-muted-foreground leading-relaxed">
          Most AI experiences fail because they optimize for generation, not comprehension. Prism optimizes for
          understanding: concise structure, relevant visuals, and response patterns that match user intent.
        </p>
        <blockquote class="border-l-4 border-cyan-500/60 pl-4 text-sm italic text-foreground/90">
          Better UX is not more content. It is the right structure, at the right depth, at the right moment.
        </blockquote>
      </Card>

      <section class="space-y-3 premium-card rounded-2xl p-5">
        <h2 class="text-xl font-semibold">How Prism works in one turn</h2>
        <ol class="list-decimal pl-5 space-y-2 text-sm text-muted-foreground leading-relaxed">
          <li>Reads intent + context from the current message and short conversation memory.</li>
          <li>Selects a response strategy using posterior confidence (bandit-based adaptation).</li>
          <li>Generates text and optional widget output when visual presentation is beneficial.</li>
          <li>Collects explicit and implicit feedback, then updates strategy quality beliefs.</li>
        </ol>
      </section>

      <section class="space-y-3 premium-card rounded-2xl p-5">
        <h2 class="text-xl font-semibold">How Prism reduces cognitive overload</h2>
        <ul class="list-disc pl-5 space-y-2 text-sm text-muted-foreground leading-relaxed">
          <li><span class="font-medium text-foreground">Adaptive formatting:</span> switches between concise bullets, narrative, table, or step-flow based on user need.</li>
          <li><span class="font-medium text-foreground">Progressive disclosure:</span> core answer first, diagnostics only when needed.</li>
          <li><span class="font-medium text-foreground">Signal-aware visuals:</span> widgets are favored for data-rich prompts, not every question.</li>
          <li><span class="font-medium text-foreground">Feedback loop:</span> helpful/not-helpful signals tune future output toward lower effort.</li>
        </ul>
      </section>

      <Card class="p-5 space-y-3 premium-card hover-lift">
        <h2 class="text-xl font-semibold">Token efficiency playbook</h2>
        <p class="text-sm text-muted-foreground leading-relaxed">
          Prism is designed to balance quality with token cost. Use this playbook to reduce output tokens safely:
        </p>
        <ul class="list-disc pl-5 space-y-2 text-sm text-muted-foreground leading-relaxed">
          <li><span class="font-medium text-foreground">Lower generation budget</span> for routine turns (`COMBINED_MAX_TOKENS`).</li>
          <li><span class="font-medium text-foreground">Tighten widget gating</span> so visuals appear only for real chart/table intent.</li>
          <li><span class="font-medium text-foreground">Prefer concise strategies</span> for operational or repetitive queries.</li>
          <li><span class="font-medium text-foreground">Compact context windows</span> by keeping only useful recent history.</li>
          <li><span class="font-medium text-foreground">Route by complexity</span> (lighter model for simple, stronger model for complex).</li>
        </ul>
        <div class="rounded-xl border border-border/50 bg-background/60 p-3 text-xs text-muted-foreground">
          Quick win: pair concise strategy defaults + strict widget gating. This usually gives the fastest token reduction
          without noticeable quality loss.
        </div>
      </Card>

      <section class="space-y-3 premium-card rounded-2xl p-5">
        <h2 class="text-xl font-semibold">What teams can control</h2>
        <p class="text-sm text-muted-foreground leading-relaxed">
          Teams can configure model/provider mode, token budgets, widget mode, strategy library, and diagnostic visibility.
          Every adjustment is observable in analytics, making iteration measurable instead of guesswork.
        </p>
      </section>

      <Card class="p-5 space-y-3 premium-card hover-lift">
        <h2 class="text-xl font-semibold">Bottom line</h2>
        <p class="text-sm text-muted-foreground leading-relaxed">
          Prism turns AI output into a readable product experience: adaptive, transparent, and cost-aware. It improves
          comprehension, lowers cognitive load, and gives engineering teams the controls needed to optimize quality and
          token efficiency together.
        </p>
      </Card>
    </article>
  </div>
</template>

