<script setup lang="ts">
import { ref } from 'vue'
import Button from '@/components/ui/Button.vue'
import WidgetRegistryRenderer from '@/components/WidgetRegistryRenderer.vue'

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ||
  ((typeof location !== 'undefined' && /^(localhost|127.0.0.1)$/.test(location.hostname))
    ? 'http://localhost:5051'
    : '')

type Check = {
  block: string
  label: string
  value: number | string
  ok: boolean
  detail: string
}
type Verification = {
  total_points: number
  passed: number
  failed: number
  grounded: boolean
  has_data?: boolean
  checks: Check[]
}
type FormatInfo = {
  chosen_format: string
  context: string
  feasible: string[]
  features: Record<string, unknown>
}
type RagResult = {
  query: string
  answer: string
  widget_schema: string
  verification: Verification
  retrieved_docs: string[]
  evidence_facts: number
  format: FormatInfo
}

const query = ref("Show Apple's revenue trend from FY2021 to FY2023")
const loading = ref(false)
const error = ref('')
const result = ref<RagResult | null>(null)

const samples = [
  "Show Apple's revenue trend from FY2021 to FY2023",
  'Break down NVIDIA revenue by segment for FY2024',
  'Compare Apple and Microsoft net income',
  "What was Amazon's AWS revenue in FY2023?",
]

async function runQuery() {
  const q = query.value.trim()
  if (!q) return
  loading.value = true
  error.value = ''
  result.value = null
  try {
    const r = await fetch(`${API_BASE}/api/rag_query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q }),
    })
    if (!r.ok) {
      const d = await r.json().catch(() => ({}))
      throw new Error(d.detail || `Request failed (${r.status})`)
    }
    result.value = (await r.json()) as RagResult
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="max-w-4xl mx-auto p-4 space-y-4">
    <div>
      <h1 class="text-xl font-semibold">Finance RAG — evidence-bound widgets</h1>
      <p class="text-sm text-muted-foreground mt-1">
        The answer is synthesized from a ChromaDB of finance filings; the widget is generated from that answer,
        and every chart/KPI value is verified to appear in the answer — numbers the answer didn't state are dropped.
      </p>
    </div>

    <div class="flex flex-wrap gap-2">
      <button
        v-for="(s, i) in samples"
        :key="i"
        class="text-[11px] px-2 py-1 rounded-full border bg-card hover:bg-muted/40 transition"
        @click="query = s"
      >
        {{ s }}
      </button>
    </div>

    <div class="flex gap-2">
      <input
        v-model="query"
        type="text"
        placeholder="Ask a finance question…"
        class="flex-1 rounded-lg border bg-card px-3 py-2 text-sm outline-none focus:border-primary"
        @keydown.enter="runQuery"
      />
      <Button :disabled="loading" @click="runQuery">{{ loading ? 'Querying…' : 'Ask' }}</Button>
    </div>

    <div v-if="error" class="rounded-lg border border-red-500/40 bg-red-500/5 px-3 py-2 text-sm text-red-600 dark:text-red-400">
      {{ error }}
    </div>

    <div v-if="result" class="space-y-4">
      <!-- Verification badge -->
      <div
        class="rounded-xl border px-4 py-3 flex items-center gap-3"
        :class="result.verification.grounded
          ? 'border-emerald-500/40 bg-emerald-500/5'
          : 'border-amber-500/40 bg-amber-500/5'"
      >
        <span
          class="text-xs font-semibold px-2 py-1 rounded-full"
          :class="result.verification.grounded
            ? 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300'
            : 'bg-amber-500/15 text-amber-700 dark:text-amber-300'"
        >
          {{ result.verification.grounded ? '✓ ANSWER-CONSISTENT' : '⚠ PARTIAL' }}
        </span>
        <span class="text-sm text-muted-foreground">
          {{ result.verification.passed }}/{{ result.verification.total_points }} visual values are stated in the answer
          ({{ result.verification.failed }} dropped) · retrieved {{ result.retrieved_docs.length }} docs
        </span>
      </div>

      <!-- Format engine (deterministic) -->
      <div class="rounded-xl border bg-card p-4">
        <div class="text-[11px] uppercase tracking-wide text-muted-foreground mb-2">
          Widget format (chosen by the model, matched to our components)
        </div>
        <div class="flex flex-wrap items-center gap-2 text-sm">
          <span class="text-muted-foreground">model chose</span>
          <span class="px-2 py-0.5 rounded-full bg-primary/15 text-primary text-xs font-semibold">
            {{ result.format.chosen_format }}
          </span>
          <span class="text-muted-foreground">· components we can render:</span>
        </div>
        <div class="mt-3 flex flex-wrap gap-1.5">
          <span
            v-for="f in result.format.feasible"
            :key="f"
            class="text-[10px] px-2 py-0.5 rounded-full border"
            :class="f === result.format.chosen_format
              ? 'border-primary/50 bg-primary/10 text-primary'
              : 'border-border text-muted-foreground'"
          >
            {{ f }}
          </span>
        </div>
      </div>

      <!-- Grounded answer -->
      <div class="rounded-xl border bg-card p-4">
        <div class="text-[11px] uppercase tracking-wide text-muted-foreground mb-2">Grounded answer</div>
        <p class="text-sm whitespace-pre-wrap leading-relaxed">{{ result.answer }}</p>
      </div>

      <!-- Evidence-bound widget (rendered by your components) -->
      <div class="rounded-xl border bg-card p-4">
        <div class="text-[11px] uppercase tracking-wide text-muted-foreground mb-2">Evidence-bound widget</div>
        <WidgetRegistryRenderer :json-str="result.widget_schema" :show-download="false" />
      </div>

      <!-- Per-value verification proof -->
      <div class="rounded-xl border bg-card overflow-hidden">
        <div class="px-4 py-2 border-b text-[11px] uppercase tracking-wide text-muted-foreground">
          Verification — every value vs the answer
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-xs">
            <thead>
              <tr class="text-left text-muted-foreground">
                <th class="px-3 py-2">Status</th>
                <th class="px-3 py-2">Label</th>
                <th class="px-3 py-2">Value</th>
                <th class="px-3 py-2">Check</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(c, i) in result.verification.checks" :key="i" class="border-t border-border/60">
                <td class="px-3 py-1.5">
                  <span :class="c.ok ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'">
                    {{ c.ok ? '✓' : '✗' }}
                  </span>
                </td>
                <td class="px-3 py-1.5">{{ c.label }}</td>
                <td class="px-3 py-1.5 font-mono">{{ c.value }}</td>
                <td class="px-3 py-1.5 text-muted-foreground">{{ c.detail }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>
