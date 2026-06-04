// ---------------------------------------------------------------------------
// Widget registry — ONE source of truth (widget-registry.json) drives BOTH:
//   • GENERATE  → backend reads the same JSON to build the LLM prompt vocabulary
//   • RENDER    → this file maps each block type to the REAL Vue component
//
// The JSON holds the LLM-facing vocabulary (type, whenToUse, spec, example).
// The COMPONENTS map below holds the render wiring (type → your component code).
// Add a block: add an entry to widget-registry.json AND one line to COMPONENTS.
// ---------------------------------------------------------------------------
import { defineAsyncComponent, type Component } from 'vue'
import registryData from '@/widget-registry.json'

type VocabEntry = { type: string; whenToUse: string; spec: string[]; example?: unknown }

// Render wiring: block type → your real component (lazy-loaded / code-split).
const COMPONENTS: Record<string, () => Promise<unknown>> = {
  text: () => import('@/components/widgets/TextBlock.vue'),
  kpi_row: () => import('@/components/widgets/KpiRow.vue'),
  chart: () => import('@/components/widgets/ChartBlock.vue'),
  table: () => import('@/components/widgets/TableBlock.vue'),
  action_row: () => import('@/components/widgets/ActionRow.vue'),
  image: () => import('@/components/widgets/ImageBlock.vue'),
  stat_card: () => import('@/components/widgets/StatCardBlock.vue'),
  progress: () => import('@/components/widgets/ProgressList.vue'),
  badge_row: () => import('@/components/widgets/BadgeRow.vue'),
}

export type RegistryEntry = {
  component: Component
  whenToUse: string
  spec: string[]
  example?: unknown
}

const vocab = (registryData as { blocks: VocabEntry[] }).blocks || []

export const widgetRegistry: Record<string, RegistryEntry> = Object.fromEntries(
  vocab
    .filter((b) => COMPONENTS[b.type])
    .map((b) => [
      b.type,
      {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        component: defineAsyncComponent(COMPONENTS[b.type] as any),
        whenToUse: b.whenToUse,
        spec: b.spec,
        example: b.example,
      },
    ]),
)

export function resolveWidget(type?: string): RegistryEntry | null {
  return widgetRegistry[String(type || '').toLowerCase()] || null
}
