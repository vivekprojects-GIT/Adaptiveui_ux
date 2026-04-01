/** Mirrors backend `config.STRATEGIES` keys and legacy index.html labels. */
export const STRATEGY_IDS = [
  'structured_bullets',
  'narrative_prose',
  'concise_direct',
  'socratic_questions',
  'step_by_step',
  'comparison_table',
  'visualization',
] as const

export type StrategyId = (typeof STRATEGY_IDS)[number]

export const STRATEGY_LABELS: Record<StrategyId, string> = {
  structured_bullets: 'Structured Bullets',
  narrative_prose: 'Narrative Prose',
  concise_direct: 'Concise & Direct',
  socratic_questions: 'Socratic Questions',
  step_by_step: 'Step-by-Step',
  comparison_table: 'Comparison Table',
  visualization: 'Visualization',
}

export const FEATURE_NAMES = [
  'msg_len',
  'word_ct',
  'has_?',
  'is_long',
  'formal',
  'avg_rwd',
  'msg_num',
  'last_s',
  'trend',
  'noise',
] as const

export type PosteriorMap = Record<string, { r: number; u: number } | undefined>
