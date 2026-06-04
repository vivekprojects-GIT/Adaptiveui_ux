/**
 * Progressive widget parser.
 *
 * Input: a growing string of widget payload text (JSON or HTML) as it streams
 * from the LLM.
 *
 * Output for JSON schema mode: an array of *fully-parsed* blocks that are
 * definitely complete, plus a flag indicating whether a partial next block
 * is being constructed. This lets the UI reveal blocks one-by-one,
 * Claude-artifact style, without ever trying to render half-built JSON.
 *
 * Output for HTML mode: best-effort sanitized HTML fragment with a small
 * completion heuristic so we only paint when the stream looks renderable.
 */

export type WidgetBlock =
  | { type: 'text'; content?: string }
  | { type: 'kpi_row'; items?: { label?: string; value?: string; tone?: string }[] }
  | { type: 'chart'; title?: string; chart?: Record<string, unknown> }
  | { type: 'image'; title?: string; src?: string; alt?: string; caption?: string; fit?: string }
  | { type: 'table'; title?: string; columns?: string[]; rows?: (string | number)[][] }
  | { type: 'action_row'; buttons?: { label?: string; intent?: string }[] }

export type ProgressiveSchemaState = {
  blocks: WidgetBlock[]
  pendingBlockHint: string | null
  totalSoFar: number
}

const KNOWN_BLOCK_TYPES = new Set(['text', 'kpi_row', 'chart', 'image', 'table', 'action_row'])

/** Map common model spellings to renderer block types. */
function canonicalBlockType(raw: string): string {
  const t = raw.trim().toLowerCase().replace(/-/g, '_')
  const aliases: Record<string, string> = {
    markdown: 'text',
    md: 'text',
    paragraph: 'text',
    rich_text: 'text',
    kpi: 'kpi_row',
    kpis: 'kpi_row',
    kpirow: 'kpi_row',
    metrics: 'kpi_row',
    metric_row: 'kpi_row',
    data_table: 'table',
    grid: 'table',
    actions: 'action_row',
    action: 'action_row',
    plot: 'chart',
    graph: 'chart',
    line_chart: 'chart',
    bar_chart: 'chart',
    photo: 'image',
    picture: 'image',
    img: 'image',
    illustration: 'image',
  }
  return aliases[t] || t
}

function looksLikeKpiItems(items: unknown): boolean {
  if (!Array.isArray(items) || items.length === 0) return false
  const first = items[0]
  if (!first || typeof first !== 'object') return false
  const o = first as Record<string, unknown>
  return (
    typeof o.label === 'string' &&
    (typeof o.value === 'string' || typeof o.value === 'number')
  )
}

/**
 * Coerce one layout entry into a supported block: primitives, missing `type`,
 * and common aliases no longer hit the "Unsupported / unknown" dead-end.
 */
export function normalizeWidgetBlock(raw: unknown): WidgetBlock {
  if (raw === null || raw === undefined) {
    return { type: 'text', content: '' }
  }
  if (typeof raw === 'string' || typeof raw === 'number' || typeof raw === 'boolean') {
    return { type: 'text', content: String(raw) }
  }
  if (Array.isArray(raw)) {
    return { type: 'text', content: JSON.stringify(raw) }
  }
  if (typeof raw !== 'object') {
    return { type: 'text', content: String(raw) }
  }

  const b = { ...(raw as Record<string, unknown>) }
  const rawType = typeof b.type === 'string' ? b.type : ''
  let t = rawType ? canonicalBlockType(rawType) : ''

  if (t && KNOWN_BLOCK_TYPES.has(t)) {
    return { ...b, type: t } as WidgetBlock
  }

  if (looksLikeKpiItems(b.items)) {
    return { ...b, type: 'kpi_row' } as WidgetBlock
  }
  if (b.chart && typeof b.chart === 'object') {
    return { ...b, type: 'chart' } as WidgetBlock
  }
  if (typeof b.src === 'string' && b.src.trim()) {
    return { ...b, type: 'image' } as WidgetBlock
  }
  if (Array.isArray(b.rows)) {
    return { ...b, type: 'table' } as WidgetBlock
  }
  if (Array.isArray(b.buttons)) {
    return { ...b, type: 'action_row' } as WidgetBlock
  }
  const prose = b.content ?? b.body ?? b.text ?? b.markdown ?? b.md
  if (typeof prose === 'string') {
    return { type: 'text', content: prose }
  }

  try {
    return { type: 'text', content: JSON.stringify(b, null, 2) }
  } catch {
    return { type: 'text', content: '[widget block]' }
  }
}

/** Strip common markdown fences / preambles so we can find the first JSON object. */
function stripFences(s: string): string {
  let out = s
  const fence = out.match(/```(?:json|html)?\s*([\s\S]*?)(?:```|$)/i)
  if (fence && fence[1]) out = fence[1]
  out = out.replace(/```[\w-]*$/i, '')
  return out.trim()
}

/** Locate the first `"layout"` or `"blocks"` array opener and return its index. */
function findLayoutArrayStart(s: string): number {
  const m = s.match(/"(layout|blocks|components)"\s*:\s*\[/i)
  if (!m) return -1
  return (m.index ?? 0) + m[0].length
}

/**
 * Starting at position `start` (the char after `[`), extract *complete* JSON
 * objects until we hit a partial one or the closing `]`. Returns the index
 * where scanning stopped (so the caller can keep going on the next tick).
 */
function extractCompleteObjects(s: string, start: number): { objs: string[]; lastComplete: number; pendingType: string | null } {
  const objs: string[] = []
  let i = start
  let lastComplete = start
  let pendingType: string | null = null

  while (i < s.length) {
    while (i < s.length && /[\s,]/.test(s[i] || '')) i += 1
    if (i >= s.length) break
    if (s[i] === ']') break
    if (s[i] !== '{') {
      i += 1
      continue
    }

    const objStart = i
    let depth = 0
    let inString = false
    let escape = false
    let complete = false

    for (; i < s.length; i += 1) {
      const c = s[i]
      if (inString) {
        if (escape) {
          escape = false
        } else if (c === '\\') {
          escape = true
        } else if (c === '"') {
          inString = false
        }
        continue
      }
      if (c === '"') {
        inString = true
        continue
      }
      if (c === '{') depth += 1
      else if (c === '}') {
        depth -= 1
        if (depth === 0) {
          i += 1
          complete = true
          break
        }
      }
    }

    if (!complete) {
      const partial = s.slice(objStart, i)
      const typeMatch = partial.match(/"type"\s*:\s*"([^"]+)"/)
      pendingType = typeMatch ? typeMatch[1] : null
      break
    }

    objs.push(s.slice(objStart, i))
    lastComplete = i
  }

  return { objs, lastComplete, pendingType }
}

export function parseProgressiveWidgetSchema(raw: string): ProgressiveSchemaState {
  const empty: ProgressiveSchemaState = { blocks: [], pendingBlockHint: null, totalSoFar: 0 }
  if (!raw || !raw.trim()) return empty

  const cleaned = stripFences(raw)
  if (!cleaned) return empty

  const layoutStart = findLayoutArrayStart(cleaned)
  if (layoutStart === -1) {
    // Maybe the JSON starts directly with an array?
    const brack = cleaned.indexOf('[')
    if (brack === -1) return empty
    return extractBlocksFromIndex(cleaned, brack + 1)
  }

  return extractBlocksFromIndex(cleaned, layoutStart)
}

function extractBlocksFromIndex(cleaned: string, idx: number): ProgressiveSchemaState {
  const { objs, pendingType } = extractCompleteObjects(cleaned, idx)
  const blocks: WidgetBlock[] = []
  for (const src of objs) {
    try {
      const parsed = JSON.parse(src)
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        blocks.push(normalizeWidgetBlock(parsed))
      }
    } catch {
      // Skip malformed block; the next one might still be fine.
    }
  }
  return {
    blocks,
    pendingBlockHint: pendingType,
    totalSoFar: blocks.length + (pendingType ? 1 : 0),
  }
}
