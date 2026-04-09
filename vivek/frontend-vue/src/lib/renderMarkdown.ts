import MarkdownIt from 'markdown-it'
import multimdTable from 'markdown-it-multimd-table'
import DOMPurify from 'dompurify'

const md = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
}).use(multimdTable)

/**
 * Renders assistant markdown (tables, lists, emphasis) to sanitized HTML.
 */
export function renderAssistantMarkdown(text: string): string {
  const s = String(text ?? '')
  if (!s.trim()) return ''
  const raw = md.render(s)
  return DOMPurify.sanitize(raw)
}
