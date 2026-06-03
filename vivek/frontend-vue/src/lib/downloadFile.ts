/** Trigger a browser download of text/binary content. */
export function downloadTextAsFile(content: string, filename: string, mime = 'text/plain;charset=utf-8') {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.rel = 'noopener'
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

/** Pretty-print JSON when valid; otherwise return the original string. */
export function prettifyJsonIfPossible(raw: string): string {
  const s = String(raw || '').trim()
  if (!s) return s
  try {
    return JSON.stringify(JSON.parse(s), null, 2)
  } catch {
    return s
  }
}
