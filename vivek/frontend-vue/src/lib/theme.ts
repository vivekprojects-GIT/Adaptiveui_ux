const THEME_KEY = 'theme'

export type ThemeMode = 'light' | 'dark' | 'system'

export function getThemeMode(): ThemeMode {
  const v = localStorage.getItem(THEME_KEY)
  if (v === 'light' || v === 'dark' || v === 'system') return v
  return 'system'
}

export function setThemeMode(mode: ThemeMode) {
  localStorage.setItem(THEME_KEY, mode)
  applyThemeMode(mode)
}

export function applyThemeMode(mode: ThemeMode) {
  const root = document.documentElement
  root.classList.remove('dark')

  if (mode === 'dark') {
    root.classList.add('dark')
    return
  }
  if (mode === 'light') {
    return
  }

  const prefersDark = window.matchMedia?.('(prefers-color-scheme: dark)')?.matches
  if (prefersDark) root.classList.add('dark')
}

let mediaListenerAttached = false

export function initThemeMode() {
  const mode = getThemeMode()
  applyThemeMode(mode)

  if (mediaListenerAttached) return
  mediaListenerAttached = true

  const media = window.matchMedia?.('(prefers-color-scheme: dark)')
  if (!media) return
  media.addEventListener('change', () => {
    if (getThemeMode() === 'system') {
      applyThemeMode('system')
    }
  })
}

