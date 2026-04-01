import { reactive } from 'vue'

export type ToastItem = {
  id: string
  title: string
  message?: string
  createdAt: number
  timeoutMs: number
}

export const toasts = reactive({
  items: [] as ToastItem[],
})

function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16)
}

export function showToast(input: { title: string; message?: string; timeoutMs?: number }) {
  const item: ToastItem = {
    id: uid(),
    title: input.title,
    message: input.message,
    createdAt: Date.now(),
    timeoutMs: input.timeoutMs ?? 3500,
  }
  toasts.items.unshift(item)
  if (toasts.items.length > 5) toasts.items.pop()

  window.setTimeout(() => dismissToast(item.id), item.timeoutMs)
}

export function dismissToast(id: string) {
  const idx = toasts.items.findIndex((t) => t.id === id)
  if (idx >= 0) toasts.items.splice(idx, 1)
}

