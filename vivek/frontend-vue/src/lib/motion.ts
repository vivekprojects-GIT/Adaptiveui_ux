import gsap from 'gsap'

export const MOTION_EASE = [0.22, 1, 0.36, 1] as const
export const MOTION_FAST = { duration: 0.2, easing: MOTION_EASE }
export const MOTION_BASE = { duration: 0.32, easing: MOTION_EASE }
export const MOTION_SLOW = { duration: 0.42, easing: MOTION_EASE }

export function animatePulse(element: Element | null | undefined) {
  if (!element) return
  gsap.fromTo(
    element,
    { opacity: 0.5, scale: 0.96 },
    { opacity: 1, scale: 1, duration: 0.9, yoyo: true, repeat: -1, ease: 'sine.inOut' },
  )
}

export function killAnimationsOf(element: Element | null | undefined) {
  if (!element) return
  gsap.killTweensOf(element)
}
