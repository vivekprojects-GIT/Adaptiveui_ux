<script setup lang="ts">
import { computed } from 'vue'
import { cn } from '@/lib/utils'

type Variant = 'default' | 'secondary' | 'destructive' | 'outline' | 'ghost'
type Size = 'sm' | 'md' | 'lg' | 'icon'

const props = withDefaults(
  defineProps<{
    variant?: Variant
    size?: Size
    type?: 'button' | 'submit' | 'reset'
    disabled?: boolean
  }>(),
  {
    variant: 'default',
    size: 'md',
    type: 'button',
    disabled: false,
  },
)

const variantClasses = computed(() => {
  switch (props.variant) {
    case 'secondary':
      return 'bg-secondary/90 text-secondary-foreground hover:bg-secondary shadow-sm shadow-slate-900/5'
    case 'destructive':
      return 'bg-destructive text-destructive-foreground hover:bg-destructive/92 shadow-sm shadow-red-600/20'
    case 'outline':
      return 'border border-input/90 bg-white/85 dark:bg-background/58 text-foreground hover:bg-accent/12 hover:border-cyan-400/38'
    case 'ghost':
      return 'text-foreground/90 hover:text-foreground hover:bg-accent/14'
    case 'default':
    default:
      return 'bg-primary/92 text-primary-foreground hover:bg-primary shadow-sm shadow-cyan-500/24 border border-cyan-400/15'
  }
})

const sizeClasses = computed(() => {
  switch (props.size) {
    case 'sm':
      return 'h-8 px-3 text-xs rounded-md'
    case 'lg':
      return 'h-11 px-5 text-sm rounded-xl'
    case 'icon':
      return 'h-9 w-9 p-0 rounded-lg'
    case 'md':
    default:
      return 'h-9 px-4 text-sm rounded-lg'
  }
})
</script>

<template>
  <button
    :type="props.type"
    :disabled="props.disabled"
    :class="
      cn(
        'inline-flex items-center justify-center gap-2 whitespace-nowrap font-medium select-none',
        'transition-all duration-200 ease-[cubic-bezier(0.22,1,0.36,1)] active:translate-y-[1px]',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/85 focus-visible:ring-offset-2 focus-visible:ring-offset-background',
        'disabled:pointer-events-none disabled:opacity-55 disabled:saturate-75',
        '[&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg]:h-4 [&_svg]:w-4',
        sizeClasses,
        variantClasses,
        ($attrs.class as string) || '',
      )
    "
  >
    <slot />
  </button>
</template>

