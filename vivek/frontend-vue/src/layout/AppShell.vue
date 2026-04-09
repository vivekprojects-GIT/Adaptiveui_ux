<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink, RouterView, useRoute, useRouter } from 'vue-router'
import { Motion } from '@motionone/vue'
import { Separator } from 'radix-vue'
import AnalyticsPanel from '@/components/AnalyticsPanel.vue'
import Button from '@/components/ui/Button.vue'
import { clearAccessToken, getAccessToken } from '@/lib/auth'
import { clearSessionUsername, setSessionUsername } from '@/lib/sessionUser'
import { getThemeMode, setThemeMode, type ThemeMode } from '@/lib/theme'
import { showToast } from '@/lib/toast'
import { fetchStrategies } from '@/lib/strategiesStore'
import { MOTION_FAST } from '@/lib/motion'
import {
  ArrowLeftIcon,
  ArrowPathIcon,
  ArrowRightIcon,
  ArrowRightOnRectangleIcon,
  Bars3Icon,
  ChartBarIcon,
  ChatBubbleLeftRightIcon,
  Cog6ToothIcon,
  EyeIcon,
  EyeSlashIcon,
  InformationCircleIcon,
  RocketLaunchIcon,
  SparklesIcon,
} from '@/components/icons'

const route = useRoute()
const router = useRouter()
const showMobileNav = ref(false)
const showInsights = ref(true)
const collapseNav = ref(true)

const theme = ref<ThemeMode>(getThemeMode())
const themeLabel = computed(() => (theme.value === 'system' ? 'System' : theme.value === 'dark' ? 'Dark' : 'Light'))

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5051'
const isAdmin = ref(false)

function linkClasses(path: string) {
  const active = route.path === path || route.path.startsWith(path + '/')
  return [
    `rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200 flex items-center ${collapseNav.value ? 'justify-center' : 'gap-2'}`,
    active
      ? 'bg-accent/15 text-foreground shadow-sm'
      : 'text-muted-foreground hover:bg-accent/10 hover:text-foreground hover:-translate-y-0.5 hover:shadow-sm',
  ]
}

const showAnalyticsAside = computed(() => route.path.startsWith('/app/analytics') && showInsights.value)
const isChatRoute = computed(() => route.path.startsWith('/app/chat'))

async function loadMe() {
  const token = getAccessToken()
  if (!token) return
  try {
    const r = await fetch(`${API_BASE}/api/me`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
    const d = await r.json().catch(() => ({}))
    if (r.ok) {
      isAdmin.value = Boolean(d?.is_admin)
      setSessionUsername(d?.username)
    }
  } catch {
    // Non-fatal: leave admin nav hidden.
  }
}

function logout() {
  clearAccessToken()
  clearSessionUsername()
  showToast({ title: 'Signed out' })
  router.push('/login')
}

function cycleTheme() {
  theme.value = theme.value === 'system' ? 'light' : theme.value === 'light' ? 'dark' : 'system'
  setThemeMode(theme.value)
}

function toggleDesktopNav() {
  collapseNav.value = !collapseNav.value
}

function emitChatControl(action: 'toggle-baseline' | 'toggle-insights' | 'open-preferences' | 'reset') {
  window.dispatchEvent(new CustomEvent('chat:control', { detail: { action } }))
}

onMounted(async () => {
  await loadMe()
  if (getAccessToken()) {
    await fetchStrategies(API_BASE)
  }
})
</script>

<template>
  <div class="min-h-screen text-foreground">
    <header v-if="!isChatRoute" class="h-16 border-b/70 glass-panel px-4 lg:px-6 flex items-center justify-between sticky top-0 z-50">
      <div class="flex items-center gap-3 min-w-0">
        <button
          class="md:hidden inline-flex h-9 w-9 items-center justify-center rounded-lg border bg-background/60 hover:bg-accent/10"
          @click="showMobileNav = !showMobileNav"
          aria-label="Toggle navigation"
        >
          <Bars3Icon class="h-4 w-4" />
        </button>
        <RouterLink to="/app/chat" class="flex items-center gap-3 min-w-0">
          <div
            class="h-9 w-9 rounded-xl bg-gradient-to-br from-emerald-500/25 to-cyan-500/15 border border-emerald-200/80 dark:border-border/70 flex items-center justify-center text-transparent bg-clip-text bg-gradient-to-r from-emerald-500 to-cyan-500 font-semibold"
          >
            A
          </div>
          <div class="flex flex-col leading-tight min-w-0">
            <div class="font-semibold truncate text-transparent bg-clip-text bg-gradient-to-r from-purple-300 via-cyan-300 to-emerald-300">
              Prism
            </div>
            <div class="text-[11px] text-muted-foreground -mt-0.5 truncate">Adaptive chat · bandit insights · widgets</div>
          </div>
        </RouterLink>
        </div>

      <div class="flex items-center gap-2">
        <Button
          v-if="route.path.startsWith('/app/analytics')"
          variant="outline"
          type="button"
          class="h-9 px-3 hidden sm:inline-flex"
          @click="showInsights = !showInsights"
        >
          {{ showInsights ? 'Hide charts' : 'Show charts' }}
        </Button>
      </div>
    </header>

    <div class="flex" :class="isChatRoute ? 'h-[100svh]' : 'h-[calc(100svh-4rem)]'">
      <aside
        class="border-r/70 glass-panel p-4 hidden md:block transition-all duration-300"
        :class="collapseNav ? 'w-[74px]' : 'w-64'"
      >
        <div class="flex items-center justify-between mb-3">
          <div v-if="!collapseNav" class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Navigation</div>
          <Motion
            tag="button"
            type="button"
            class="inline-flex h-8 w-8 items-center justify-center rounded-lg border bg-white/80 dark:bg-background/60 hover:bg-accent/15 transition-colors"
            :title="collapseNav ? 'Expand sidebar' : 'Collapse sidebar'"
            @click="toggleDesktopNav"
            :hover="{ scale: 1.04 }"
            :press="{ scale: 0.96 }"
            :transition="MOTION_FAST"
          >
            <component :is="collapseNav ? ArrowRightIcon : ArrowLeftIcon" class="h-4 w-4" />
          </Motion>
        </div>

        <Separator class="h-px bg-border/80 my-3" />
        <div class="space-y-2">
          <RouterLink to="/app/chat" :class="linkClasses('/app/chat')">
            <ChatBubbleLeftRightIcon class="h-4 w-4 shrink-0" /><span v-if="!collapseNav">Chat</span>
          </RouterLink>
          <RouterLink to="/app/analytics" :class="linkClasses('/app/analytics')">
            <ChartBarIcon class="h-4 w-4 shrink-0" /><span v-if="!collapseNav">Analytics</span>
          </RouterLink>
          <RouterLink v-if="isAdmin" to="/app/strategies" :class="linkClasses('/app/strategies')">
            <SparklesIcon class="h-4 w-4 shrink-0" /><span v-if="!collapseNav">Strategies</span>
          </RouterLink>
          <RouterLink to="/app/future-work" :class="linkClasses('/app/future-work')">
            <RocketLaunchIcon class="h-4 w-4 shrink-0" /><span v-if="!collapseNav">Future work</span>
          </RouterLink>
          <RouterLink to="/app/about" :class="linkClasses('/app/about')">
            <InformationCircleIcon class="h-4 w-4 shrink-0" /><span v-if="!collapseNav">About</span>
          </RouterLink>
        </div>

        <Separator class="h-px bg-border/80 my-3" />
        <div class="space-y-2">
          <template v-if="isChatRoute">
            <button
              type="button"
              class="rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200 flex items-center hover:bg-accent/10 hover:text-foreground hover:-translate-y-0.5 hover:shadow-sm text-muted-foreground w-full"
              :class="collapseNav ? 'justify-center' : 'gap-2'"
              @click="emitChatControl('toggle-baseline')"
            >
              <EyeSlashIcon class="h-4 w-4 shrink-0" /><span v-if="!collapseNav">Hide baseline</span>
            </button>
            <button
              type="button"
              class="rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200 flex items-center hover:bg-accent/10 hover:text-foreground hover:-translate-y-0.5 hover:shadow-sm text-muted-foreground w-full"
              :class="collapseNav ? 'justify-center' : 'gap-2'"
              @click="emitChatControl('toggle-insights')"
            >
              <EyeIcon class="h-4 w-4 shrink-0" /><span v-if="!collapseNav">Toggle insights</span>
            </button>
            <button
              type="button"
              class="rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200 flex items-center hover:bg-accent/10 hover:text-foreground hover:-translate-y-0.5 hover:shadow-sm text-muted-foreground w-full"
              :class="collapseNav ? 'justify-center' : 'gap-2'"
              @click="emitChatControl('open-preferences')"
            >
              <Cog6ToothIcon class="h-4 w-4 shrink-0" /><span v-if="!collapseNav">Preferences</span>
            </button>
            <button
              type="button"
              class="rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200 flex items-center hover:bg-accent/10 hover:text-foreground hover:-translate-y-0.5 hover:shadow-sm text-muted-foreground w-full"
              :class="collapseNav ? 'justify-center' : 'gap-2'"
              @click="emitChatControl('reset')"
            >
              <ArrowPathIcon class="h-4 w-4 shrink-0" /><span v-if="!collapseNav">Reset</span>
            </button>
          </template>
          <button
            type="button"
            class="rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200 flex items-center hover:bg-accent/10 hover:text-foreground hover:-translate-y-0.5 hover:shadow-sm text-muted-foreground w-full"
            :class="collapseNav ? 'justify-center' : 'gap-2'"
            @click="cycleTheme"
          >
            <SparklesIcon class="h-4 w-4 shrink-0" /><span v-if="!collapseNav">Theme: {{ themeLabel }}</span>
          </button>
          <button
            v-if="getAccessToken()"
            type="button"
            class="rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200 flex items-center hover:bg-accent/10 hover:text-foreground hover:-translate-y-0.5 hover:shadow-sm text-muted-foreground w-full"
            :class="collapseNav ? 'justify-center' : 'gap-2'"
            @click="logout"
          >
            <ArrowRightOnRectangleIcon class="h-4 w-4 shrink-0" /><span v-if="!collapseNav">Logout</span>
          </button>
        </div>
      </aside>

      <div v-if="showMobileNav" class="md:hidden fixed inset-0 bg-black/30" @click="showMobileNav = false" />
      <aside
        v-if="showMobileNav"
        class="md:hidden fixed left-0 w-64 glass-panel border-r p-4 z-50"
        :class="isChatRoute ? 'top-0 h-[100svh]' : 'top-14 h-[calc(100svh-3.5rem)]'"
      >
        <div class="space-y-2">
          <RouterLink to="/app/chat" :class="linkClasses('/app/chat')" @click="showMobileNav = false">Chat</RouterLink>
          <RouterLink to="/app/analytics" :class="linkClasses('/app/analytics')" @click="showMobileNav = false"
            >Analytics</RouterLink
          >
          <RouterLink
            v-if="isAdmin"
            to="/app/strategies"
            :class="linkClasses('/app/strategies')"
            @click="showMobileNav = false"
            >Strategies</RouterLink
          >
          <RouterLink to="/app/future-work" :class="linkClasses('/app/future-work')" @click="showMobileNav = false"
            >Future work</RouterLink
          >
          <RouterLink to="/app/about" :class="linkClasses('/app/about')" @click="showMobileNav = false"
            >About</RouterLink
          >
          <Separator class="h-px bg-border/80 my-3" />
          <template v-if="isChatRoute">
            <Button variant="outline" type="button" class="h-9 w-full justify-start" @click="emitChatControl('toggle-baseline')">
              Hide baseline
            </Button>
            <Button variant="outline" type="button" class="h-9 w-full justify-start" @click="emitChatControl('toggle-insights')">
              Toggle insights
            </Button>
            <Button variant="outline" type="button" class="h-9 w-full justify-start" @click="emitChatControl('open-preferences')">
              Preferences
            </Button>
            <Button variant="outline" type="button" class="h-9 w-full justify-start" @click="emitChatControl('reset')">
              Reset
            </Button>
          </template>
          <Button variant="outline" type="button" class="h-9 w-full justify-start" @click="cycleTheme">
            Theme: {{ themeLabel }}
          </Button>
          <Button
            v-if="getAccessToken()"
            variant="outline"
            type="button"
            class="h-9 w-full justify-start"
            @click="logout"
          >
            Logout
          </Button>
        </div>
      </aside>

      <main class="flex-1 flex min-h-0 overflow-hidden">
        <section
          class="flex-1 min-h-0 bg-background/30 backdrop-blur"
          :class="isChatRoute ? 'p-1 lg:p-2 overflow-hidden min-h-0' : 'p-4 lg:p-6 overflow-y-auto min-h-0'"
        >
          <RouterView />
        </section>

        <aside v-if="showAnalyticsAside" class="w-[380px] border-l/70 glass-panel p-4 overflow-y-auto hidden lg:block">
          <div class="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-3">Analytics (live)</div>
          <AnalyticsPanel />
        </aside>
      </main>
    </div>
  </div>
</template>

