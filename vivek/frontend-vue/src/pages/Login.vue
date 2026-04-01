<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { z } from 'zod'
import Input from '@/components/ui/Input.vue'
import Button from '@/components/ui/Button.vue'
import { setAccessToken } from '@/lib/auth'
import { showToast } from '@/lib/toast'

const router = useRouter()
const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5051'
const mode = ref<'login' | 'register'>('login')
const username = ref('')
const email = ref('')
const password = ref('')
const rememberMe = ref(false)
const error = ref<string | null>(null)
const loading = ref(false)

// Forgot-password flow (dev/demo: backend returns reset token).
const forgotOpen = ref(false)
const forgotStep = ref<'request' | 'reset'>('request')
const resetEmail = ref('')
const resetToken = ref('')
const resetPassword = ref('')
const resetPasswordConfirm = ref('')
const forgotLoading = ref(false)
const forgotError = ref<string | null>(null)
const authSchema = z.object({
  email: z.string().email('Please enter a valid email address.'),
  password: z.string().min(6, 'Password must be at least 6 characters.'),
})

async function onSubmit() {
  error.value = null
  loading.value = true
  try {
    const valid = authSchema.safeParse({ email: email.value.trim(), password: password.value })
    if (!valid.success) throw new Error(valid.error.issues[0]?.message || 'Invalid input')

    const url = mode.value === 'register' ? `${API_BASE}/api/auth/register` : `${API_BASE}/api/auth/login`
    const body =
      mode.value === 'register'
        ? { username: username.value, email: email.value, password: password.value }
        : { username_or_email: email.value, password: password.value }

    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })

    const data = await res.json().catch(() => ({}))
    if (!res.ok) {
      throw new Error(data?.detail || data?.error || 'Login failed')
    }

    if (!data?.access_token) {
      throw new Error('Missing access_token in response')
    }
    setAccessToken(data.access_token)
    showToast({ title: mode.value === 'register' ? 'Account created' : 'Signed in', message: 'Welcome!' })

    router.push('/app/chat')
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Request failed'
  } finally {
    loading.value = false
  }
}

function onForgotPassword(e: Event) {
  e.preventDefault()
  forgotOpen.value = true
  forgotStep.value = 'request'
  resetEmail.value = ''
  resetToken.value = ''
  resetPassword.value = ''
  resetPasswordConfirm.value = ''
  forgotError.value = null
}

function onSocialClick(provider: string) {
  showToast({ title: `${provider} login`, message: 'Social login is not implemented in this demo.' })
}

async function requestReset() {
  forgotError.value = null
  forgotLoading.value = true
  try {
    const res = await fetch(`${API_BASE}/api/auth/forgot-password`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: resetEmail.value }),
    })
    const d = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(d?.detail || d?.error || 'Request failed')
    if (!d?.reset_token) {
      forgotError.value = 'Email not found (or reset unavailable).'
      return
    }
    resetToken.value = String(d.reset_token)
    forgotStep.value = 'reset'
    showToast({ title: 'Reset token generated', message: 'Use the token below to set a new password.' })
  } catch (e) {
    forgotError.value = e instanceof Error ? e.message : 'Request failed'
  } finally {
    forgotLoading.value = false
  }
}

async function resetPasswordSubmit() {
  forgotError.value = null
  forgotLoading.value = true
  try {
    if (resetPassword.value.trim().length < 6) throw new Error('Password must be at least 6 characters.')
    if (resetPassword.value !== resetPasswordConfirm.value) throw new Error('Passwords do not match.')

    const res = await fetch(`${API_BASE}/api/auth/reset-password`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: resetToken.value, new_password: resetPassword.value }),
    })
    const d = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(d?.detail || d?.error || 'Reset failed')

    showToast({ title: 'Password updated', message: 'Please log in with your new password.' })
    forgotOpen.value = false
  } catch (e) {
    forgotError.value = e instanceof Error ? e.message : 'Reset failed'
  } finally {
    forgotLoading.value = false
  }
}
</script>

<template>
  <div class="min-h-screen w-full flex items-center justify-center px-5 py-10 bg-background">
    <div class="w-full max-w-6xl grid lg:grid-cols-2 gap-8 items-center">
      <!-- Left: Auth -->
      <div class="mx-auto w-full max-w-md">
        <div class="rounded-3xl glass-panel shadow-xl shadow-emerald-500/10 overflow-hidden">
          <div class="p-8 border-b border-border/60">
            <div class="flex items-center gap-3">
              <div class="h-10 w-10 rounded-2xl bg-primary/20 border border-primary/30 flex items-center justify-center text-primary font-semibold">
                P
              </div>
              <div>
                <div class="text-xs text-muted-foreground">Prism</div>
                <div class="text-sm font-semibold text-foreground/90">{{ mode === 'register' ? 'Create your account' : 'Welcome back' }}</div>
              </div>
            </div>

            <h1 class="text-2xl font-semibold mt-6"> {{ mode === 'register' ? 'Create account' : 'Welcome Back' }} </h1>
            <p class="text-sm text-muted-foreground mt-1">
              {{ mode === 'register' ? 'Enter your details to get started.' : 'Enter your email and password to access your account.' }}
            </p>
          </div>

          <div class="p-8">
            <form class="space-y-4" @submit.prevent="onSubmit">
              <label v-if="mode === 'register'" class="block">
                <div class="text-xs font-semibold mb-1 text-muted-foreground uppercase tracking-wide">Username</div>
                <Input v-model="username" required placeholder="yourname" class="bg-background/60" />
              </label>

              <label class="block">
                <div class="text-xs font-semibold mb-1 text-muted-foreground uppercase tracking-wide">Email</div>
                <Input
                  v-model="email"
                  type="email"
                  required
                  placeholder="you@prism.ai"
                  class="bg-background/60"
                />
              </label>

              <label class="block">
                <div class="text-xs font-semibold mb-1 text-muted-foreground uppercase tracking-wide">Password</div>
                <Input v-model="password" type="password" required placeholder="Prism.lens." class="bg-background/60" />
              </label>

              <div class="flex items-center justify-between gap-3">
                <label class="flex items-center gap-2 text-xs text-muted-foreground select-none">
                  <input v-model="rememberMe" type="checkbox" class="rounded border-input" />
                  Remember Me
                </label>
                <a href="#" class="text-xs text-muted-foreground hover:text-foreground" @click="onForgotPassword">
                  Forgot Your Password?
                </a>
              </div>

              <Button
                type="submit"
                :disabled="loading"
                class="w-full h-11 rounded-xl bg-emerald-600 text-white hover:bg-emerald-700 border border-emerald-600 shadow-md shadow-emerald-500/20"
              >
                {{
                  loading
                    ? mode === 'register'
                      ? 'Creating…'
                      : 'Signing in…'
                    : mode === 'register'
                      ? 'Create'
                      : 'Log In'
                }}
              </Button>
            </form>

            <div v-if="error" class="mt-4 text-sm text-destructive">
              {{ error }}
            </div>

            <div class="mt-6 flex items-center gap-3">
              <div class="h-px bg-border flex-1" />
              <div class="text-xs text-muted-foreground">Or Login With</div>
              <div class="h-px bg-border flex-1" />
            </div>

            <div class="mt-4 grid grid-cols-2 gap-3">
              <Button type="button" variant="outline" class="h-11 rounded-xl" @click="onSocialClick('Google')">Google</Button>
              <Button type="button" variant="outline" class="h-11 rounded-xl" @click="onSocialClick('Apple')">Apple</Button>
            </div>

            <div class="mt-6 text-[11px] text-muted-foreground text-center">
              <span v-if="mode === 'login'">Don't Have An Account? </span>
              <span v-else>You Have An Account? </span>
              <a
                href="#"
                class="font-medium text-foreground hover:underline"
                @click.prevent="mode = mode === 'login' ? 'register' : 'login'; error = null"
              >
                {{ mode === 'login' ? 'Register Now.' : 'Log In' }}
              </a>
            </div>
          </div>
        </div>
      </div>

      <!-- Right: Marketing panel -->
      <div class="hidden lg:block">
        <div
          class="rounded-3xl overflow-hidden relative bg-emerald-700 border border-emerald-300/40 shadow-xl shadow-emerald-500/20 h-[520px]"
        >
          <div class="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(16,185,129,0.72),transparent_55%),radial-gradient(circle_at_80%_30%,rgba(236,253,245,0.35),transparent_60%)]" />
          <div class="absolute inset-0 bg-black/10" />

          <div class="relative h-full p-12 flex flex-col">
            <div class="flex items-center gap-2 text-white/90">
              <div class="h-3 w-3 rounded-full bg-white/80" />
              <div class="text-sm">Adaptive analytics</div>
            </div>

            <div class="mt-12">
              <h2 class="text-4xl font-semibold text-white leading-tight">
                See responses through multiple strategic lenses.
              </h2>
              <p class="text-white/80 mt-4 text-sm max-w-sm">
                Log in to view bandit posteriors, global priors, and live widget insights.
              </p>
            </div>

            <!-- Mock dashboard cards -->
            <div class="mt-auto grid grid-cols-2 gap-4">
              <div class="rounded-2xl bg-white/12 border border-white/20 p-4 backdrop-blur">
                <div class="text-xs text-white/80">Total Responses</div>
                <div class="text-2xl font-semibold text-white mt-2">189,374</div>
                <div class="text-[11px] text-white/70 mt-2">learning across conversations</div>
              </div>
              <div class="rounded-2xl bg-white/12 border border-white/20 p-4 backdrop-blur">
                <div class="text-xs text-white/80">Widget Renders</div>
                <div class="text-2xl font-semibold text-white mt-2">6,248</div>
                <div class="text-[11px] text-white/70 mt-2">interactive insights</div>
              </div>
              <div class="rounded-2xl bg-white/12 border border-white/20 p-4 col-span-2 backdrop-blur">
                <div class="flex items-center justify-between gap-3">
                  <div>
                    <div class="text-xs text-white/80">Signal Updates</div>
                    <div class="text-sm font-medium text-white mt-2">6,248 Units</div>
                  </div>
                  <div class="h-10 w-10 rounded-2xl bg-white/15 border border-white/20 flex items-center justify-center text-white">
                    ↗
                  </div>
                </div>
                <div class="mt-4 h-2 rounded-full bg-white/10 overflow-hidden">
                  <div class="h-full w-2/3 bg-emerald-200/80 rounded-full" />
                </div>
                <div class="mt-2 text-[11px] text-white/70">Last updated just now</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Forgot-password modal -->
  <div
    v-if="forgotOpen"
    class="fixed inset-0 z-[200] bg-black/40 backdrop-blur flex items-center justify-center px-4"
    role="dialog"
    aria-modal="true"
    @click="forgotOpen = false"
  >
    <div
      class="w-full max-w-lg rounded-2xl border glass-panel p-6 shadow-xl"
      @click.stop
    >
      <div class="flex items-start justify-between gap-4 mb-4">
        <div>
          <h2 class="text-lg font-semibold tracking-tight">Forgot password</h2>
          <p class="text-sm text-muted-foreground mt-1">
            Enter your email, get a reset token, then set a new password.
          </p>
        </div>
        <Button type="button" variant="outline" class="h-9 px-3" @click="forgotOpen = false">Close</Button>
      </div>

      <div v-if="forgotStep === 'request'" class="space-y-4">
        <div>
          <div class="text-xs font-semibold mb-1 text-muted-foreground uppercase tracking-wide">Email</div>
          <Input v-model="resetEmail" type="email" required placeholder="you@example.com" class="bg-background/60" />
        </div>
        <Button
          type="button"
          :disabled="forgotLoading"
          class="w-full h-11 rounded-xl"
          @click="requestReset"
        >
          {{ forgotLoading ? 'Generating…' : 'Send reset token' }}
        </Button>
        <div v-if="forgotError" class="text-sm text-destructive">{{ forgotError }}</div>
      </div>

      <div v-else class="space-y-4">
        <div>
          <div class="text-xs font-semibold mb-1 text-muted-foreground uppercase tracking-wide">Reset token</div>
          <Input v-model="resetToken" placeholder="token" class="bg-background/60" />
        </div>

        <div class="grid grid-cols-1 gap-3">
          <div>
            <div class="text-xs font-semibold mb-1 text-muted-foreground uppercase tracking-wide">New password</div>
            <Input v-model="resetPassword" type="password" required placeholder="••••••••" class="bg-background/60" />
          </div>
          <div>
            <div class="text-xs font-semibold mb-1 text-muted-foreground uppercase tracking-wide">Confirm password</div>
            <Input
              v-model="resetPasswordConfirm"
              type="password"
              required
              placeholder="••••••••"
              class="bg-background/60"
            />
          </div>
        </div>

        <Button
          type="button"
          :disabled="forgotLoading"
          class="w-full h-11 rounded-xl"
          @click="resetPasswordSubmit"
        >
          {{ forgotLoading ? 'Updating…' : 'Update password' }}
        </Button>
        <div v-if="forgotError" class="text-sm text-destructive">{{ forgotError }}</div>
      </div>
    </div>
  </div>
</template>

