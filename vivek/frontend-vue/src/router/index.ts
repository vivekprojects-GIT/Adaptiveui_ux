import { createRouter, createWebHistory } from 'vue-router'
import LoginPage from '@/pages/Login.vue'
import AppShell from '@/layout/AppShell.vue'
import ChatPage from '@/pages/Chat.vue'
import AnalyticsPage from '@/pages/Analytics.vue'
import FutureWorkPage from '@/pages/FutureWork.vue'
import StrategiesPage from '@/pages/Strategies.vue'
import StrategyDetailPage from '@/pages/StrategyDetail.vue'
import AboutPage from '@/pages/About.vue'
import { getAccessToken } from '@/lib/auth'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/app/chat' },
    { path: '/login', component: LoginPage },
    {
      path: '/app',
      component: AppShell,
      children: [
        { path: '', redirect: 'chat' },
        { path: 'chat', component: ChatPage },
        { path: 'analytics', component: AnalyticsPage },
        { path: 'strategies', component: StrategiesPage },
        { path: 'strategies/:id', component: StrategyDetailPage },
        { path: 'future-work', component: FutureWorkPage },
        { path: 'about', component: AboutPage },
      ],
    },
  ],
})

router.beforeEach((to) => {
  const isAuthed = !!getAccessToken()
  if (to.path.startsWith('/app') && !isAuthed) return '/login'
  if (to.path === '/login' && isAuthed) return '/app/chat'
  return true
})

export default router

