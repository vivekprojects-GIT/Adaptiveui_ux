import { createApp } from 'vue'
import './styles/globals.css'
import App from './App.vue'
import router from './router'
import { initThemeMode } from './lib/theme'

initThemeMode()
createApp(App).use(router).mount('#app')
