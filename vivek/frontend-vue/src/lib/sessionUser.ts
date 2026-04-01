import { reactive } from 'vue'

export const sessionUserState = reactive({
  username: '',
})

export function setSessionUsername(username: string | null | undefined) {
  sessionUserState.username = String(username ?? '').trim()
}

export function clearSessionUsername() {
  sessionUserState.username = ''
}

