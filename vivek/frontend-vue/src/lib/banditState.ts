import { reactive } from 'vue'
import type { PosteriorMap } from '@/lib/strategies'

export type RewardLogEntry = {
  strategy: string
  reward: number
  detail: string
  source: string
}

export const banditState = reactive({
  activeStrategy: '',
  activeInstruction: '',
  selectedStrategy: '',
  userPosterior: {} as PosteriorMap,
  globalPosterior: {} as PosteriorMap,
  userBPosterior: {} as PosteriorMap,
  globalN: 0,
  nUsers: 1,
  scores: null as Record<string, number> | null,
  lastXVec: null as number[] | null,
  rewardLog: [] as RewardLogEntry[],
  msgCount: null as number | null,
})

export function applyPosteriorPack(d: {
  posterior?: PosteriorMap
  global?: PosteriorMap
  userb?: PosteriorMap
  global_n?: number
}) {
  if (d.posterior) banditState.userPosterior = d.posterior
  if (d.global) banditState.globalPosterior = d.global
  if (d.userb) banditState.userBPosterior = d.userb
  if (d.global_n != null) banditState.globalN = d.global_n
}

export function resetBanditUi() {
  banditState.rewardLog = []
  banditState.activeStrategy = ''
  banditState.activeInstruction = ''
  banditState.selectedStrategy = ''
  banditState.scores = null
  banditState.lastXVec = null
}

export async function fetchBanditStateFromApi(
  apiBase: string,
  getHeaders: () => HeadersInit,
): Promise<{ msg_count?: number } | null> {
  const r = await fetch(`${apiBase}/api/state`, { headers: getHeaders() })
  if (!r.ok) return null
  const d = await r.json()
  applyPosteriorPack(d)
  banditState.nUsers = d.n_users ?? 1
  banditState.msgCount = typeof d.msg_count === 'number' ? d.msg_count : null
  return { msg_count: d.msg_count }
}
