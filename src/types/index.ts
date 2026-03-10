export type AppStep = 0 | 1 | 2 | 3 | 4

export interface AppState {
  step: AppStep
  originalPhoto: Blob | null
  removedBgPhoto: Blob | null
  selectedBackgroundId: string | null
  personPosition: { x: number; y: number }
  personScale: number
  bgRemovalProgress: number
  bgRemovalDone: boolean
  modelsPreloaded: boolean
  error: string | null
}

export type AppAction =
  | { type: 'SET_STEP'; step: AppStep }
  | { type: 'SET_ORIGINAL_PHOTO'; photo: Blob }
  | { type: 'SET_REMOVED_BG'; photo: Blob }
  | { type: 'SET_BACKGROUND'; id: string }
  | { type: 'SET_PERSON_POSITION'; position: { x: number; y: number } }
  | { type: 'SET_PERSON_SCALE'; scale: number }
  | { type: 'SET_BG_REMOVAL_PROGRESS'; percent: number }
  | { type: 'SET_BG_REMOVAL_DONE' }
  | { type: 'SET_MODELS_PRELOADED' }
  | { type: 'SET_ERROR'; error: string }
  | { type: 'CLEAR_ERROR' }
  | { type: 'RESET' }

export interface BackgroundOption {
  id: string
  src: string
  thumbSrc: string
  nameKey: string
}

export const BACKGROUNDS: BackgroundOption[] = [
  {
    id: 'tbilisi',
    src: '/backgrounds/tbilisi.jpg',
    thumbSrc: '/backgrounds-thumb/tbilisi.jpg',
    nameKey: 'tbilisi',
  },
  {
    id: 'kutaisi',
    src: '/backgrounds/kutaisi.jpg',
    thumbSrc: '/backgrounds-thumb/kutaisi.jpg',
    nameKey: 'kutaisi',
  },
  {
    id: 'vardzia',
    src: '/backgrounds/vardzia.jpg',
    thumbSrc: '/backgrounds-thumb/vardzia.jpg',
    nameKey: 'vardzia',
  },
  {
    id: 'uplistsikhe',
    src: '/backgrounds/uplistsikhe.jpg',
    thumbSrc: '/backgrounds-thumb/uplistsikhe.jpg',
    nameKey: 'uplistsikhe',
  },
  {
    id: 'tskaltubo',
    src: '/backgrounds/tskaltubo.jpg',
    thumbSrc: '/backgrounds-thumb/tskaltubo.jpg',
    nameKey: 'tskaltubo',
  },
]
