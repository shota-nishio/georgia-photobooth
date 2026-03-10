import { useReducer, useEffect, useCallback, useState } from 'react'
import type { AppState, AppAction, AppStep } from './types'
import { preloadModels, removeBg } from './services/backgroundRemoval'
import { Header } from './components/layout/Header'
import { ProgressBar } from './components/ui/ProgressBar'
import { WelcomeScreen } from './components/steps/WelcomeScreen'
import { PhotoCapture } from './components/steps/PhotoCapture'
import { BackgroundSelect } from './components/steps/BackgroundSelect'
import { PreviewEdit } from './components/steps/PreviewEdit'
import { DownloadShare } from './components/steps/DownloadShare'
import './App.css'

const initialState: AppState = {
  step: 0,
  originalPhoto: null,
  removedBgPhoto: null,
  selectedBackgroundId: null,
  personPosition: { x: 0, y: 0 },
  personScale: 1,
  bgRemovalProgress: 0,
  bgRemovalDone: false,
  modelsPreloaded: false,
  error: null,
}

function reducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_STEP':
      return { ...state, step: action.step }
    case 'SET_ORIGINAL_PHOTO':
      return {
        ...state,
        originalPhoto: action.photo,
        removedBgPhoto: null,
        bgRemovalDone: false,
        bgRemovalProgress: 0,
      }
    case 'SET_REMOVED_BG':
      return { ...state, removedBgPhoto: action.photo, bgRemovalDone: true }
    case 'SET_BACKGROUND':
      return { ...state, selectedBackgroundId: action.id }
    case 'SET_PERSON_POSITION':
      return { ...state, personPosition: action.position }
    case 'SET_PERSON_SCALE':
      return { ...state, personScale: action.scale }
    case 'SET_BG_REMOVAL_PROGRESS':
      return { ...state, bgRemovalProgress: action.percent }
    case 'SET_BG_REMOVAL_DONE':
      return { ...state, bgRemovalDone: true }
    case 'SET_MODELS_PRELOADED':
      return { ...state, modelsPreloaded: true }
    case 'SET_ERROR':
      return { ...state, error: action.error }
    case 'CLEAR_ERROR':
      return { ...state, error: null }
    case 'RESET':
      return { ...initialState, modelsPreloaded: state.modelsPreloaded }
    default:
      return state
  }
}

function App() {
  const [state, dispatch] = useReducer(reducer, initialState)
  const [finalCanvas, setFinalCanvas] = useState<HTMLCanvasElement | null>(null)
  const [isTransitioning, setIsTransitioning] = useState(false)
  const [displayStep, setDisplayStep] = useState<AppStep>(0)

  // Preload background removal models
  useEffect(() => {
    preloadModels((percent) => {
      console.log(`Model preload: ${percent}%`)
    }).then(() => {
      dispatch({ type: 'SET_MODELS_PRELOADED' })
    })
  }, [])

  // Start background removal when photo is set
  const startBgRemoval = useCallback(
    async (photo: Blob) => {
      dispatch({ type: 'SET_BG_REMOVAL_PROGRESS', percent: 0 })
      try {
        const result = await removeBg(photo, (percent) => {
          dispatch({ type: 'SET_BG_REMOVAL_PROGRESS', percent })
        })
        dispatch({ type: 'SET_REMOVED_BG', photo: result })
      } catch (err) {
        console.error('Background removal failed:', err)
        dispatch({
          type: 'SET_ERROR',
          error: 'Background removal failed. Please try again.',
        })
      }
    },
    []
  )

  const goToStep = useCallback((step: AppStep) => {
    setIsTransitioning(true)
    setTimeout(() => {
      dispatch({ type: 'SET_STEP', step })
      setDisplayStep(step)
      setIsTransitioning(false)
    }, 200)
  }, [])

  const handlePhotoSelected = (photo: Blob) => {
    dispatch({ type: 'SET_ORIGINAL_PHOTO', photo })
    goToStep(2)
    startBgRemoval(photo)
  }

  const handleReset = () => {
    setFinalCanvas(null)
    dispatch({ type: 'RESET' })
    setDisplayStep(0)
  }

  return (
    <div className="app">
      <Header />
      <ProgressBar currentStep={state.step} />

      {state.error && (
        <div className="error-banner" onClick={() => dispatch({ type: 'CLEAR_ERROR' })}>
          {state.error}
        </div>
      )}

      <div className={`step-transition ${isTransitioning ? 'step-exit' : 'step-enter'}`}>
        {displayStep === 0 && (
          <WelcomeScreen onStart={() => goToStep(1)} />
        )}

        {displayStep === 1 && (
          <PhotoCapture
            onPhotoSelected={handlePhotoSelected}
            onBack={() => goToStep(0)}
          />
        )}

        {displayStep === 2 && (
          <BackgroundSelect
            selectedId={state.selectedBackgroundId}
            onSelect={(id) => dispatch({ type: 'SET_BACKGROUND', id })}
            onNext={() => goToStep(3)}
            onBack={() => goToStep(1)}
            bgRemovalDone={state.bgRemovalDone}
            bgRemovalProgress={state.bgRemovalProgress}
          />
        )}

        {displayStep === 3 && state.removedBgPhoto && state.selectedBackgroundId && (
          <PreviewEdit
            personBlob={state.removedBgPhoto}
            backgroundId={state.selectedBackgroundId}
            onComplete={(canvas) => {
              setFinalCanvas(canvas)
              goToStep(4)
            }}
            onChangeBackground={() => goToStep(2)}
            onBack={() => goToStep(2)}
          />
        )}

        {displayStep === 4 && finalCanvas && (
          <DownloadShare
            canvas={finalCanvas}
            onRetake={handleReset}
          />
        )}
      </div>
    </div>
  )
}

export default App
