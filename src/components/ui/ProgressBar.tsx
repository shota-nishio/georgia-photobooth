import type { AppStep } from '../../types'
import './ProgressBar.css'

interface Props {
  currentStep: AppStep
}

const STEP_ICONS = ['\u{1F4F7}', '\u{1F3D4}\uFE0F', '\u2728', '\u{1F4E5}']
const TOTAL_STEPS = 4

export function ProgressBar({ currentStep }: Props) {
  if (currentStep === 0) return null

  return (
    <div className="progress-bar">
      <div className="progress-track">
        <div
          className="progress-fill"
          style={{ width: `${((currentStep - 1) / (TOTAL_STEPS - 1)) * 100}%` }}
        />
      </div>
      <div className="progress-steps">
        {Array.from({ length: TOTAL_STEPS }, (_, i) => {
          const step = i + 1
          return (
            <div
              key={step}
              className={`progress-step ${
                step < currentStep ? 'done' :
                step === currentStep ? 'active' : ''
              }`}
            >
              <span className="progress-step-icon">{STEP_ICONS[i]}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
