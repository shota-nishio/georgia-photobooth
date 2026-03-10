import type { AppStep } from '../../types'
import './ProgressBar.css'

interface Props {
  currentStep: AppStep
}

const TOTAL_STEPS = 4

export function ProgressBar({ currentStep }: Props) {
  if (currentStep === 0) return null

  return (
    <div className="progress-bar">
      {Array.from({ length: TOTAL_STEPS }, (_, i) => {
        const step = i + 1
        return (
          <div
            key={step}
            className={`progress-dot ${
              step < currentStep ? 'done' :
              step === currentStep ? 'active' : ''
            }`}
          />
        )
      })}
    </div>
  )
}
