import { useTranslation } from '../../i18n/useTranslation'
import './LoadingOverlay.css'

interface Props {
  progress: number
}

export function LoadingOverlay({ progress }: Props) {
  const { t } = useTranslation()

  return (
    <div className="loading-overlay">
      <div className="loading-card">
        <div className="loading-spinner" />
        <p className="loading-text">{t.background.processing}</p>
        <div className="loading-bar-track">
          <div
            className="loading-bar-fill"
            style={{ width: `${Math.min(progress, 100)}%` }}
          />
        </div>
        <p className="loading-percent">{progress}%</p>
      </div>
    </div>
  )
}
