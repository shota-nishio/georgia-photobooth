import { useTranslation } from '../../i18n/useTranslation'
import { BACKGROUNDS } from '../../types'
import { LoadingOverlay } from '../ui/LoadingOverlay'
import './BackgroundSelect.css'

interface Props {
  selectedId: string | null
  onSelect: (id: string) => void
  onNext: () => void
  onBack: () => void
  bgRemovalDone: boolean
  bgRemovalProgress: number
}

export function BackgroundSelect({
  selectedId,
  onSelect,
  onNext,
  onBack,
  bgRemovalDone,
  bgRemovalProgress,
}: Props) {
  const { t } = useTranslation()
  const bgNames = t.background as Record<string, string>

  return (
    <div className="bg-select fade-in">
      {!bgRemovalDone && <LoadingOverlay progress={bgRemovalProgress} />}

      <h2 className="step-title">{t.background.title}</h2>

      <div className="bg-grid">
        {BACKGROUNDS.map((bg) => (
          <button
            key={bg.id}
            className={`bg-card ${selectedId === bg.id ? 'selected' : ''}`}
            onClick={() => onSelect(bg.id)}
          >
            <div className="bg-thumb-wrap">
              <img
                src={bg.thumbSrc}
                alt={bgNames[bg.nameKey] ?? bg.nameKey}
                className="bg-thumb"
                loading="lazy"
              />
              <div className="bg-thumb-overlay">
                <span className="bg-overlay-name">
                  {bgNames[bg.nameKey] ?? bg.nameKey}
                </span>
              </div>
              {selectedId === bg.id && (
                <div className="bg-check">&#10003;</div>
              )}
            </div>
          </button>
        ))}
      </div>

      <div className="step-actions">
        <button
          className="btn-primary"
          disabled={!selectedId || !bgRemovalDone}
          onClick={onNext}
        >
          {t.background.next}
        </button>
        <button className="btn-outline" onClick={onBack}>
          {t.common.back}
        </button>
      </div>
    </div>
  )
}
