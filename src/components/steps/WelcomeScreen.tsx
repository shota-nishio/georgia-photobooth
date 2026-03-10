import { useTranslation } from '../../i18n/useTranslation'
import './WelcomeScreen.css'

interface Props {
  onStart: () => void
}

export function WelcomeScreen({ onStart }: Props) {
  const { t } = useTranslation()

  return (
    <div className="welcome fade-in">
      <div className="welcome-hero">
        <div className="welcome-cross">
          <div className="cross-h" />
          <div className="cross-v" />
        </div>
        <h1 className="welcome-title">{t.welcome.title}</h1>
        <p className="welcome-subtitle">{t.welcome.subtitle}</p>
      </div>
      <p className="welcome-desc">
        {t.welcome.description.split('\n').map((line, i) => (
          <span key={i}>
            {line}
            {i === 0 && <br />}
          </span>
        ))}
      </p>
      <div className="welcome-action">
        <button className="btn-primary btn-start" onClick={onStart}>
          {t.welcome.start}
        </button>
      </div>
    </div>
  )
}
