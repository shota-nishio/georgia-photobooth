import { useTranslation } from '../../i18n/useTranslation'
import './WelcomeScreen.css'

interface Props {
  onStart: () => void
}

export function WelcomeScreen({ onStart }: Props) {
  const { t } = useTranslation()

  return (
    <div className="welcome">
      {/* Full-bleed hero background with slow zoom */}
      <div className="welcome-hero-bg">
        <img
          src="/backgrounds/tbilisi.jpg"
          alt=""
          className="welcome-hero-img"
        />
        <div className="welcome-hero-overlay" />
      </div>

      {/* Content layer */}
      <div className="welcome-content">
        {/* Georgian Bolnisi cross */}
        <div className="welcome-cross-wrap">
          <div className="welcome-cross">
            <div className="cross-h" />
            <div className="cross-v" />
            <div className="cross-arm cross-tl" />
            <div className="cross-arm cross-tr" />
            <div className="cross-arm cross-bl" />
            <div className="cross-arm cross-br" />
          </div>
        </div>

        {/* Glassmorphism card */}
        <div className="welcome-glass">
          <h1 className="welcome-title">{t.welcome.title}</h1>
          <p className="welcome-subtitle">{t.welcome.subtitle}</p>
          <div className="welcome-divider" />
          <p className="welcome-desc">
            {t.welcome.description.split('\n').map((line, i) => (
              <span key={i}>
                {line}
                {i === 0 && <br />}
              </span>
            ))}
          </p>
        </div>

        {/* Premium CTA */}
        <div className="welcome-action">
          <button className="btn-primary btn-start" onClick={onStart}>
            <span className="btn-start-text">{t.welcome.start}</span>
            <span className="btn-start-arrow">&#8594;</span>
          </button>
        </div>
      </div>
    </div>
  )
}
