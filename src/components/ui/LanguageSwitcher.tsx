import { useTranslation } from '../../i18n/useTranslation'
import type { Language } from '../../i18n/context'
import './LanguageSwitcher.css'

const LANGS: { code: Language; label: string; flag: string }[] = [
  { code: 'ja', label: '日本語', flag: '🇯🇵' },
  { code: 'en', label: 'EN', flag: '🇬🇧' },
  { code: 'ka', label: 'ქარ', flag: '🇬🇪' },
]

export function LanguageSwitcher() {
  const { lang, setLang } = useTranslation()

  return (
    <div className="lang-switcher">
      {LANGS.map((l) => (
        <button
          key={l.code}
          className={`lang-btn ${lang === l.code ? 'active' : ''}`}
          onClick={() => setLang(l.code)}
          aria-label={l.label}
        >
          <span className="lang-flag">{l.flag}</span>
          <span className="lang-label">{l.label}</span>
        </button>
      ))}
    </div>
  )
}
