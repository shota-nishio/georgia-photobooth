import { LanguageSwitcher } from '../ui/LanguageSwitcher'
import './Header.css'

export function Header() {
  return (
    <header className="header">
      <div className="header-brand">
        <span className="header-flag">🇬🇪</span>
      </div>
      <LanguageSwitcher />
    </header>
  )
}
