import { createContext, useState, type ReactNode } from 'react'
import { ja, type Translations } from './ja'
import { en } from './en'
import { ka } from './ka'

export type Language = 'ja' | 'en' | 'ka'

const translations: Record<Language, Translations> = { ja, en, ka }

interface LanguageContextType {
  lang: Language
  setLang: (lang: Language) => void
  t: Translations
}

export const LanguageContext = createContext<LanguageContextType>({
  lang: 'ja',
  setLang: () => {},
  t: ja,
})

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Language>('ja')
  return (
    <LanguageContext.Provider value={{ lang, setLang, t: translations[lang] }}>
      {children}
    </LanguageContext.Provider>
  )
}
