import { useContext } from 'react'
import { LanguageContext } from './context'

export function useTranslation() {
  return useContext(LanguageContext)
}
