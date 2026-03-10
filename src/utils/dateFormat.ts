import type { Language } from '../i18n/context'

export function formatDate(lang: Language): string {
  const now = new Date()
  const y = now.getFullYear()
  const m = now.getMonth() + 1
  const d = now.getDate()

  switch (lang) {
    case 'ja':
      return `${y}年${m}月${d}日`
    case 'en':
      return now.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      })
    case 'ka':
      return `${d}.${String(m).padStart(2, '0')}.${y}`
    default:
      return `${y}-${String(m).padStart(2, '0')}-${String(d).padStart(2, '0')}`
  }
}
