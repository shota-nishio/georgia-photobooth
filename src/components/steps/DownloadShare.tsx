import { useEffect, useRef, useState } from 'react'
import { useTranslation } from '../../i18n/useTranslation'
import { canvasToBlob } from '../../services/compositor'
import './DownloadShare.css'

interface Props {
  canvas: HTMLCanvasElement
  onRetake: () => void
}

export function DownloadShare({ canvas, onRetake }: Props) {
  const { t } = useTranslation()
  const previewRef = useRef<HTMLCanvasElement>(null)
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null)

  useEffect(() => {
    // Show preview
    if (previewRef.current) {
      previewRef.current.width = canvas.width
      previewRef.current.height = canvas.height
      const ctx = previewRef.current.getContext('2d')!
      ctx.drawImage(canvas, 0, 0)
    }

    // Create download URL
    canvasToBlob(canvas).then((blob) => {
      const url = URL.createObjectURL(blob)
      setDownloadUrl(url)
    })

    return () => {
      if (downloadUrl) URL.revokeObjectURL(downloadUrl)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canvas])

  const filename = `georgia-photobooth-${new Date().toISOString().split('T')[0]}.png`

  return (
    <div className="download fade-in">
      <h2 className="step-title">{t.download.title}</h2>

      <div className="download-preview-wrap">
        <canvas ref={previewRef} className="download-preview" />
      </div>

      <div className="download-thank-you">
        {t.download.thankYou.split('\n').map((line, i) => (
          <p key={i}>{line}</p>
        ))}
      </div>

      <div className="step-actions">
        {downloadUrl && (
          <a
            href={downloadUrl}
            download={filename}
            className="btn-primary download-btn"
          >
            📥 {t.download.downloadPhoto}
          </a>
        )}
        <button className="btn-secondary" onClick={onRetake}>
          📷 {t.download.takeAnother}
        </button>
      </div>
    </div>
  )
}
