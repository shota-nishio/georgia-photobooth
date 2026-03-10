import { useRef, useState } from 'react'
import { useTranslation } from '../../i18n/useTranslation'
import { resizeImage } from '../../utils/imageHelpers'
import './PhotoCapture.css'

interface Props {
  onPhotoSelected: (photo: Blob) => void
  onBack: () => void
}

export function PhotoCapture({ onPhotoSelected, onBack }: Props) {
  const { t } = useTranslation()
  const [preview, setPreview] = useState<string | null>(null)
  const [photoBlob, setPhotoBlob] = useState<Blob | null>(null)
  const cameraRef = useRef<HTMLInputElement>(null)
  const uploadRef = useRef<HTMLInputElement>(null)

  const handleFile = async (file: File) => {
    const resized = await resizeImage(file)
    setPhotoBlob(resized)
    setPreview(URL.createObjectURL(resized))
  }

  const handleCapture = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  const handleRetake = () => {
    if (preview) URL.revokeObjectURL(preview)
    setPreview(null)
    setPhotoBlob(null)
  }

  const handleUse = () => {
    if (photoBlob) onPhotoSelected(photoBlob)
  }

  if (preview) {
    return (
      <div className="capture fade-in">
        <div className="capture-preview-wrap">
          <img src={preview} alt="Preview" className="capture-preview-img" />
        </div>
        <div className="step-actions">
          <button className="btn-primary" onClick={handleUse}>
            {t.capture.usePhoto}
          </button>
          <button className="btn-outline" onClick={handleRetake}>
            {t.capture.retake}
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="capture fade-in">
      <h2 className="step-title">{t.capture.title}</h2>
      <div className="capture-options">
        <button
          className="capture-option"
          onClick={() => cameraRef.current?.click()}
        >
          <span className="capture-icon">📷</span>
          <span className="capture-label">{t.capture.takePhoto}</span>
        </button>
        <button
          className="capture-option"
          onClick={() => uploadRef.current?.click()}
        >
          <span className="capture-icon">🖼️</span>
          <span className="capture-label">{t.capture.uploadPhoto}</span>
        </button>
      </div>

      <input
        ref={cameraRef}
        type="file"
        accept="image/*"
        capture="user"
        onChange={handleCapture}
        hidden
      />
      <input
        ref={uploadRef}
        type="file"
        accept="image/*"
        onChange={handleCapture}
        hidden
      />

      <div className="step-actions">
        <button className="btn-outline" onClick={onBack}>
          {t.common.back}
        </button>
      </div>
    </div>
  )
}
