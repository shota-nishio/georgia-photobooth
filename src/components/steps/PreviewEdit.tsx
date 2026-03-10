import { useRef, useEffect, useCallback, useState } from 'react'
import { useTranslation } from '../../i18n/useTranslation'
import { compositeImage } from '../../services/compositor'
import { loadImage, blobToImage } from '../../utils/imageHelpers'
import { formatDate } from '../../utils/dateFormat'
import { BACKGROUNDS } from '../../types'
import './PreviewEdit.css'

const CANVAS_W = 1080
const CANVAS_H = 1440

interface Props {
  personBlob: Blob
  backgroundId: string
  onComplete: (canvas: HTMLCanvasElement) => void
  onChangeBackground: () => void
  onBack: () => void
}

export function PreviewEdit({
  personBlob,
  backgroundId,
  onComplete,
  onChangeBackground,
  onBack,
}: Props) {
  const { t, lang } = useTranslation()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [personPos, setPersonPos] = useState({ x: 0, y: 0 })
  const [personScale, setPersonScale] = useState(1)
  const [ready, setReady] = useState(false)
  const imagesRef = useRef<{
    bg: HTMLImageElement
    person: HTMLImageElement
  } | null>(null)
  const dragRef = useRef<{
    startX: number
    startY: number
    startPosX: number
    startPosY: number
    initialDist: number
    initialScale: number
  } | null>(null)

  const render = useCallback(() => {
    if (!imagesRef.current || !canvasRef.current) return
    const { bg, person } = imagesRef.current

    const canvas = compositeImage({
      background: bg,
      person,
      frame: null,
      logo: null,
      dateText: formatDate(lang),
      personPosition: personPos,
      personScale,
      canvasWidth: CANVAS_W,
      canvasHeight: CANVAS_H,
    })

    const displayCtx = canvasRef.current.getContext('2d')!
    canvasRef.current.width = CANVAS_W
    canvasRef.current.height = CANVAS_H
    displayCtx.drawImage(canvas, 0, 0)
  }, [personPos, personScale, lang])

  useEffect(() => {
    const bgOption = BACKGROUNDS.find((b) => b.id === backgroundId)
    if (!bgOption) return

    Promise.all([loadImage(bgOption.src), blobToImage(personBlob)]).then(
      ([bg, person]) => {
        // Center person, scale to fit canvas
        const fitScale = Math.min(
          CANVAS_W / person.width,
          CANVAS_H / person.height,
          1
        )
        const scaledW = person.width * fitScale
        const x = (CANVAS_W - scaledW) / 2
        const y = CANVAS_H - person.height * fitScale

        imagesRef.current = { bg, person }
        setPersonScale(fitScale)
        setPersonPos({ x, y })
        setReady(true)
      }
    )
  }, [personBlob, backgroundId])

  useEffect(() => {
    if (ready) render()
  }, [ready, render])

  const getCanvasCoords = (clientX: number, clientY: number) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    const scaleX = CANVAS_W / rect.width
    const scaleY = CANVAS_H / rect.height
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    }
  }

  const handleTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length === 1) {
      const { x, y } = getCanvasCoords(
        e.touches[0].clientX,
        e.touches[0].clientY
      )
      dragRef.current = {
        startX: x,
        startY: y,
        startPosX: personPos.x,
        startPosY: personPos.y,
        initialDist: 0,
        initialScale: personScale,
      }
    } else if (e.touches.length === 2) {
      const dx = e.touches[0].clientX - e.touches[1].clientX
      const dy = e.touches[0].clientY - e.touches[1].clientY
      const dist = Math.sqrt(dx * dx + dy * dy)
      if (dragRef.current) {
        dragRef.current.initialDist = dist
        dragRef.current.initialScale = personScale
      }
    }
  }

  const handleTouchMove = (e: React.TouchEvent) => {
    e.preventDefault()
    if (!dragRef.current) return

    if (e.touches.length === 1) {
      const { x, y } = getCanvasCoords(
        e.touches[0].clientX,
        e.touches[0].clientY
      )
      const dx = x - dragRef.current.startX
      const dy = y - dragRef.current.startY
      setPersonPos({
        x: dragRef.current.startPosX + dx,
        y: dragRef.current.startPosY + dy,
      })
    } else if (e.touches.length === 2 && dragRef.current.initialDist > 0) {
      const dx = e.touches[0].clientX - e.touches[1].clientX
      const dy = e.touches[0].clientY - e.touches[1].clientY
      const dist = Math.sqrt(dx * dx + dy * dy)
      const ratio = dist / dragRef.current.initialDist
      setPersonScale(
        Math.max(0.2, Math.min(3, dragRef.current.initialScale * ratio))
      )
    }
  }

  const handleTouchEnd = () => {
    dragRef.current = null
  }

  // Mouse drag for desktop
  const handleMouseDown = (e: React.MouseEvent) => {
    const { x, y } = getCanvasCoords(e.clientX, e.clientY)
    dragRef.current = {
      startX: x,
      startY: y,
      startPosX: personPos.x,
      startPosY: personPos.y,
      initialDist: 0,
      initialScale: personScale,
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!dragRef.current) return
    const { x, y } = getCanvasCoords(e.clientX, e.clientY)
    setPersonPos({
      x: dragRef.current.startPosX + (x - dragRef.current.startX),
      y: dragRef.current.startPosY + (y - dragRef.current.startY),
    })
  }

  const handleMouseUp = () => {
    dragRef.current = null
  }

  const handleComplete = () => {
    if (!imagesRef.current) return
    const { bg, person } = imagesRef.current
    const finalCanvas = compositeImage({
      background: bg,
      person,
      frame: null,
      logo: null,
      dateText: formatDate(lang),
      personPosition: personPos,
      personScale,
      canvasWidth: CANVAS_W,
      canvasHeight: CANVAS_H,
    })
    onComplete(finalCanvas)
  }

  return (
    <div className="preview-edit fade-in">
      <h2 className="step-title">{t.preview.title}</h2>
      <p className="preview-hint">
        {t.preview.dragToMove} / {t.preview.pinchToZoom}
      </p>

      <div className="preview-canvas-wrap">
        <canvas
          ref={canvasRef}
          className="preview-canvas"
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        />
      </div>

      <div className="step-actions">
        <button className="btn-primary" onClick={handleComplete} disabled={!ready}>
          {t.preview.complete}
        </button>
        <button className="btn-secondary" onClick={onChangeBackground}>
          {t.preview.changeBackground}
        </button>
        <button className="btn-outline" onClick={onBack}>
          {t.common.back}
        </button>
      </div>
    </div>
  )
}
