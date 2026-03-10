export interface CompositeOptions {
  background: HTMLImageElement
  person: HTMLImageElement
  frame: HTMLImageElement | null
  logo: HTMLImageElement | null
  dateText: string
  personPosition: { x: number; y: number }
  personScale: number
  canvasWidth: number
  canvasHeight: number
}

export function compositeImage(options: CompositeOptions): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = options.canvasWidth
  canvas.height = options.canvasHeight
  const ctx = canvas.getContext('2d')!

  // Layer 1: Background (cover-fit)
  drawCover(ctx, options.background, canvas.width, canvas.height)

  // Layer 2: Person (positioned & scaled)
  ctx.save()
  const personW = options.person.width * options.personScale
  const personH = options.person.height * options.personScale
  ctx.drawImage(
    options.person,
    options.personPosition.x,
    options.personPosition.y,
    personW,
    personH
  )
  ctx.restore()

  // Layer 3: Frame overlay
  if (options.frame) {
    ctx.drawImage(options.frame, 0, 0, canvas.width, canvas.height)
  }

  // Layer 4: Watermark logo (bottom-right, 15% opacity)
  if (options.logo) {
    ctx.save()
    ctx.globalAlpha = 0.15
    const logoSize = Math.round(canvas.width * 0.12)
    ctx.drawImage(
      options.logo,
      canvas.width - logoSize - 24,
      canvas.height - logoSize - 24,
      logoSize,
      logoSize
    )
    ctx.restore()
  }

  // Layer 5: Date stamp (bottom-left)
  if (options.dateText) {
    ctx.save()
    const fontSize = Math.round(canvas.width * 0.028)
    ctx.font = `500 ${fontSize}px "Inter", "Noto Sans JP", sans-serif`
    ctx.fillStyle = 'rgba(255, 255, 255, 0.85)'
    ctx.shadowColor = 'rgba(0, 0, 0, 0.6)'
    ctx.shadowBlur = 4
    ctx.shadowOffsetX = 1
    ctx.shadowOffsetY = 1
    ctx.fillText(options.dateText, 24, canvas.height - 24)
    ctx.restore()
  }

  return canvas
}

function drawCover(
  ctx: CanvasRenderingContext2D,
  img: HTMLImageElement,
  cw: number,
  ch: number
) {
  const imgRatio = img.width / img.height
  const canvasRatio = cw / ch
  let sx = 0,
    sy = 0,
    sw = img.width,
    sh = img.height

  if (imgRatio > canvasRatio) {
    sw = img.height * canvasRatio
    sx = (img.width - sw) / 2
  } else {
    sh = img.width / canvasRatio
    sy = (img.height - sh) / 2
  }

  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, cw, ch)
}

export function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) resolve(blob)
        else reject(new Error('Canvas toBlob failed'))
      },
      'image/png',
      1.0
    )
  })
}
