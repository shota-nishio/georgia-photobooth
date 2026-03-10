const MAX_SIZE = 2048

export async function resizeImage(blob: Blob): Promise<Blob> {
  const bitmap = await createImageBitmap(blob)
  const { width, height } = bitmap

  if (width <= MAX_SIZE && height <= MAX_SIZE) {
    bitmap.close()
    return blob
  }

  const scale = MAX_SIZE / Math.max(width, height)
  const newW = Math.round(width * scale)
  const newH = Math.round(height * scale)

  const canvas = document.createElement('canvas')
  canvas.width = newW
  canvas.height = newH
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(bitmap, 0, 0, newW, newH)
  bitmap.close()

  return new Promise((resolve) => {
    canvas.toBlob(
      (b) => resolve(b!),
      'image/jpeg',
      0.92
    )
  })
}

export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

export function blobToImage(blob: Blob): Promise<HTMLImageElement> {
  const url = URL.createObjectURL(blob)
  return loadImage(url)
}
