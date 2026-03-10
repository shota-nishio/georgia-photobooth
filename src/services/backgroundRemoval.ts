/**
 * Hybrid background removal: Depth Estimation + Person Segmentation
 *
 * 1. Depth mask  → keeps everything close to camera (table, food, objects)
 * 2. Person mask → keeps people with clean edges
 * 3. OR merge    → union of both masks = complete foreground
 */
import { pipeline, RawImage, env } from '@huggingface/transformers'
import { removeBackground, preload, type Config } from '@imgly/background-removal'

// ── Depth estimation config ──
env.allowLocalModels = false
const DEPTH_MODEL_ID = 'onnx-community/depth-anything-v2-small'
const DEPTH_MODEL_DTYPE = 'q8' as const
const DEPTH_THRESHOLD_PERCENTILE = 55
const BLUR_RADIUS = 5

// ── Person segmentation config (@imgly) ──
const imglyConfig: Config = {
  model: 'isnet_fp16',
  device: 'gpu',
  output: {
    format: 'image/png',
    quality: 0.8,
  },
  progress: () => {},
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let depthEstimator: any = null
let imglyPreloaded = false

function detectDevice(): 'webgpu' | 'wasm' {
  try {
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      return 'webgpu'
    }
  } catch {
    // ignore
  }
  return 'wasm'
}

/**
 * Preload both models in parallel.
 */
export async function preloadModels(
  onProgress?: (percent: number) => void
): Promise<void> {
  const tasks: Promise<void>[] = []

  // Preload depth estimation model
  if (!depthEstimator) {
    tasks.push(
      pipeline('depth-estimation', DEPTH_MODEL_ID, {
        dtype: DEPTH_MODEL_DTYPE,
        device: detectDevice(),
        progress_callback: (progress: { status: string; progress?: number }) => {
          if (onProgress && progress.progress != null) {
            onProgress(Math.round(progress.progress * 0.5)) // 0-50%
          }
        },
      }).then((p) => {
        depthEstimator = p
      })
    )
  }

  // Preload person segmentation model
  if (!imglyPreloaded) {
    tasks.push(
      preload({
        ...imglyConfig,
        progress: (_key: string, current: number, total: number) => {
          if (onProgress && total > 0) {
            onProgress(50 + Math.round((current / total) * 50)) // 50-100%
          }
        },
      }).then(() => {
        imglyPreloaded = true
      })
    )
  }

  try {
    await Promise.all(tasks)
  } catch {
    // Non-fatal: models will download on demand
  }
}

/**
 * Remove background using hybrid approach:
 * depth estimation (for table/food) + person segmentation (for people).
 */
export async function removeBg(
  imageSource: Blob,
  onProgress?: (percent: number) => void
): Promise<Blob> {
  onProgress?.(0)

  // Ensure depth model is loaded
  if (!depthEstimator) {
    depthEstimator = await pipeline('depth-estimation', DEPTH_MODEL_ID, {
      dtype: DEPTH_MODEL_DTYPE,
      device: detectDevice(),
    })
  }

  onProgress?.(10)

  // Run both models in parallel
  const [depthMask, personMask] = await Promise.all([
    runDepthEstimation(imageSource, onProgress),
    runPersonSegmentation(imageSource, onProgress),
  ])

  onProgress?.(80)

  // Get original image dimensions
  const origBitmap = await createImageBitmap(imageSource)
  const origW = origBitmap.width
  const origH = origBitmap.height

  // Draw original to canvas
  const canvas = document.createElement('canvas')
  canvas.width = origW
  canvas.height = origH
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(origBitmap, 0, 0)
  origBitmap.close()

  // Get person mask as pixel data (same size as original)
  const personCanvas = document.createElement('canvas')
  personCanvas.width = origW
  personCanvas.height = origH
  const personCtx = personCanvas.getContext('2d')!
  const personBitmap = await createImageBitmap(personMask)
  personCtx.drawImage(personBitmap, 0, 0, origW, origH)
  personBitmap.close()
  const personImageData = personCtx.getImageData(0, 0, origW, origH)

  // Resize depth mask to original image size
  const resizedDepthMask = resizeMask(depthMask.data, depthMask.width, depthMask.height, origW, origH)

  onProgress?.(90)

  // Merge masks with OR: foreground if EITHER depth says close OR person detected
  const imageData = ctx.getImageData(0, 0, origW, origH)
  for (let i = 0; i < origW * origH; i++) {
    const depthAlpha = resizedDepthMask[i]
    const personAlpha = personImageData.data[i * 4 + 3] // alpha channel of person mask
    // Take the maximum of both masks (OR merge)
    imageData.data[i * 4 + 3] = Math.max(depthAlpha, personAlpha)
  }
  ctx.putImageData(imageData, 0, 0)

  onProgress?.(95)

  // Export as PNG
  const result = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('Canvas toBlob failed'))),
      'image/png'
    )
  })

  onProgress?.(100)
  return result
}

// ──────────────────────────────────────────────
// Depth estimation pipeline
// ──────────────────────────────────────────────

interface MaskResult {
  data: Uint8Array
  width: number
  height: number
}

async function runDepthEstimation(
  imageSource: Blob,
  onProgress?: (percent: number) => void
): Promise<MaskResult> {
  const url = URL.createObjectURL(imageSource)
  const rawImage = await RawImage.fromURL(url)
  URL.revokeObjectURL(url)

  onProgress?.(20)

  const output = await depthEstimator(rawImage)
  const depthResult = Array.isArray(output) ? output[0] : output
  const depth = depthResult.depth as RawImage

  onProgress?.(40)

  const depthData = depth.data as Uint8Array
  const mask = createDepthMask(depthData, depth.width, depth.height, DEPTH_THRESHOLD_PERCENTILE, BLUR_RADIUS)

  return { data: mask, width: depth.width, height: depth.height }
}

// ──────────────────────────────────────────────
// Person segmentation pipeline (@imgly)
// ──────────────────────────────────────────────

async function runPersonSegmentation(
  imageSource: Blob,
  onProgress?: (percent: number) => void
): Promise<Blob> {
  const result = await removeBackground(imageSource, {
    ...imglyConfig,
    progress: (_key: string, current: number, total: number) => {
      if (onProgress && total > 0) {
        onProgress(30 + Math.round((current / total) * 30)) // 30-60%
      }
    },
  })
  return result
}

// ──────────────────────────────────────────────
// Utility functions
// ──────────────────────────────────────────────

function createDepthMask(
  depthData: Uint8Array,
  width: number,
  height: number,
  percentile: number,
  blurRadius: number
): Uint8Array {
  const sorted = Array.from(depthData).sort((a, b) => a - b)
  const cutoffIndex = Math.floor(sorted.length * (percentile / 100))
  const cutoff = sorted[cutoffIndex]

  const mask = new Uint8Array(width * height)
  for (let i = 0; i < depthData.length; i++) {
    mask[i] = depthData[i] >= cutoff ? 255 : 0
  }

  return gaussianBlur(mask, width, height, blurRadius)
}

function gaussianBlur(
  data: Uint8Array,
  width: number,
  height: number,
  radius: number
): Uint8Array {
  if (radius <= 0) return data

  const size = radius * 2 + 1
  const kernel = new Float32Array(size)
  const sigma = radius / 2
  let sum = 0
  for (let i = 0; i < size; i++) {
    const x = i - radius
    kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma))
    sum += kernel[i]
  }
  for (let i = 0; i < size; i++) kernel[i] /= sum

  const temp = new Float32Array(width * height)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let val = 0
      for (let k = -radius; k <= radius; k++) {
        const sx = Math.min(Math.max(x + k, 0), width - 1)
        val += data[y * width + sx] * kernel[k + radius]
      }
      temp[y * width + x] = val
    }
  }

  const result = new Uint8Array(width * height)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let val = 0
      for (let k = -radius; k <= radius; k++) {
        const sy = Math.min(Math.max(y + k, 0), height - 1)
        val += temp[sy * width + x] * kernel[k + radius]
      }
      result[y * width + x] = Math.round(val)
    }
  }

  return result
}

function resizeMask(
  mask: Uint8Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number
): Uint8Array {
  if (srcW === dstW && srcH === dstH) return mask

  const result = new Uint8Array(dstW * dstH)
  const xRatio = srcW / dstW
  const yRatio = srcH / dstH

  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      const srcX = x * xRatio
      const srcY = y * yRatio
      const x0 = Math.floor(srcX)
      const y0 = Math.floor(srcY)
      const x1 = Math.min(x0 + 1, srcW - 1)
      const y1 = Math.min(y0 + 1, srcH - 1)
      const xFrac = srcX - x0
      const yFrac = srcY - y0

      const topLeft = mask[y0 * srcW + x0]
      const topRight = mask[y0 * srcW + x1]
      const bottomLeft = mask[y1 * srcW + x0]
      const bottomRight = mask[y1 * srcW + x1]

      const top = topLeft + (topRight - topLeft) * xFrac
      const bottom = bottomLeft + (bottomRight - bottomLeft) * xFrac
      result[y * dstW + x] = Math.round(top + (bottom - top) * yFrac)
    }
  }

  return result
}
