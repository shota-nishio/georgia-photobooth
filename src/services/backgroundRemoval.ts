import { pipeline, RawImage, env } from '@huggingface/transformers'

// Disable local model check (always fetch from HuggingFace Hub)
env.allowLocalModels = false

const MODEL_ID = 'onnx-community/depth-anything-v2-small'
const MODEL_DTYPE = 'q8' as const
const DEPTH_THRESHOLD_PERCENTILE = 40 // keep closest 60% as foreground
const BLUR_RADIUS = 5

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let depthEstimator: any = null
let preloaded = false

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
 * Preload the depth estimation model.
 * Called once at app startup (Welcome screen).
 */
export async function preloadModels(
  onProgress?: (percent: number) => void
): Promise<void> {
  if (preloaded) return
  try {
    depthEstimator = await pipeline('depth-estimation', MODEL_ID, {
      dtype: MODEL_DTYPE,
      device: detectDevice(),
      progress_callback: (progress: { status: string; progress?: number }) => {
        if (onProgress && progress.progress != null) {
          onProgress(Math.round(progress.progress))
        }
      },
    })
    preloaded = true
  } catch {
    // Preload failure is non-fatal; model will download on demand
  }
}

/**
 * Remove background using depth estimation.
 * Keeps everything close to the camera (person, table, food, etc.)
 * and makes distant background transparent.
 */
export async function removeBg(
  imageSource: Blob,
  onProgress?: (percent: number) => void
): Promise<Blob> {
  // Ensure model is loaded
  if (!depthEstimator) {
    onProgress?.(0)
    depthEstimator = await pipeline('depth-estimation', MODEL_ID, {
      dtype: MODEL_DTYPE,
      device: detectDevice(),
      progress_callback: (progress: { status: string; progress?: number }) => {
        if (onProgress && progress.progress != null) {
          onProgress(Math.round(progress.progress * 0.5)) // 0-50% for model load
        }
      },
    })
  }

  onProgress?.(50)

  // 1. Load image as RawImage
  const url = URL.createObjectURL(imageSource)
  const rawImage = await RawImage.fromURL(url)
  URL.revokeObjectURL(url)

  onProgress?.(55)

  // 2. Run depth estimation
  const output = await depthEstimator(rawImage)
  const depthResult = Array.isArray(output) ? output[0] : output
  const depth = depthResult.depth as RawImage

  onProgress?.(75)

  // 3. Create binary mask from depth map using percentile threshold
  const depthData = depth.data as Uint8Array
  const depthW = depth.width
  const depthH = depth.height
  const mask = createDepthMask(depthData, depthW, depthH, DEPTH_THRESHOLD_PERCENTILE, BLUR_RADIUS)

  onProgress?.(85)

  // 4. Get original image dimensions and pixel data
  const origBitmap = await createImageBitmap(imageSource)
  const origW = origBitmap.width
  const origH = origBitmap.height

  // Draw original image to canvas
  const canvas = document.createElement('canvas')
  canvas.width = origW
  canvas.height = origH
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(origBitmap, 0, 0)
  origBitmap.close()

  // Get pixel data
  const imageData = ctx.getImageData(0, 0, origW, origH)

  // 5. Resize mask from depth map size to original image size
  const resizedMask = resizeMask(mask, depthW, depthH, origW, origH)

  // 6. Apply mask as alpha channel
  for (let i = 0; i < resizedMask.length; i++) {
    imageData.data[i * 4 + 3] = resizedMask[i]
  }
  ctx.putImageData(imageData, 0, 0)

  onProgress?.(95)

  // 7. Export as PNG blob
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
// Utility functions
// ──────────────────────────────────────────────

/**
 * Create a binary mask from a depth map using percentile thresholding.
 * Higher depth values = closer to camera = foreground.
 */
function createDepthMask(
  depthData: Uint8Array,
  width: number,
  height: number,
  percentile: number,
  blurRadius: number
): Uint8Array {
  // Calculate cutoff value at given percentile
  const sorted = Array.from(depthData).sort((a, b) => a - b)
  const cutoffIndex = Math.floor(sorted.length * (percentile / 100))
  const cutoff = sorted[cutoffIndex]

  // Create binary mask: foreground (close) = 255, background (far) = 0
  const mask = new Uint8Array(width * height)
  for (let i = 0; i < depthData.length; i++) {
    mask[i] = depthData[i] >= cutoff ? 255 : 0
  }

  // Smooth edges with Gaussian blur
  return gaussianBlur(mask, width, height, blurRadius)
}

/**
 * Separable Gaussian blur (horizontal pass then vertical pass).
 */
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

  // Horizontal pass
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

  // Vertical pass
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

/**
 * Resize a single-channel mask using bilinear interpolation.
 */
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
