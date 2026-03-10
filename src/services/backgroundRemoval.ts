/**
 * Color-first + Depth-assisted background removal
 *
 * Strategy (4 layers):
 * 1. White wall detection (COLOR)  → primary: detect bright low-saturation wall pixels
 * 2. Depth estimation (DEPTH)      → secondary: remove non-white distant objects (tapestry etc.)
 * 3. Person mask boost (IS-NET)    → guarantee people are fully opaque
 * 4. Table floor (POSITION)        → guarantee bottom area stays opaque
 *
 * Merge logic:
 *   baseAlpha = isWhiteWall ? 0 : depthAlpha
 *   finalAlpha = max(baseAlpha, boostedPersonAlpha, tableFloor)
 *
 * Key insight: Ambassador's room has WHITE walls. Color detection is far more
 * reliable than depth for separating wall from foreground. Depth only handles
 * non-white background elements (tapestry, ceiling decorations).
 */
import { pipeline, RawImage, env } from '@huggingface/transformers'
import { removeBackground, preload, type Config } from '@imgly/background-removal'

// ── White wall detection config (COLOR) ──
const WALL_LIGHTNESS_MIN = 0.72    // HSL lightness threshold (0-1): above = "bright"
const WALL_SATURATION_MAX = 0.18   // HSL saturation threshold (0-1): below = "colorless"
const WALL_BLUR_RADIUS = 8         // blur the wall mask to smooth edges

// ── Depth estimation config ──
env.allowLocalModels = false
const DEPTH_MODEL_ID = 'onnx-community/depth-anything-v2-small'
const DEPTH_MODEL_DTYPE = 'q8' as const
const DEPTH_BLUR_RADIUS = 10
const SIGMOID_STEEPNESS = 0.10     // gentle transition
const THRESHOLD_BIAS = 0.85        // lenient: keep more foreground
const VERTICAL_WEIGHT = 0.30       // mild vertical bias

// ── Person mask boost config ──
const PERSON_BOOST_THRESHOLD = 30  // IS-Net alpha > this → 255 (fully opaque)

// ── Table retention config ──
const TABLE_FLOOR_START = 0.50     // vertical ratio where floor begins (0=top, 1=bottom)
const TABLE_FLOOR_RAMP = 0.12      // fast ramp: full strength by verticalPos 0.62
const TABLE_FLOOR_STRENGTH = 240   // max floor alpha (0-255)

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
 * Remove background using color-first + depth-assisted approach.
 *
 * Layer 1: White wall detection (COLOR) — primary removal
 * Layer 2: Depth estimation (DEPTH) — removes non-white distant objects
 * Layer 3: Person mask boost (IS-NET) — guarantees people are opaque
 * Layer 4: Table floor (POSITION) — guarantees bottom area is opaque
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

  // Get original image dimensions and pixel data
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

  // Get original pixel data (for white wall detection)
  const imageData = ctx.getImageData(0, 0, origW, origH)
  const pixels = imageData.data

  // ── Layer 1: White wall detection (COLOR) ──
  const wallMaskRaw = detectWhiteWall(pixels, origW, origH)
  const wallMask = gaussianBlur(wallMaskRaw, origW, origH, WALL_BLUR_RADIUS)

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

  // ── Merge all 4 layers ──
  for (let i = 0; i < origW * origH; i++) {
    const y = Math.floor(i / origW)
    const verticalPos = y / origH // 0 = top, 1 = bottom

    // Layer 1: White wall → alpha = 0 (transparent)
    // wallMask[i] = 255 means "this IS white wall" → remove it
    const isWall = wallMask[i] / 255 // 0.0 = not wall, 1.0 = wall
    const depthAlpha = resizedDepthMask[i]

    // baseAlpha: if wall detected, force transparent. Otherwise use depth.
    const baseAlpha = Math.round(depthAlpha * (1 - isWall))

    // Layer 3: Person mask boost — IS-Net alpha > threshold → fully opaque
    const rawPersonAlpha = personImageData.data[i * 4 + 3]
    const boostedPerson = rawPersonAlpha > PERSON_BOOST_THRESHOLD ? 255 : 0

    // Layer 4: Table floor — bottom portion guaranteed opaque
    let tableFloor = 0
    if (verticalPos > TABLE_FLOOR_START) {
      const t = Math.min(1, (verticalPos - TABLE_FLOOR_START) / TABLE_FLOOR_RAMP)
      tableFloor = Math.round(t * TABLE_FLOOR_STRENGTH)
    }

    // Final merge: keep pixel if ANY protective layer says keep
    pixels[i * 4 + 3] = Math.max(baseAlpha, boostedPerson, tableFloor)
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
// Layer 1: White wall detection (COLOR)
// ──────────────────────────────────────────────

/**
 * Detect white/bright wall pixels using HSL color space.
 * Returns mask: 255 = "this is white wall" (should be removed), 0 = "not wall"
 *
 * White wall characteristics:
 * - High lightness (bright)
 * - Low saturation (colorless / neutral)
 *
 * This will also flag white tablecloths and white clothing, but those
 * are protected by person mask (Layer 3) and table floor (Layer 4).
 */
function detectWhiteWall(
  pixels: Uint8ClampedArray,
  width: number,
  height: number
): Uint8Array {
  const mask = new Uint8Array(width * height)

  for (let i = 0; i < width * height; i++) {
    const r = pixels[i * 4] / 255
    const g = pixels[i * 4 + 1] / 255
    const b = pixels[i * 4 + 2] / 255

    // RGB → HSL (we only need S and L)
    const max = Math.max(r, g, b)
    const min = Math.min(r, g, b)
    const l = (max + min) / 2 // lightness

    let s = 0 // saturation
    if (max !== min) {
      const d = max - min
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
    }

    // White wall: bright AND colorless
    if (l >= WALL_LIGHTNESS_MIN && s <= WALL_SATURATION_MAX) {
      // Soft edge: how strongly "wall-like" this pixel is
      // More white = stronger wall signal
      const lightnessStrength = Math.min(1, (l - WALL_LIGHTNESS_MIN) / (1 - WALL_LIGHTNESS_MIN))
      const saturationStrength = Math.min(1, (WALL_SATURATION_MAX - s) / WALL_SATURATION_MAX)
      mask[i] = Math.round(lightnessStrength * saturationStrength * 255)
    }
    // else mask[i] remains 0 (not wall)
  }

  return mask
}

// ──────────────────────────────────────────────
// Layer 2: Depth estimation pipeline
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
  const mask = createDepthMask(depthData, depth.width, depth.height, DEPTH_BLUR_RADIUS)

  return { data: mask, width: depth.width, height: depth.height }
}

// ──────────────────────────────────────────────
// Layer 3: Person segmentation pipeline (@imgly)
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

/**
 * Create a soft mask using position-aware biased Otsu threshold + sigmoid.
 *
 * For the color-first approach, depth is secondary:
 * it only needs to catch NON-white background (tapestry, ceiling decorations).
 * Settings are lenient to avoid removing foreground objects.
 */
function createDepthMask(
  depthData: Uint8Array,
  width: number,
  height: number,
  blurRadius: number
): Uint8Array {
  const otsu = otsuThreshold(depthData)
  const baseThreshold = Math.round(otsu * THRESHOLD_BIAS)

  // Apply position-aware sigmoid soft mask
  const mask = new Uint8Array(width * height)
  for (let i = 0; i < depthData.length; i++) {
    const y = Math.floor(i / width)
    const verticalPos = y / height // 0 = top, 1 = bottom

    // Top: threshold rises (stricter), Bottom: stays at base (lenient)
    const threshold = baseThreshold + (1 - verticalPos) * baseThreshold * VERTICAL_WEIGHT

    const diff = depthData[i] - threshold
    const sigmoid = 1 / (1 + Math.exp(-diff * SIGMOID_STEEPNESS))
    mask[i] = Math.round(sigmoid * 255)
  }

  return gaussianBlur(mask, width, height, blurRadius)
}

/**
 * Otsu's method: find the threshold that minimizes intra-class variance.
 */
function otsuThreshold(data: Uint8Array): number {
  const histogram = new Array(256).fill(0)
  for (let i = 0; i < data.length; i++) {
    histogram[data[i]]++
  }

  const total = data.length
  let sumAll = 0
  for (let i = 0; i < 256; i++) sumAll += i * histogram[i]

  let sumBg = 0
  let weightBg = 0
  let maxVariance = 0
  let bestThreshold = 0

  for (let t = 0; t < 256; t++) {
    weightBg += histogram[t]
    if (weightBg === 0) continue

    const weightFg = total - weightBg
    if (weightFg === 0) break

    sumBg += t * histogram[t]
    const meanBg = sumBg / weightBg
    const meanFg = (sumAll - sumBg) / weightFg

    const variance = weightBg * weightFg * (meanBg - meanFg) * (meanBg - meanFg)
    if (variance > maxVariance) {
      maxVariance = variance
      bestThreshold = t
    }
  }

  return bestThreshold
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
