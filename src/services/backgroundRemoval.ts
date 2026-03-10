/**
 * Color-first + Depth-assisted background removal
 *
 * Strategy (3 layers):
 * 1. Neutral background detection (COLOR) → primary: position-aware HSL detection
 *    - Top of image: strict (only clearly white/bright wall)
 *    - Bottom of image: aggressive (catches floor, shadows, grey surfaces)
 * 2. Depth estimation (DEPTH) → secondary: remove non-neutral distant objects
 * 3. Person mask boost (IS-NET) → guarantee people are fully opaque
 *
 * Merge logic:
 *   baseAlpha = isBackground ? 0 : depthAlpha
 *   finalAlpha = max(baseAlpha, boostedPersonAlpha)
 *
 * Table/food protection: The patterned tablecloth has high saturation (blue patterns)
 * and food is colorful → neither triggers neutral background detection.
 * Table is also close to camera → depth keeps it.
 *
 * Mobile optimization:
 * - Models run sequentially and are DISPOSED after use
 * - Only one AI model in memory at a time (prevents OOM on mobile)
 * - Model files are cached in browser Cache API → fast reload from cache
 */
import { pipeline, RawImage, env } from '@huggingface/transformers'
import { removeBackground, preload, type Config } from '@imgly/background-removal'

// ── Position-aware neutral background detection config (COLOR) ──
// Top of image: strict thresholds (avoid catching people's faces/clothing)
const WALL_LIGHTNESS_MIN_TOP = 0.62   // only clearly bright wall
const WALL_SATURATION_MAX_TOP = 0.22  // only clearly colorless
// Bottom of image: aggressive thresholds (catch floor, shadows, grey surfaces)
const WALL_LIGHTNESS_MIN_BOTTOM = 0.38 // catches grey/beige floor
const WALL_SATURATION_MAX_BOTTOM = 0.32 // catches warm-toned floor
// Transition zone
const WALL_AGGRESSIVE_START = 0.40    // start easing toward aggressive at 40% from top
const WALL_AGGRESSIVE_FULL = 0.65     // fully aggressive by 65% from top
const WALL_BLUR_RADIUS = 8            // blur the wall mask to smooth edges

// ── Depth estimation config ──
env.allowLocalModels = false
const DEPTH_MODEL_ID = 'onnx-community/depth-anything-v2-small'
const DEPTH_MODEL_DTYPE = 'q8' as const
const DEPTH_BLUR_RADIUS = 10
const SIGMOID_STEEPNESS = 0.14     // moderate transition (person boost protects people)
const THRESHOLD_BIAS = 0.78        // slightly aggressive (safe: person boost overrides)
const VERTICAL_WEIGHT = 0.30       // mild vertical bias

// ── Person mask boost config ──
const PERSON_BOOST_THRESHOLD = 30  // IS-Net alpha > this → 255 (fully opaque)

// ── Person segmentation config (@imgly) ──
// Use 'cpu' as safe default — 'gpu' crashes on many mobile devices.
// IS-Net fp16 is small enough to run fast on CPU via WASM.
const imglyConfig: Config = {
  model: 'isnet_fp16',
  device: 'cpu',
  output: {
    format: 'image/png',
    quality: 0.8,
  },
  progress: () => {},
}

// Cached device detection result
let detectedDevice: 'webgpu' | 'wasm' | null = null

/**
 * Detect available device for depth estimation model.
 * Attempts WebGPU with actual adapter check, falls back to WASM.
 * On mobile, WebGPU adapter usually fails → safe WASM fallback.
 * Result is cached after first detection.
 */
async function detectDevice(): Promise<'webgpu' | 'wasm'> {
  if (detectedDevice) return detectedDevice
  try {
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      const adapter = await (navigator as any).gpu.requestAdapter()
      if (adapter) {
        detectedDevice = 'webgpu'
        return 'webgpu'
      }
    }
  } catch {
    // WebGPU not available or adapter request failed
  }
  detectedDevice = 'wasm'
  return 'wasm'
}

/**
 * Create a depth estimation pipeline with WebGPU → WASM fallback.
 * The pipeline is NOT kept in memory — caller is responsible for disposal.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function createDepthPipeline(onProgress?: (progress: { status: string; progress?: number }) => void): Promise<any> {
  const device = await detectDevice()
  try {
    return await pipeline('depth-estimation', DEPTH_MODEL_ID, {
      dtype: DEPTH_MODEL_DTYPE,
      device,
      ...(onProgress ? { progress_callback: onProgress } : {}),
    })
  } catch {
    if (device === 'webgpu') {
      console.warn('WebGPU pipeline failed, falling back to WASM')
      detectedDevice = 'wasm' // Cache the fallback result
      return await pipeline('depth-estimation', DEPTH_MODEL_ID, {
        dtype: DEPTH_MODEL_DTYPE,
        device: 'wasm',
        ...(onProgress ? { progress_callback: onProgress } : {}),
      })
    }
    throw new Error('Failed to load depth estimation model')
  }
}

/**
 * Preload model files to browser cache (download only).
 * Models are NOT kept in memory — they are loaded on demand in removeBg
 * and disposed after use to minimize mobile memory pressure.
 */
export async function preloadModels(
  onProgress?: (percent: number) => void
): Promise<void> {
  try {
    // Phase 1: Download depth model to cache (0-50%)
    const depthPipeline = await createDepthPipeline((progress) => {
      if (onProgress && progress.progress != null) {
        onProgress(Math.round(progress.progress * 0.5)) // 0-50%
      }
    })
    // Immediately dispose to free memory — model files stay in browser cache
    if (depthPipeline.dispose) {
      await depthPipeline.dispose()
    }

    // Phase 2: Download person segmentation model to cache (50-100%)
    await preload({
      ...imglyConfig,
      progress: (_key: string, current: number, total: number) => {
        if (onProgress && total > 0) {
          onProgress(50 + Math.round((current / total) * 50)) // 50-100%
        }
      },
    })
  } catch {
    // Non-fatal: models will download on demand in removeBg
  }
}

/**
 * Remove background using color-first + depth-assisted approach.
 *
 * Layer 1: Neutral background detection (COLOR) — position-aware primary removal
 * Layer 2: Depth estimation (DEPTH) — removes non-neutral distant objects
 * Layer 3: Person mask boost (IS-NET) — guarantees people are opaque
 *
 * No table floor protection needed: table has patterned/colorful surface
 * (high saturation) so it's not caught by neutral detection, and it's
 * close to camera so depth keeps it.
 *
 * Mobile memory strategy:
 * 1. Load depth model → run → DISPOSE (free ~27MB)
 * 2. Load person model → run → auto-cleaned by @imgly
 * Only one AI model in memory at a time.
 */
export async function removeBg(
  imageSource: Blob,
  onProgress?: (percent: number) => void
): Promise<Blob> {
  onProgress?.(0)

  // ── Phase 1: Depth estimation (load → run → dispose) ──
  const depthPipeline = await createDepthPipeline()
  onProgress?.(10)

  const depthMask = await runDepthEstimation(depthPipeline, imageSource, onProgress)

  // CRITICAL: Dispose depth model BEFORE loading person model
  // This frees ~27MB of WASM memory, preventing OOM on mobile
  try {
    if (depthPipeline.dispose) {
      await depthPipeline.dispose()
    }
  } catch {
    // Non-fatal: GC will eventually clean up
  }

  onProgress?.(50)

  // ── Phase 2: Person segmentation (load → run → auto-cleanup) ──
  const personMask = await runPersonSegmentation(imageSource, onProgress)

  onProgress?.(80)

  // ── Phase 3: Merge all layers ──
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

  // Free person canvas memory immediately
  personCanvas.width = 0
  personCanvas.height = 0

  // Resize depth mask to original image size
  const resizedDepthMask = resizeMask(depthMask.data, depthMask.width, depthMask.height, origW, origH)

  onProgress?.(90)

  // ── Merge all 3 layers ──
  for (let i = 0; i < origW * origH; i++) {
    // Layer 1: Neutral background → alpha = 0 (transparent)
    // wallMask[i] = 255 means "this IS background" → remove it
    const isBg = wallMask[i] / 255 // 0.0 = foreground, 1.0 = background
    const depthAlpha = resizedDepthMask[i]

    // baseAlpha: color detection weighted by distance.
    // Close objects (table/food) resist color removal; far objects (wall/floor) don't.
    // Math.pow(1 - depth/255, 1.5) = 0 when close, ~1 when far
    const farness = Math.pow(1 - depthAlpha / 255, 1.5)
    const baseAlpha = Math.round(depthAlpha * (1 - isBg * farness))

    // Layer 3: Person mask boost — with white fringe removal
    // Edge pixels (low IS-Net alpha) that look like wall → don't boost (removes white halo)
    // Strong person pixels (high IS-Net alpha) → always boost
    const rawPersonAlpha = personImageData.data[i * 4 + 3]
    let boostedPerson = 0
    if (rawPersonAlpha > PERSON_BOOST_THRESHOLD) {
      if (rawPersonAlpha >= 128 || isBg < 0.5) {
        // Clearly person OR edge on non-wall area → full boost
        boostedPerson = 255
      }
      // else: weak edge pixel on wall-colored area → white fringe, skip boost
    }

    // Final merge: keep pixel if ANY protective layer says keep
    // Table/food survives via: depth (close to camera) + color (high saturation)
    pixels[i * 4 + 3] = Math.max(baseAlpha, boostedPerson)
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

  // Free canvas memory
  canvas.width = 0
  canvas.height = 0

  onProgress?.(100)
  return result
}

// ──────────────────────────────────────────────
// Layer 1: White wall detection (COLOR)
// ──────────────────────────────────────────────

/**
 * Position-aware neutral background detection using HSL color space.
 * Returns mask: 255 = "this is background" (should be removed), 0 = "foreground"
 *
 * Position-aware thresholds:
 * - Top of image (people's heads, upper wall): STRICT detection
 *   → Only clearly white/bright wall. Avoids faces, light clothing.
 * - Bottom of image (floor, lower wall): AGGRESSIVE detection
 *   → Catches grey, beige, shadowed surfaces. Safe because table has
 *     colorful patterned tablecloth (high saturation) and food is colorful.
 *
 * White clothing on people is protected by person mask boost (Layer 3).
 */
function detectWhiteWall(
  pixels: Uint8ClampedArray,
  width: number,
  height: number
): Uint8Array {
  const mask = new Uint8Array(width * height)

  for (let i = 0; i < width * height; i++) {
    const y = Math.floor(i / width)
    const verticalPos = y / height // 0 = top, 1 = bottom

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

    // Position-aware thresholds: interpolate between strict (top) and aggressive (bottom)
    let blend = 0
    if (verticalPos > WALL_AGGRESSIVE_START) {
      blend = Math.min(1, (verticalPos - WALL_AGGRESSIVE_START) / (WALL_AGGRESSIVE_FULL - WALL_AGGRESSIVE_START))
    }
    const lightnessMin = WALL_LIGHTNESS_MIN_TOP + (WALL_LIGHTNESS_MIN_BOTTOM - WALL_LIGHTNESS_MIN_TOP) * blend
    const saturationMax = WALL_SATURATION_MAX_TOP + (WALL_SATURATION_MAX_BOTTOM - WALL_SATURATION_MAX_TOP) * blend

    // Neutral background: light enough AND colorless enough
    if (l >= lightnessMin && s <= saturationMax) {
      // Soft edge: how strongly "background-like" this pixel is
      const lightnessStrength = Math.min(1, (l - lightnessMin) / Math.max(0.01, 1 - lightnessMin))
      const saturationStrength = Math.min(1, (saturationMax - s) / Math.max(0.01, saturationMax))
      mask[i] = Math.round(lightnessStrength * saturationStrength * 255)
    }
    // else mask[i] remains 0 (foreground)
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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  depthPipeline: any,
  imageSource: Blob,
  onProgress?: (percent: number) => void
): Promise<MaskResult> {
  const url = URL.createObjectURL(imageSource)
  let rawImage: RawImage
  try {
    rawImage = await RawImage.fromURL(url)
  } finally {
    URL.revokeObjectURL(url)
  }

  onProgress?.(20)

  const output = await depthPipeline(rawImage)
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
        onProgress(50 + Math.round((current / total) * 30)) // 50-80%
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
