import { removeBackground, preload, type Config } from '@imgly/background-removal'

const config: Config = {
  model: 'isnet',
  device: 'gpu',
  output: {
    format: 'image/png',
    quality: 0.8,
  },
  progress: () => {},
}

let preloaded = false

export async function preloadModels(
  onProgress?: (percent: number) => void
): Promise<void> {
  if (preloaded) return
  try {
    await preload({
      ...config,
      progress: (_key: string, current: number, total: number) => {
        if (onProgress && total > 0) {
          onProgress(Math.round((current / total) * 100))
        }
      },
    })
    preloaded = true
  } catch {
    // Preload failure is non-fatal; removal will download models on demand
  }
}

export async function removeBg(
  imageSource: Blob,
  onProgress?: (percent: number) => void
): Promise<Blob> {
  const result = await removeBackground(imageSource, {
    ...config,
    progress: (_key: string, current: number, total: number) => {
      if (onProgress && total > 0) {
        onProgress(Math.round((current / total) * 100))
      }
    },
  })
  return result
}
