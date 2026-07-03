use ndarray::Array4;

/// Image preprocessing utilities for SigLIP2 and CLIP-L/14 models.
pub struct ImagePreprocessor;

impl ImagePreprocessor {
    /// SigLIP2 image preprocessing from file path.
    ///
    /// Steps:
    /// 1. Load image from disk (supports jpg, png, gif, bmp, webp, etc.)
    /// 2. Resize to 256×256 (Catmull-Rom interpolation)
    /// 3. Convert to RGB
    /// 4. Normalize: mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]
    /// 5. Output shape: (1, 3, 256, 256), dtype: f32  (CHW format)
    pub fn preprocess_siglip2(path: &str) -> Result<Array4<f32>, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("Failed to read file '{}': {}", path, e))?;
        Self::preprocess_siglip2_from_bytes(&bytes)
    }

    /// SigLIP2 image preprocessing from memory bytes.
    ///
    /// This avoids re-reading the file from disk when the bytes are already available.
    pub fn preprocess_siglip2_from_bytes(bytes: &[u8]) -> Result<Array4<f32>, String> {
        Self::preprocess_from_bytes(bytes, 256, 256, &[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5])
    }

    /// CLIP-L/14 image preprocessing from file path (for batch encoding).
    ///
    /// Steps:
    /// 1. Load image from disk
    /// 2. Resize to 224×224 (Catmull-Rom interpolation)
    /// 3. Convert to RGB
    /// 4. Normalize: mean = [0.48145466, 0.4578275, 0.40821073],
    ///                std  = [0.26862954, 0.26130258, 0.27577711]
    /// 5. Output shape: (1, 3, 224, 224), dtype: f32  (CHW format)
    pub fn preprocess_cliplarge(path: &str) -> Result<Array4<f32>, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("Failed to read file '{}': {}", path, e))?;
        Self::preprocess_cliplarge_from_bytes(&bytes)
    }

    /// CLIP-L/14 image preprocessing from memory bytes.
    ///
    /// This avoids re-reading the file from disk when the bytes are already available.
    pub fn preprocess_cliplarge_from_bytes(bytes: &[u8]) -> Result<Array4<f32>, String> {
        Self::preprocess_from_bytes(bytes, 224, 224, &[0.48145466, 0.4578275, 0.40821073], &[0.26862954, 0.26130258, 0.27577711])
    }

    /// Generic image preprocessing pipeline from memory bytes.
    ///
    /// * `bytes` — image file bytes in memory
    /// * `target_w` — target width after resize
    /// * `target_h` — target height after resize
    /// * `mean` — per-channel mean (length 3)
    /// * `std` — per-channel std  (length 3)
    fn preprocess_from_bytes(
        bytes: &[u8],
        target_w: u32,
        target_h: u32,
        mean: &[f32; 3],
        std: &[f32; 3],
    ) -> Result<Array4<f32>, String> {
        // 1. Detect image format and load from memory
        let img = if let Ok(format) = image::guess_format(bytes) {
            image::load_from_memory_with_format(bytes, format)
                .map_err(|e| format!("Failed to decode image with detected format: {}", e))?
        } else {
            // Fallback: try to decode without specifying format
            image::load_from_memory(bytes)
                .map_err(|e| format!("Failed to decode image from bytes: {}", e))?
        };

        // 2. Convert to RGB
        let rgb = img.to_rgb8();

        // 3. Resize to target dimensions
        let resized = image::imageops::resize(
            &rgb,
            target_w,
            target_h,
            image::imageops::FilterType::CatmullRom,
        );

        let (width, height) = resized.dimensions();
        debug_assert_eq!(width, target_w);
        debug_assert_eq!(height, target_h);

        // 4. Normalize and build CHW array
        //    Shape: (1, 3, H, W)
        let mut data = Vec::with_capacity((3 * target_h * target_w) as usize);

        for c in 0..3 {
            for y in 0..target_h {
                for x in 0..target_w {
                    let pixel = resized.get_pixel(x, y)[c] as f32 / 255.0_f32;
                    let normalized = (pixel - mean[c]) / std[c];
                    data.push(normalized);
                }
            }
        }

        Array4::from_shape_vec((1, 3, target_h as usize, target_w as usize), data)
            .map_err(|e| format!("Failed to create ndarray: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_siglip2_output_shape() {
        // Placeholder — needs an actual image on disk
    }

    #[test]
    fn test_preprocess_cliplarge_output_shape() {
        // Placeholder — needs an actual image on disk
    }
}
