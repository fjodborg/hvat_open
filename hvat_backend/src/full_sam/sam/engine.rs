//! ONNX Runtime-based SAM engine implementation.
//!
//! Uses the `ort` crate for cross-platform ONNX inference with
//! support for multiple execution providers (CUDA, ROCm, DirectML, CoreML, CPU).
//!
//! ## Concurrency Model
//!
//! The encoder and decoder sessions are protected by `tokio::sync::Mutex` and accessed
//! via `spawn_blocking` tasks. This design prevents thread pool exhaustion when multiple
//! concurrent encode/decode operations are queued:
//!
//! - `tokio::sync::Mutex`: Async-aware mutex that integrates with tokio runtime
//! - `blocking_lock()`: Used within `spawn_blocking` to acquire locks without blocking threads
//! - Tasks wait asynchronously in a queue rather than exhausting the blocking thread pool

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use hvat_common::{bilinear_sample, pixel_count};
use ort::execution_providers::cuda::CuDNNConvAlgorithmSearch;
use ort::execution_providers::webgpu::WebGPUExecutionProvider;
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
    DirectMLExecutionProvider, ExecutionProvider as OrtExecutionProviderTrait,
    ExecutionProviderDispatch, ROCmExecutionProvider,
};
use ort::session::Session;
use ort::value::Tensor;
use tokio::sync::Mutex;

use super::backend::{EncoderOutput, ExecutionProvider, SamBackend, SamMaskResult, SamPoint};
use super::models::SamVariant;

/// SAM 2 input image size (model expects 1024x1024).
const SAM_INPUT_SIZE: u32 = 1024;
/// Maximum resolution used for contour/component extraction.
///
/// Keeping this at SAM working resolution avoids expensive O(W*H) component
/// analysis on very large source images.
const MAX_CONTOUR_RESOLUTION: usize = SAM_INPUT_SIZE as usize;
/// SAM protocol currently returns polygons, not raster masks.
///
/// Keep this false to avoid expensive full-resolution binary mask materialization
/// on each inference.
const EMIT_BINARY_MASKS: bool = false;

/// ONNX Runtime-based SAM engine.
///
/// Manages encoder and decoder sessions for SAM 2 inference.
/// Sessions are wrapped in async Mutex because ONNX Runtime's `run` requires `&mut self`.
/// Using tokio::sync::Mutex prevents thread pool exhaustion when multiple concurrent
/// encode/decode operations try to acquire the lock - they wait asynchronously instead
/// of blocking threads.
pub struct OnnxSamEngine {
    encoder: Arc<Mutex<Session>>,
    decoder: Arc<Mutex<Session>>,
    provider: ExecutionProvider,
}

impl OnnxSamEngine {
    /// Create a new ONNX SAM engine.
    ///
    /// # Arguments
    /// * `model_dir` - Directory containing the ONNX model files
    /// * `variant` - Which SAM variant to load (Tiny, Small, etc.)
    /// * `provider` - Execution provider for hardware acceleration
    pub fn new(model_dir: &Path, variant: SamVariant, provider: ExecutionProvider) -> Result<Self> {
        let encoder_path = model_dir.join(variant.encoder_filename());
        let decoder_path = model_dir.join(variant.decoder_filename());

        log::info!(
            "Loading SAM {} models with {} provider",
            variant.name(),
            provider.name()
        );

        // Ensure the global ONNX Runtime environment (and its default logger) is
        // committed *before* any execution provider is configured. Providers such
        // as WebGPU initialize their backend (Dawn) during registration and log
        // through ORT's default logger; that logger only exists once the
        // environment has been created. Relying on `ort`'s lazy init is too late
        // because registration happens during `with_execution_providers`, before
        // `commit_from_file` would otherwise trigger environment creation.
        // `commit()` is idempotent (backed by a `OnceLock`), so repeated calls and
        // concurrent engines are safe.
        if let Err(err) = ort::init().with_name("hvat_sam").commit() {
            log::warn!("Failed to initialize ONNX Runtime environment: {err:#}");
        }

        // The decoder may need a different provider than the encoder (see
        // `decoder_provider`): under WebGPU the decoder's mask output is
        // corrupted, so we keep the encoder on the GPU (the expensive,
        // verified-correct part) and run the cheap decoder on CPU.
        let decoder_provider = Self::decoder_provider(provider);

        // Build both sessions. EP registration is configured to fail loudly
        // (see `provider_dispatch_chain`) rather than silently degrading to CPU
        // while still reporting the requested provider in logs/metrics.
        let built = Self::build_session(&encoder_path, provider)
            .and_then(|enc| Ok((enc, Self::build_session(&decoder_path, decoder_provider)?)));

        let (encoder, decoder, actual_provider) = match built {
            Ok((enc, dec)) => (enc, dec, provider),
            Err(err) if provider != ExecutionProvider::Cpu => {
                log::error!(
                    "SAM {} execution provider failed to initialize ({err:#}); \
                     falling back to CPU. Inference will run on CPU and be \
                     significantly slower.",
                    provider.name()
                );
                let encoder = Self::build_session(&encoder_path, ExecutionProvider::Cpu)
                    .context("Failed to load encoder model (CPU fallback)")?;
                let decoder = Self::build_session(&decoder_path, ExecutionProvider::Cpu)
                    .context("Failed to load decoder model (CPU fallback)")?;
                (encoder, decoder, ExecutionProvider::Cpu)
            }
            Err(err) => return Err(err.context("Failed to load SAM models")),
        };

        log::info!(
            "SAM engine initialized successfully with {} provider",
            actual_provider.name()
        );

        Ok(Self {
            encoder: Arc::new(Mutex::new(encoder)),
            decoder: Arc::new(Mutex::new(decoder)),
            provider: actual_provider,
        })
    }

    /// Build an ONNX session with the specified execution provider.
    fn build_session(model_path: &Path, provider: ExecutionProvider) -> Result<Session> {
        let builder = Session::builder()?;
        let provider_chain = Self::provider_dispatch_chain(provider);
        log::info!(
            "Building ONNX session for '{}' with execution providers: {:?}",
            model_path.display(),
            provider_chain
        );
        let builder = builder.with_execution_providers(provider_chain)?;
        let session = builder.commit_from_file(model_path)?;
        Ok(session)
    }

    /// Provider to use for the *decoder* session given the requested provider.
    ///
    /// The SAM2 decoder produces an incorrect (all-foreground) mask output under
    /// the WebGPU execution provider — its embeddings and IoU predictions match
    /// CPU exactly, but the mask logits are corrupted (the decoder graph runs
    /// many ops on CPU fallback under WebGPU, and the mask branch comes back
    /// wrong; reproduced by `gpu_matches_cpu_reference`). The decoder is cheap
    /// (~30 ms, already mostly CPU-fallback), so we run it on CPU when WebGPU is
    /// requested, preserving the large, verified-correct encoder speedup while
    /// producing correct masks. Other providers (CUDA, etc.) use the decoder on
    /// the same provider as requested.
    fn decoder_provider(requested: ExecutionProvider) -> ExecutionProvider {
        match requested {
            ExecutionProvider::WebGpu => ExecutionProvider::Cpu,
            other => other,
        }
    }

    /// Build the execution provider dispatch chain for a session.
    ///
    /// Non-CPU providers are configured with `error_on_failure()` so that a
    /// registration failure (e.g. a missing CUDA/cuDNN shared library) is
    /// reported as an error from `commit_from_file` instead of silently
    /// falling back to CPU at the ONNX Runtime level. `OnnxSamEngine::new`
    /// handles the CPU fallback explicitly (and updates `self.provider`
    /// accordingly), so timing logs always reflect the provider actually in
    /// use.
    fn provider_dispatch_chain(provider: ExecutionProvider) -> Vec<ExecutionProviderDispatch> {
        match provider {
            ExecutionProvider::Cpu => vec![CPUExecutionProvider::default().build()],
            ExecutionProvider::Cuda => vec![
                CUDAExecutionProvider::default()
                    .with_conv_algorithm_search(CuDNNConvAlgorithmSearch::Heuristic)
                    .with_tf32(true)
                    .with_prefer_nhwc(true)
                    .build()
                    .error_on_failure(),
            ],
            ExecutionProvider::Rocm => {
                vec![ROCmExecutionProvider::default().build().error_on_failure()]
            }
            ExecutionProvider::DirectML => {
                vec![
                    DirectMLExecutionProvider::default()
                        .build()
                        .error_on_failure(),
                ]
            }
            ExecutionProvider::CoreML => {
                vec![
                    CoreMLExecutionProvider::default()
                        .build()
                        .error_on_failure(),
                ]
            }
            // Cross-platform, vendor-neutral GPU path (Dawn → Vulkan/D3D12). No
            // cuDNN/TensorRT dependency. Left at defaults: graph capture and
            // non-default buffer caching corrupted the decoder's mask output
            // (the SAM2 decoder runs many ops on CPU fallback, which interacts
            // badly with those options) — see `gpu_matches_cpu_reference`.
            ExecutionProvider::WebGpu => {
                vec![
                    WebGPUExecutionProvider::default()
                        .build()
                        .error_on_failure(),
                ]
            }
        }
    }

    /// Preprocess image for SAM encoder.
    ///
    /// Resizes to 1024x1024 and normalizes pixel values.
    /// Returns data in NCHW format as a flat Vec.
    fn preprocess_image(image_rgb: &[u8], width: u32, height: u32) -> Result<Vec<f32>> {
        // Create image from raw bytes
        let img = image::RgbImage::from_raw(width, height, image_rgb.to_vec())
            .context("Failed to create image from bytes")?;

        // Resize to SAM input size
        let resized = image::imageops::resize(
            &img,
            SAM_INPUT_SIZE,
            SAM_INPUT_SIZE,
            image::imageops::FilterType::Lanczos3,
        );

        // Convert to float and normalize to [0, 1]
        // SAM expects NCHW format: (1, 3, 1024, 1024)
        let size = SAM_INPUT_SIZE as usize;
        let mut input = vec![0.0f32; 3 * size * size];

        for (y, row) in resized.enumerate_rows() {
            for (x, _, pixel) in row {
                let base = y as usize * size + x as usize;
                input[base] = pixel[0] as f32 / 255.0; // R
                input[size * size + base] = pixel[1] as f32 / 255.0; // G
                input[2 * size * size + base] = pixel[2] as f32 / 255.0; // B
            }
        }

        Ok(input)
    }

    /// Scale points and box prompt from original image coordinates to SAM input coordinates.
    ///
    /// SAM expects box prompts as two additional points with labels:
    /// - Label 2 = top-left corner
    /// - Label 3 = bottom-right corner
    ///
    /// If no prompts are provided, a padding point with label -1 is added.
    fn scale_points_and_box(
        points: &[SamPoint],
        box_prompt: Option<[f32; 4]>,
        original_width: u32,
        original_height: u32,
    ) -> (Vec<f32>, Vec<f32>) {
        let scale_x = SAM_INPUT_SIZE as f32 / original_width as f32;
        let scale_y = SAM_INPUT_SIZE as f32 / original_height as f32;

        let mut coords = Vec::new();
        let mut labels = Vec::new();

        // Add point prompts (label 0 = background, 1 = foreground)
        for point in points {
            coords.push(point.x * scale_x);
            coords.push(point.y * scale_y);
            labels.push(point.label as f32);
        }

        // Add box prompt as two points with labels 2 and 3
        if let Some([x1, y1, x2, y2]) = box_prompt {
            // Top-left corner (label 2)
            coords.push(x1 * scale_x);
            coords.push(y1 * scale_y);
            labels.push(2.0);

            // Bottom-right corner (label 3)
            coords.push(x2 * scale_x);
            coords.push(y2 * scale_y);
            labels.push(3.0);

            log::debug!(
                "Box prompt scaled: ({}, {}) -> ({}, {}) to ({:.1}, {:.1}) -> ({:.1}, {:.1})",
                x1,
                y1,
                x2,
                y2,
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y
            );
        }

        // If no prompts at all, add a padding point with label -1
        if coords.is_empty() {
            coords.push(0.0);
            coords.push(0.0);
            labels.push(-1.0);
            log::warn!("No prompts provided, using padding point");
        }

        (coords, labels)
    }

    /// Post-process mask output to binary mask and polygon.
    ///
    /// SAM 2 ONNX decoder outputs 256x256 logits. We need to:
    /// 1. Bilinear interpolate to 1024x1024 (SAM's working resolution) for smooth edges
    /// 2. Then scale to original image size
    fn postprocess_mask(
        mask_data: &[f32],
        mask_width: usize,
        mask_height: usize,
        original_width: u32,
        original_height: u32,
    ) -> (Vec<u8>, Vec<[f32; 2]>) {
        // Step 1: Bilinear upsample from 256x256 to 1024x1024 (4x)
        // This matches the F.interpolate in PyTorch SAM 2
        let upscale_factor = 4;
        let hi_res_w = mask_width * upscale_factor;
        let hi_res_h = mask_height * upscale_factor;
        let hi_res_mask =
            Self::bilinear_upsample(mask_data, mask_width, mask_height, hi_res_w, hi_res_h);

        // Step 2: Compute contour on a capped working grid (max 1024x1024),
        // then scale vertices back to original image coordinates.
        let contour_w = (original_width as usize).min(MAX_CONTOUR_RESOLUTION);
        let contour_h = (original_height as usize).min(MAX_CONTOUR_RESOLUTION);
        let contour_mask =
            Self::threshold_resample_mask(&hi_res_mask, hi_res_w, hi_res_h, contour_w, contour_h);
        let mut polygon =
            Self::extract_largest_contour(&contour_mask, contour_w as u32, contour_h as u32);

        if contour_w as u32 != original_width || contour_h as u32 != original_height {
            let scale_x = original_width as f32 / contour_w as f32;
            let scale_y = original_height as f32 / contour_h as f32;
            for point in &mut polygon {
                point[0] *= scale_x;
                point[1] *= scale_y;
            }
        }

        // Step 3: Optional full-size binary mask generation.
        // This is disabled by default because protocol consumers currently only
        // use polygons and IoU scores.
        let binary_mask = if EMIT_BINARY_MASKS {
            Self::threshold_resample_mask(
                &hi_res_mask,
                hi_res_w,
                hi_res_h,
                original_width as usize,
                original_height as usize,
            )
        } else {
            Vec::new()
        };

        (binary_mask, polygon)
    }

    /// Threshold and resample a float logit mask into binary 0/255 values.
    ///
    /// Uses nearest-neighbor sampling for speed because this path is used in
    /// CPU postprocessing hot loops.
    fn threshold_resample_mask(
        src: &[f32],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Vec<u8> {
        let mut dst = vec![0u8; pixel_count(dst_w, dst_h)];
        if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
            return dst;
        }

        if src_w == dst_w && src_h == dst_h {
            for (out, &value) in dst.iter_mut().zip(src.iter()) {
                if value > 0.0 {
                    *out = 255;
                }
            }
            return dst;
        }

        let scale_x = src_w as f32 / dst_w as f32;
        let scale_y = src_h as f32 / dst_h as f32;
        let max_x = (src_w.saturating_sub(1)) as isize;
        let max_y = (src_h.saturating_sub(1)) as isize;

        for y in 0..dst_h {
            let src_y = (((y as f32 + 0.5) * scale_y - 0.5).round() as isize).clamp(0, max_y);
            let src_row = src_y as usize * src_w;
            let dst_row = y * dst_w;

            for x in 0..dst_w {
                let src_x = (((x as f32 + 0.5) * scale_x - 0.5).round() as isize).clamp(0, max_x);
                if src[src_row + src_x as usize] > 0.0 {
                    dst[dst_row + x] = 255;
                }
            }
        }

        dst
    }

    /// Bilinear upsample a 2D float array.
    fn bilinear_upsample(
        data: &[f32],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Vec<f32> {
        let mut result = vec![0.0f32; pixel_count(dst_w, dst_h)];

        for dst_y in 0..dst_h {
            for dst_x in 0..dst_w {
                // Map destination to source coordinates (align_corners=False style)
                let src_x = (dst_x as f32 + 0.5) * (src_w as f32 / dst_w as f32) - 0.5;
                let src_y = (dst_y as f32 + 0.5) * (src_h as f32 / dst_h as f32) - 0.5;

                result[dst_y * dst_w + dst_x] = bilinear_sample(data, src_w, src_h, src_x, src_y);
            }
        }

        result
    }

    /// Extract contour of the largest connected component using flood fill + Moore-Neighbor tracing.
    ///
    /// This filters out small noise blobs and only traces the main segmented region.
    fn extract_largest_contour(mask: &[u8], width: u32, height: u32) -> Vec<[f32; 2]> {
        let w = width as usize;
        let h = height as usize;

        // Find all connected components and their sizes using flood fill
        let mut labels = vec![0u32; w * h];
        let mut component_sizes: Vec<usize> = vec![0]; // Index 0 is background
        let mut current_label = 0u32;

        for start_y in 0..h {
            for start_x in 0..w {
                let idx = start_y * w + start_x;
                if mask[idx] == 255 && labels[idx] == 0 {
                    // New component found - flood fill to label it
                    current_label += 1;
                    let mut size = 0usize;
                    let mut stack = vec![(start_x, start_y)];

                    while let Some((x, y)) = stack.pop() {
                        let i = y * w + x;
                        if labels[i] != 0 || mask[i] != 255 {
                            continue;
                        }

                        labels[i] = current_label;
                        size += 1;

                        // Add 4-connected neighbors
                        if x > 0 {
                            stack.push((x - 1, y));
                        }
                        if x < w - 1 {
                            stack.push((x + 1, y));
                        }
                        if y > 0 {
                            stack.push((x, y - 1));
                        }
                        if y < h - 1 {
                            stack.push((x, y + 1));
                        }
                    }

                    component_sizes.push(size);
                }
            }
        }

        if current_label == 0 {
            return Vec::new(); // No components found
        }

        // Find the largest component
        let largest_label = component_sizes
            .iter()
            .enumerate()
            .skip(1) // Skip background (index 0)
            .max_by_key(|(_, size)| **size)
            .map(|(label, _)| label as u32)
            .unwrap_or(1);

        log::debug!(
            "Found {} components, largest has {} pixels",
            current_label,
            component_sizes.get(largest_label as usize).unwrap_or(&0)
        );

        // Create mask with only the largest component
        let largest_mask: Vec<u8> = labels
            .iter()
            .map(|&l| if l == largest_label { 255 } else { 0 })
            .collect();

        // Now trace the contour of the largest component using Moore-Neighbor
        Self::trace_contour(&largest_mask, width, height)
    }

    /// Trace contour using Moore-Neighbor algorithm.
    fn trace_contour(mask: &[u8], width: u32, height: u32) -> Vec<[f32; 2]> {
        let w = width as usize;
        let h = height as usize;

        // Find first boundary pixel (starting point)
        let mut start = None;
        'outer: for y in 0..h {
            for x in 0..w {
                if mask[y * w + x] == 255 {
                    start = Some((x as i32, y as i32));
                    break 'outer;
                }
            }
        }

        let Some((start_x, start_y)) = start else {
            return Vec::new();
        };

        // Moore neighbor offsets (8-connected, clockwise from left)
        let neighbors: [(i32, i32); 8] = [
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
        ];

        let get_pixel = |x: i32, y: i32| -> bool {
            if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                false
            } else {
                mask[(y as usize) * w + (x as usize)] == 255
            }
        };

        let mut contour = Vec::new();
        let mut current = (start_x, start_y);
        let mut backtrack_dir = 4;

        let max_iterations = w * h;
        let mut iterations = 0;

        loop {
            contour.push([current.0 as f32, current.1 as f32]);

            let mut found = false;
            for i in 0..8 {
                let dir = (backtrack_dir + i) % 8;
                let (dx, dy) = neighbors[dir];
                let nx = current.0 + dx;
                let ny = current.1 + dy;

                if get_pixel(nx, ny) {
                    backtrack_dir = (dir + 5) % 8;
                    current = (nx, ny);
                    found = true;
                    break;
                }
            }

            if !found
                || (current.0 == start_x && current.1 == start_y)
                || iterations > max_iterations
            {
                break;
            }

            iterations += 1;
        }

        // Simplify contour using Douglas-Peucker
        if contour.len() > 500 {
            contour = Self::simplify_polygon(&contour, 1.5);
        }

        // Further subsample if still too many
        if contour.len() > 1000 {
            let step = contour.len() / 500;
            contour = contour.into_iter().step_by(step).collect();
        }

        contour
    }

    /// Simplify polygon using Douglas-Peucker algorithm.
    fn simplify_polygon(points: &[[f32; 2]], epsilon: f32) -> Vec<[f32; 2]> {
        if points.len() < 3 {
            return points.to_vec();
        }

        // Find the point with the maximum distance from the line between first and last
        let first = points[0];
        let last = points[points.len() - 1];

        let mut max_dist = 0.0f32;
        let mut max_idx = 0;

        for (i, p) in points.iter().enumerate().skip(1).take(points.len() - 2) {
            let dist = Self::perpendicular_distance(*p, first, last);
            if dist > max_dist {
                max_dist = dist;
                max_idx = i;
            }
        }

        // If max distance is greater than epsilon, recursively simplify
        if max_dist > epsilon {
            let mut result1 = Self::simplify_polygon(&points[..=max_idx], epsilon);
            let result2 = Self::simplify_polygon(&points[max_idx..], epsilon);

            result1.pop(); // Remove duplicate point
            result1.extend(result2);
            result1
        } else {
            vec![first, last]
        }
    }

    /// Calculate perpendicular distance from point to line.
    fn perpendicular_distance(point: [f32; 2], line_start: [f32; 2], line_end: [f32; 2]) -> f32 {
        let dx = line_end[0] - line_start[0];
        let dy = line_end[1] - line_start[1];

        let length_sq = dx * dx + dy * dy;
        if length_sq == 0.0 {
            // Line start and end are the same point
            let pdx = point[0] - line_start[0];
            let pdy = point[1] - line_start[1];
            return (pdx * pdx + pdy * pdy).sqrt();
        }

        let numerator = ((line_end[0] - line_start[0]) * (line_start[1] - point[1])
            - (line_start[0] - point[0]) * (line_end[1] - line_start[1]))
            .abs();

        numerator / length_sq.sqrt()
    }
}

#[async_trait]
impl SamBackend for OnnxSamEngine {
    fn name(&self) -> &'static str {
        "onnx"
    }

    fn provider(&self) -> ExecutionProvider {
        self.provider
    }

    async fn encode_image(
        &self,
        image_rgb: &[u8],
        width: u32,
        height: u32,
    ) -> Result<EncoderOutput> {
        let image_rgb = image_rgb.to_vec();
        let encoder = Arc::clone(&self.encoder);
        let provider = self.provider;
        let request_start = std::time::Instant::now();

        // Run encoding in blocking task
        tokio::task::spawn_blocking(move || {
            // Preprocess image
            let preprocess_start = std::time::Instant::now();
            let input_data = Self::preprocess_image(&image_rgb, width, height)?;
            let preprocess_ms = preprocess_start.elapsed().as_millis();

            // Create tensor with shape (1, 3, 1024, 1024)
            let size = SAM_INPUT_SIZE as i64;
            let input_tensor = Tensor::from_array(([1i64, 3, size, size], input_data))?;

            // Lock the encoder session using blocking_lock() which is safe within spawn_blocking.
            // This prevents thread pool exhaustion by allowing the tokio runtime to manage
            // the lock queue asynchronously.
            let lock_start = std::time::Instant::now();
            let mut encoder_guard = encoder.blocking_lock();
            let lock_wait_ms = lock_start.elapsed().as_millis();

            // Run encoder with named inputs
            let inference_start = std::time::Instant::now();
            let outputs = encoder_guard.run(ort::inputs![
                "image" => input_tensor
            ])?;
            let inference_ms = inference_start.elapsed().as_millis();

            // Extract all three encoder outputs
            // 1. Main image embedding (1, 256, 64, 64)
            let image_embed_value = outputs
                .get("image_embed")
                .context("Missing image_embed output")?;
            let (_, image_embed_data) = image_embed_value.try_extract_tensor::<f32>()?;

            // 2. High resolution features 0 (1, 32, 256, 256)
            let high_res_0_value = outputs
                .get("high_res_feats_0")
                .context("Missing high_res_feats_0 output")?;
            let (_, high_res_0_data) = high_res_0_value.try_extract_tensor::<f32>()?;

            // 3. High resolution features 1 (1, 64, 128, 128)
            let high_res_1_value = outputs
                .get("high_res_feats_1")
                .context("Missing high_res_feats_1 output")?;
            let (_, high_res_1_data) = high_res_1_value.try_extract_tensor::<f32>()?;

            log::debug!(
                "Encoder outputs: image_embed={}, high_res_0={}, high_res_1={}",
                image_embed_data.len(),
                high_res_0_data.len(),
                high_res_1_data.len()
            );

            log::info!(
                "SAM encode_image timing [{}]: preprocess={}ms, lock_wait={}ms, inference={}ms, total={}ms ({}x{} -> {SAM_INPUT_SIZE}x{SAM_INPUT_SIZE})",
                provider.name(),
                preprocess_ms,
                lock_wait_ms,
                inference_ms,
                request_start.elapsed().as_millis(),
                width,
                height,
            );

            Ok(EncoderOutput::sam2_onnx(
                image_embed_data.to_vec(),
                high_res_0_data.to_vec(),
                high_res_1_data.to_vec(),
            ))
        })
        .await?
    }

    #[allow(clippy::too_many_lines)]
    async fn decode_mask(
        &self,
        encoder_output: &EncoderOutput,
        original_width: u32,
        original_height: u32,
        points: &[SamPoint],
        box_prompt: Option<[f32; 4]>,
    ) -> Result<SamMaskResult> {
        let encoder_output = encoder_output.clone();
        let points = points.to_vec();
        let decoder = Arc::clone(&self.decoder);
        let provider = self.provider;
        let request_start = std::time::Instant::now();

        tokio::task::spawn_blocking(move || {
            let sam2_state = match encoder_output {
                EncoderOutput::Sam2Onnx(state) => state,
                other => {
                    anyhow::bail!(
                        "Unsupported encoded state '{}' for SAM2 ONNX decoder",
                        other.kind()
                    );
                }
            };

            // Prepare point inputs (including box prompt as label 2/3 points)
            let prep_start = std::time::Instant::now();
            let (point_coords, point_labels) =
                Self::scale_points_and_box(&points, box_prompt, original_width, original_height);

            let num_points = point_labels.len() as i64;

            // Create image embedding tensor (1, 256, 64, 64)
            let image_embed_tensor =
                Tensor::from_array(([1i64, 256, 64, 64], sam2_state.image_embed))?;

            // Create high resolution feature tensors
            // high_res_feats_0: (1, 32, 256, 256)
            let high_res_0_tensor =
                Tensor::from_array(([1i64, 32, 256, 256], sam2_state.high_res_feats_0))?;

            // high_res_feats_1: (1, 64, 128, 128)
            let high_res_1_tensor =
                Tensor::from_array(([1i64, 64, 128, 128], sam2_state.high_res_feats_1))?;

            // Create point tensors
            let coords_tensor = Tensor::from_array(([1i64, num_points, 2], point_coords))?;
            let labels_tensor = Tensor::from_array(([1i64, num_points], point_labels))?;

            // Prepare mask input (no previous mask)
            let mask_input = vec![0.0f32; 256 * 256];
            let mask_tensor = Tensor::from_array(([1i64, 1, 256, 256], mask_input))?;

            let has_mask = vec![0.0f32];
            let has_mask_tensor = Tensor::from_array(([1i64], has_mask))?;
            let prep_ms = prep_start.elapsed().as_millis();

            // Lock the decoder session using blocking_lock() which is safe within spawn_blocking.
            // This prevents thread pool exhaustion by allowing the tokio runtime to manage
            // the lock queue asynchronously.
            let lock_start = std::time::Instant::now();
            let mut decoder_guard = decoder.blocking_lock();
            let lock_wait_ms = lock_start.elapsed().as_millis();

            // Run decoder with all required inputs
            let decoder_start = std::time::Instant::now();
            let outputs = decoder_guard.run(ort::inputs![
                "image_embed" => image_embed_tensor,
                "high_res_feats_0" => high_res_0_tensor,
                "high_res_feats_1" => high_res_1_tensor,
                "point_coords" => coords_tensor,
                "point_labels" => labels_tensor,
                "mask_input" => mask_tensor,
                "has_mask_input" => has_mask_tensor
            ])?;
            let decoder_ms = decoder_start.elapsed().as_millis();

            // Extract masks and scores
            let masks_value = outputs.get("masks").context("Missing masks output")?;
            let scores_value = outputs
                .get("iou_predictions")
                .context("Missing iou_predictions output")?;

            let (masks_shape, masks_data) = masks_value.try_extract_tensor::<f32>()?;
            let (_, scores_data) = scores_value.try_extract_tensor::<f32>()?;

            // masks_shape is typically [1, num_masks, H, W]
            let shape_dims: Vec<i64> = masks_shape.iter().copied().collect();
            let num_masks = if shape_dims.len() >= 2 {
                shape_dims[1] as usize
            } else {
                1
            };
            let mask_h = if shape_dims.len() >= 3 {
                shape_dims[2] as usize
            } else {
                256
            };
            let mask_w = if shape_dims.len() >= 4 {
                shape_dims[3] as usize
            } else {
                256
            };

            log::debug!(
                "Decoder output: {} masks of size {}x{} (onnx run {}ms)",
                num_masks,
                mask_w,
                mask_h,
                decoder_ms
            );

            let mask_pixels = mask_h * mask_w;

            // Process each mask prediction
            let mut result_masks = Vec::new();
            let mut result_polygons = Vec::new();
            let mut result_scores = Vec::new();

            let postprocess_start = std::time::Instant::now();
            for (i, score) in scores_data.iter().copied().enumerate().take(num_masks) {
                let mask_offset = i * mask_pixels;
                let mask_slice = &masks_data[mask_offset..mask_offset + mask_pixels];

                let mask_post_start = std::time::Instant::now();
                let (binary_mask, polygon) = Self::postprocess_mask(
                    mask_slice,
                    mask_w,
                    mask_h,
                    original_width,
                    original_height,
                );
                let mask_post_ms = mask_post_start.elapsed().as_millis();
                log::debug!(
                    "Postprocessed SAM mask {} in {}ms ({} polygon points)",
                    i,
                    mask_post_ms,
                    polygon.len()
                );

                result_masks.push(binary_mask);
                result_polygons.push(polygon);
                result_scores.push(score);
            }

            let postprocess_ms = postprocess_start.elapsed().as_millis();
            log::debug!(
                "SAM mask postprocess complete: {} masks in {}ms",
                num_masks,
                postprocess_ms
            );

            log::info!(
                "SAM decode_mask timing [{}]: prep={}ms, lock_wait={}ms, inference={}ms, postprocess={}ms, total={}ms ({} points)",
                provider.name(),
                prep_ms,
                lock_wait_ms,
                decoder_ms,
                postprocess_ms,
                request_start.elapsed().as_millis(),
                num_points,
            );

            Ok(SamMaskResult {
                masks: result_masks,
                polygons: result_polygons,
                iou_scores: result_scores,
                width: original_width,
                height: original_height,
            })
        })
        .await?
    }

    fn is_available(&self) -> bool {
        true // If we got here, the session was created successfully
    }
}

/// Try to auto-detect the best available execution provider.
pub fn detect_best_provider() -> ExecutionProvider {
    fn provider_available<T: OrtExecutionProviderTrait>(provider: &T) -> bool {
        if !provider.supported_by_platform() {
            return false;
        }
        match provider.is_available() {
            Ok(v) => v,
            Err(e) => {
                log::warn!(
                    "Provider availability check failed for {}: {}",
                    provider.name(),
                    e
                );
                false
            }
        }
    }

    if provider_available(&CUDAExecutionProvider::default()) {
        log::info!("Auto-detected CUDA execution provider");
        return ExecutionProvider::Cuda;
    }
    // WebGPU is the vendor-neutral GPU fallback (no cuDNN/TensorRT). Preferred
    // over the platform-specific providers below when CUDA is unavailable so
    // that any GPU exposing Vulkan/D3D12 still gets hardware acceleration.
    if provider_available(&WebGPUExecutionProvider::default()) {
        log::info!("Auto-detected WebGPU execution provider");
        return ExecutionProvider::WebGpu;
    }
    if provider_available(&ROCmExecutionProvider::default()) {
        log::info!("Auto-detected ROCm execution provider");
        return ExecutionProvider::Rocm;
    }
    if provider_available(&DirectMLExecutionProvider::default()) {
        log::info!("Auto-detected DirectML execution provider");
        return ExecutionProvider::DirectML;
    }
    if provider_available(&CoreMLExecutionProvider::default()) {
        log::info!("Auto-detected CoreML execution provider");
        return ExecutionProvider::CoreML;
    }

    log::info!("Falling back to CPU execution provider");
    ExecutionProvider::Cpu
}

/// Tests that load the real SAM ONNX models and run actual inference.
///
/// These are `#[ignore]`d by default because they require the SAM Tiny
/// model files (~150MB) to be present at `.cache/models/` and can take a
/// long time on CPU. Run explicitly with:
///
/// ```text
/// cargo test -p hvat_backend --features sam-cuda --lib \
///     full_sam::sam::engine::real_model_tests -- --ignored --nocapture
/// ```
///
/// The `--nocapture` flag is required to see the `SAM encode_image timing`
/// / `SAM decode_mask timing` log lines emitted by this module.
#[cfg(test)]
mod real_model_tests {
    use super::*;
    use crate::sam::{SamBackend, SamPoint};
    use std::path::Path;

    /// Generate a deterministic synthetic RGB image (no external file deps).
    fn synth_image(width: u32, height: u32, seed: u8) -> Vec<u8> {
        let mut data = vec![0u8; (width * height * 3) as usize];
        for (i, px) in data.chunks_mut(3).enumerate() {
            px[0] = (i as u32).wrapping_add(seed as u32) as u8;
            px[1] = (i as u32 * 2).wrapping_add(seed as u32) as u8;
            px[2] = (i as u32 * 3).wrapping_add(seed as u32) as u8;
        }
        data
    }

    /// Deterministic image with a clear foreground object: a filled bright
    /// rectangle on a dark background. Unlike pure noise, this yields a
    /// well-defined SAM mask whose boundary is robust to sub-threshold
    /// numerical noise — the right probe for comparing provider correctness.
    fn synth_shape_image(width: u32, height: u32, rect: (u32, u32, u32, u32)) -> Vec<u8> {
        let (rx0, ry0, rx1, ry1) = rect;
        let mut data = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let inside = x >= rx0 && x < rx1 && y >= ry0 && y < ry1;
                let base = ((y * width + x) * 3) as usize;
                let v = if inside { 235 } else { 20 };
                data[base] = v;
                data[base + 1] = v;
                data[base + 2] = v;
            }
        }
        data
    }

    /// Max and mean absolute element-wise difference between two equal-length
    /// slices. Panics on length mismatch (a shape bug we want to surface).
    fn diff_stats(reference: &[f32], candidate: &[f32]) -> (f32, f32) {
        assert_eq!(
            reference.len(),
            candidate.len(),
            "embedding length mismatch: reference {} vs candidate {}",
            reference.len(),
            candidate.len()
        );
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        for (r, c) in reference.iter().zip(candidate.iter()) {
            let d = (r - c).abs();
            max_abs = max_abs.max(d);
            sum_abs += f64::from(d);
        }
        let mean_abs = (sum_abs / reference.len() as f64) as f32;
        (max_abs, mean_abs)
    }

    /// Verify the GPU execution provider produces the same encoder embeddings and
    /// decoder mask as CPU (the reference). CPU is available in every build, so
    /// this compares CPU vs whichever GPU provider `detect_best_provider` selects
    /// (WebGPU or CUDA depending on the build). Skips when no GPU is available.
    #[tokio::test]
    #[ignore = "requires real SAM ONNX models in .cache/models and is slow"]
    #[allow(clippy::too_many_lines)]
    async fn gpu_matches_cpu_reference() {
        let _ = env_logger::Builder::new()
            .filter_level(log::LevelFilter::Warn)
            .is_test(true)
            .try_init();

        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crate dir has a parent")
            .join(".cache/models");
        assert!(
            model_dir.join(SamVariant::Tiny.encoder_filename()).exists(),
            "missing SAM Tiny models in {}",
            model_dir.display()
        );

        let gpu_provider = detect_best_provider();
        if gpu_provider == ExecutionProvider::Cpu {
            eprintln!("no GPU provider available; skipping CPU/GPU comparison");
            return;
        }

        let cpu = OnnxSamEngine::new(&model_dir, SamVariant::Tiny, ExecutionProvider::Cpu)
            .expect("build CPU engine");
        let gpu = OnnxSamEngine::new(&model_dir, SamVariant::Tiny, gpu_provider)
            .expect("build GPU engine");
        assert_eq!(
            gpu.provider(),
            gpu_provider,
            "GPU provider {} failed to initialize (fell back to {})",
            gpu_provider.name(),
            gpu.provider().name()
        );

        let (w, h) = (800, 600);
        // Two distinct images with clear, well-defined foreground objects (so the
        // mask boundary is stable), encoded back-to-back on the same GPU engine.
        // The back-to-back pattern also catches cross-call staleness (e.g. graph
        // capture replaying the previous image's computation).
        // (label, image, prompt-point-at-object-center)
        let images = [
            (
                "A",
                synth_shape_image(w, h, (250, 150, 550, 450)),
                [400.0, 300.0],
            ),
            (
                "B",
                synth_shape_image(w, h, (100, 100, 300, 300)),
                [200.0, 200.0],
            ),
        ];

        let mut max_embed_diff = 0.0f32;
        let mut cpu_absmax_all = 1.0f32;
        let mut max_bbox_diff = 0.0f32;
        let mut gpu_embeds = Vec::new();

        for (label, rgb, pt) in &images {
            let point = [SamPoint {
                x: pt[0],
                y: pt[1],
                label: 1,
            }];
            let cpu_embed = cpu.encode_image(rgb, w, h).await.expect("cpu encode");
            let gpu_embed = gpu.encode_image(rgb, w, h).await.expect("gpu encode");

            let cpu_s = cpu_embed.as_sam2_onnx().expect("cpu sam2 state");
            let gpu_s = gpu_embed.as_sam2_onnx().expect("gpu sam2 state");

            let (img_max, img_mean) = diff_stats(&cpu_s.image_embed, &gpu_s.image_embed);
            let (h0_max, _) = diff_stats(&cpu_s.high_res_feats_0, &gpu_s.high_res_feats_0);
            let (h1_max, _) = diff_stats(&cpu_s.high_res_feats_1, &gpu_s.high_res_feats_1);
            let cpu_absmax = cpu_s.image_embed.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            cpu_absmax_all = cpu_absmax_all.max(cpu_absmax);
            max_embed_diff = max_embed_diff.max(img_max).max(h0_max).max(h1_max);

            let cpu_mask = cpu
                .decode_mask(&cpu_embed, w, h, &point, None)
                .await
                .expect("cpu decode");
            let gpu_mask = gpu
                .decode_mask(&gpu_embed, w, h, &point, None)
                .await
                .expect("gpu decode");

            println!("\n=== {} vs CPU, image {label} ===", gpu_provider.name());
            println!(
                "image_embed      : max_abs={img_max:.5}  mean_abs={img_mean:.6}  (cpu |max|={cpu_absmax:.3})"
            );
            println!("high_res_feats   : max_abs h0={h0_max:.5}  h1={h1_max:.5}");
            // Bounding box of the largest polygon — tells us whether the mask is
            // in the same place/shape (vs just a noisier contour of it).
            let bbox = |poly: &[[f32; 2]]| -> (f32, f32, f32, f32) {
                poly.iter().fold(
                    (f32::MAX, f32::MAX, f32::MIN, f32::MIN),
                    |(x0, y0, x1, y1), p| (x0.min(p[0]), y0.min(p[1]), x1.max(p[0]), y1.max(p[1])),
                )
            };
            let cpu_poly = cpu_mask.polygons.first().cloned().unwrap_or_default();
            let gpu_poly = gpu_mask.polygons.first().cloned().unwrap_or_default();
            let cb = bbox(&cpu_poly);
            let gb = bbox(&gpu_poly);
            println!(
                "decode iou       : cpu={:.4}  gpu={:.4}",
                cpu_mask.iou_scores.first().copied().unwrap_or(0.0),
                gpu_mask.iou_scores.first().copied().unwrap_or(0.0),
            );
            println!(
                "polygon pts      : cpu={}  gpu={}",
                cpu_poly.len(),
                gpu_poly.len()
            );
            println!(
                "polygon bbox cpu : [{:.0},{:.0} -> {:.0},{:.0}]",
                cb.0, cb.1, cb.2, cb.3
            );
            println!(
                "polygon bbox gpu : [{:.0},{:.0} -> {:.0},{:.0}]",
                gb.0, gb.1, gb.2, gb.3
            );

            // Both polygons must be non-empty and cover the same region. An empty
            // GPU polygon or a wildly different bbox means a corrupted mask.
            assert!(
                !cpu_poly.is_empty() && !gpu_poly.is_empty(),
                "image {label}: empty polygon (cpu={}, gpu={})",
                cpu_poly.len(),
                gpu_poly.len()
            );
            let bbox_diff = (cb.0 - gb.0)
                .abs()
                .max((cb.1 - gb.1).abs())
                .max((cb.2 - gb.2).abs())
                .max((cb.3 - gb.3).abs());
            max_bbox_diff = max_bbox_diff.max(bbox_diff);

            gpu_embeds.push(gpu_embed);
        }

        // Sanity: the two GPU embeddings must differ from each other — if they
        // are (nearly) identical the engine is returning a stale cached result
        // for the second image instead of recomputing it.
        let g0 = gpu_embeds[0].as_sam2_onnx().unwrap();
        let g1 = gpu_embeds[1].as_sam2_onnx().unwrap();
        let (cross_max, _) = diff_stats(&g0.image_embed, &g1.image_embed);
        println!("\nGPU image A vs B embed diff: max_abs={cross_max:.5} (must be large)");

        // Tolerance: fp32 across backends differs slightly, but a correct provider
        // matches CPU closely. A large diff means wrong/stale results.
        let tol = 0.05 * cpu_absmax_all;
        assert!(
            max_embed_diff < tol,
            "{} embeddings differ from CPU by {max_embed_diff:.5} (tol {tol:.5}) — provider is computing incorrect embeddings",
            gpu_provider.name()
        );
        assert!(
            cross_max > tol,
            "GPU embeddings for two different images are nearly identical (diff {cross_max:.5}) — provider is returning stale results across calls"
        );

        // The decoded mask must land in the same place as CPU. A corrupted mask
        // (e.g. the WebGPU decoder returning all-foreground) blows the bbox out to
        // the full image, far beyond this tolerance.
        let bbox_tol = 24.0;
        assert!(
            max_bbox_diff < bbox_tol,
            "GPU mask bbox differs from CPU by {max_bbox_diff:.0}px (tol {bbox_tol:.0}px) — provider is computing incorrect masks"
        );
    }

    #[tokio::test]
    #[ignore = "requires real SAM ONNX models in .cache/models and is slow"]
    async fn real_engine_multiple_embeddings_and_segmentations() {
        let _ = env_logger::Builder::new()
            .filter_level(log::LevelFilter::Info)
            .is_test(true)
            .try_init();

        // Models live in `.cache/models` at the workspace root, but `cargo test`
        // runs with the crate directory as the working directory.
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crate dir has a parent")
            .join(".cache/models");
        let encoder_path = model_dir.join(SamVariant::Tiny.encoder_filename());
        assert!(
            encoder_path.exists(),
            "missing {} - download the SAM Tiny model first",
            encoder_path.display()
        );

        let provider = detect_best_provider();
        let engine = OnnxSamEngine::new(&model_dir, SamVariant::Tiny, provider)
            .expect("failed to load SAM engine");

        // Image A: 1 embedding + 2 segmentations
        let (width_a, height_a) = (640, 480);
        let rgb_a = synth_image(width_a, height_a, 10);
        let embed_a = engine
            .encode_image(&rgb_a, width_a, height_a)
            .await
            .expect("encode image A");

        for [x, y] in [[100.0, 100.0], [300.0, 200.0]] {
            engine
                .decode_mask(
                    &embed_a,
                    width_a,
                    height_a,
                    &[SamPoint { x, y, label: 1 }],
                    None,
                )
                .await
                .expect("decode image A");
        }

        // Image B: 1 new embedding + 3 segmentations
        let (width_b, height_b) = (800, 600);
        let rgb_b = synth_image(width_b, height_b, 99);
        let embed_b = engine
            .encode_image(&rgb_b, width_b, height_b)
            .await
            .expect("encode image B");

        for [x, y] in [[50.0, 50.0], [400.0, 300.0], [700.0, 500.0]] {
            engine
                .decode_mask(
                    &embed_b,
                    width_b,
                    height_b,
                    &[SamPoint { x, y, label: 1 }],
                    None,
                )
                .await
                .expect("decode image B");
        }
    }

    /// Min / median / mean of a set of millisecond samples.
    fn summarize(samples: &[f64]) -> (f64, f64, f64) {
        if samples.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min = sorted[0];
        let median = sorted[sorted.len() / 2];
        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        (min, median, mean)
    }

    /// Timed samples for one execution provider.
    struct ProviderBench {
        encode_ms: Vec<f64>,
        decode_ms: Vec<f64>,
    }

    /// Benchmark a single provider end-to-end through the real engine.
    ///
    /// Returns `Err(reason)` if the provider could not be used — either it is
    /// not compiled into this build, or it failed to initialize on the host
    /// (e.g. no GPU/driver, or a missing shared library). The engine falls back
    /// to CPU internally in that case, which we detect via a provider mismatch
    /// and surface as an explicit error rather than silently benchmarking CPU
    /// under the wrong label.
    async fn bench_provider(
        model_dir: &Path,
        provider: ExecutionProvider,
        warmup: usize,
        iters: usize,
    ) -> Result<ProviderBench, String> {
        let engine = OnnxSamEngine::new(model_dir, SamVariant::Tiny, provider)
            .map_err(|e| format!("engine init failed: {e:#}"))?;

        if engine.provider() != provider {
            return Err(format!(
                "requested {} but engine initialized on {} — {} unavailable \
                 in this build/environment (not compiled in, or no GPU/driver/library)",
                provider.name(),
                engine.provider().name(),
                provider.name(),
            ));
        }

        let (width, height) = (800, 600);
        let rgb = synth_image(width, height, 99);
        let point = [SamPoint {
            x: 400.0,
            y: 300.0,
            label: 1,
        }];

        // Warmup: absorb first-call costs (graph capture, kernel JIT, allocator
        // priming) so the measured samples reflect steady-state latency.
        let mut embed = engine
            .encode_image(&rgb, width, height)
            .await
            .map_err(|e| format!("warmup encode failed: {e:#}"))?;
        for _ in 0..warmup {
            embed = engine
                .encode_image(&rgb, width, height)
                .await
                .map_err(|e| format!("warmup encode failed: {e:#}"))?;
            engine
                .decode_mask(&embed, width, height, &point, None)
                .await
                .map_err(|e| format!("warmup decode failed: {e:#}"))?;
        }

        // Timed encoder passes (each call re-runs the full encoder).
        let mut encode_ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = std::time::Instant::now();
            embed = engine
                .encode_image(&rgb, width, height)
                .await
                .map_err(|e| format!("encode failed: {e:#}"))?;
            encode_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        // Timed decoder passes reusing the last embedding (the real per-click path).
        let mut decode_ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = std::time::Instant::now();
            engine
                .decode_mask(&embed, width, height, &point, None)
                .await
                .map_err(|e| format!("decode failed: {e:#}"))?;
            decode_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        Ok(ProviderBench {
            encode_ms,
            decode_ms,
        })
    }

    /// Benchmark CPU, WebGPU, and CUDA on the same SAM2 Tiny model and image.
    ///
    /// All three providers are attempted. CPU is always available; WebGPU and
    /// CUDA are optional and report an explicit `UNAVAILABLE` line (instead of
    /// failing the test) when they are not compiled into this build or cannot
    /// initialize on the host. Because the WebGPU and CUDA prebuilt ONNX Runtime
    /// binaries are mutually exclusive, a single build can measure CPU plus at
    /// most one GPU provider — run once per feature to cover all three:
    ///
    /// ```text
    /// # CPU + WebGPU (CUDA reported UNAVAILABLE):
    /// cargo test -p hvat_backend --profile release-dev --features sam-webgpu --lib \
    ///     full_sam::sam::engine::real_model_tests::benchmark_sam_providers \
    ///     -- --ignored --nocapture
    ///
    /// # CPU + CUDA (WebGPU reported UNAVAILABLE; needs libcudnn.so.9 on the loader path):
    /// cargo test -p hvat_backend --profile release-dev --features sam-cuda --lib \
    ///     full_sam::sam::engine::real_model_tests::benchmark_sam_providers \
    ///     -- --ignored --nocapture
    /// ```
    ///
    /// `scripts/bench_sam_providers.sh` runs both builds and aggregates the output.
    #[tokio::test]
    #[ignore = "requires real SAM ONNX models in .cache/models and is slow"]
    async fn benchmark_sam_providers() {
        let _ = env_logger::Builder::new()
            .filter_level(log::LevelFilter::Warn)
            .is_test(true)
            .try_init();

        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crate dir has a parent")
            .join(".cache/models");
        let encoder_path = model_dir.join(SamVariant::Tiny.encoder_filename());
        assert!(
            encoder_path.exists(),
            "missing {} - download the SAM Tiny model first",
            encoder_path.display()
        );

        let warmup = 2;
        let iters = 10;

        println!(
            "\n=== SAM provider benchmark (SAM2 Tiny, 800x600, warmup={warmup}, iters={iters}) ==="
        );
        println!(
            "{:<8} | {:>26} | {:>26}",
            "provider", "encode ms (min/med/mean)", "decode ms (min/med/mean)"
        );
        println!("{:-<8}-+-{:-<26}-+-{:-<26}", "", "", "");

        let mut cpu_ok = false;
        for provider in [
            ExecutionProvider::Cpu,
            ExecutionProvider::WebGpu,
            ExecutionProvider::Cuda,
        ] {
            match bench_provider(&model_dir, provider, warmup, iters).await {
                Ok(bench) => {
                    if provider == ExecutionProvider::Cpu {
                        cpu_ok = true;
                    }
                    let (e_min, e_med, e_mean) = summarize(&bench.encode_ms);
                    let (d_min, d_med, d_mean) = summarize(&bench.decode_ms);
                    println!(
                        "{:<8} | {:>8.0} /{:>7.0} /{:>7.0} | {:>8.0} /{:>7.0} /{:>7.0}",
                        provider.name(),
                        e_min,
                        e_med,
                        e_mean,
                        d_min,
                        d_med,
                        d_mean,
                    );
                }
                Err(reason) => {
                    println!("{:<8} | UNAVAILABLE — {reason}", provider.name());
                }
            }
        }
        println!();

        // CPU is the always-available baseline; if it can't run, the benchmark
        // itself is broken (missing models, etc.) and should fail loudly.
        assert!(cpu_ok, "CPU provider benchmark did not run");
    }
}
