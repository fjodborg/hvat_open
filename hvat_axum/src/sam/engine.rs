//! ONNX Runtime-based SAM engine implementation.
//!
//! Uses the `ort` crate for cross-platform ONNX inference with
//! support for multiple execution providers (CUDA, ROCm, DirectML, CoreML, CPU).

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use async_trait::async_trait;
use ort::session::Session;
use ort::value::Tensor;

use super::backend::{EncoderOutput, ExecutionProvider, SamBackend, SamMaskResult, SamPoint};
use super::models::SamVariant;

/// SAM 2 input image size (model expects 1024x1024).
const SAM_INPUT_SIZE: u32 = 1024;

/// ONNX Runtime-based SAM engine.
///
/// Manages encoder and decoder sessions for SAM 2 inference.
/// Sessions are wrapped in Mutex because ONNX Runtime's `run` requires `&mut self`.
pub struct OnnxSamEngine {
    encoder: Arc<Mutex<Session>>,
    decoder: Arc<Mutex<Session>>,
    provider: ExecutionProvider,
    #[allow(dead_code)]
    variant: SamVariant,
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

        // Build session with execution provider
        let encoder =
            Self::build_session(&encoder_path, provider).context("Failed to load encoder model")?;
        let decoder =
            Self::build_session(&decoder_path, provider).context("Failed to load decoder model")?;

        log::info!("SAM engine initialized successfully");

        Ok(Self {
            encoder: Arc::new(Mutex::new(encoder)),
            decoder: Arc::new(Mutex::new(decoder)),
            provider,
            variant,
        })
    }

    /// Build an ONNX session with the specified execution provider.
    fn build_session(model_path: &Path, provider: ExecutionProvider) -> Result<Session> {
        let builder = Session::builder()?;

        // Configure execution providers based on selection
        // Note: Features must be enabled in Cargo.toml for each provider
        // For now, we just use CPU - GPU providers require feature flags
        match provider {
            ExecutionProvider::Cpu => {
                // CPU is always available, no additional setup needed
            }
            _ => {
                log::warn!(
                    "{} provider requested but GPU features not enabled, falling back to CPU",
                    provider.name()
                );
            }
        }

        let session = builder.commit_from_file(model_path)?;
        Ok(session)
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
        let mut input = vec![0.0f32; 1 * 3 * size * size];

        for (y, row) in resized.enumerate_rows() {
            for (x, _, pixel) in row {
                let base = y as usize * size + x as usize;
                input[0 * size * size + base] = pixel[0] as f32 / 255.0; // R
                input[1 * size * size + base] = pixel[1] as f32 / 255.0; // G
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
                x1, y1, x2, y2,
                x1 * scale_x, y1 * scale_y,
                x2 * scale_x, y2 * scale_y
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

        // Step 2: Threshold and scale to original image size
        let mut binary_mask = vec![0u8; (original_width * original_height) as usize];

        let scale_x = hi_res_w as f32 / original_width as f32;
        let scale_y = hi_res_h as f32 / original_height as f32;

        for y in 0..original_height {
            for x in 0..original_width {
                // Bilinear sample from hi-res mask for smooth edges
                let src_x = x as f32 * scale_x;
                let src_y = y as f32 * scale_y;

                let value = Self::bilinear_sample(&hi_res_mask, hi_res_w, hi_res_h, src_x, src_y);

                // Threshold at 0 (logits)
                if value > 0.0 {
                    binary_mask[(y * original_width + x) as usize] = 255;
                }
            }
        }

        // Extract contour from largest connected component only
        let polygon = Self::extract_largest_contour(&binary_mask, original_width, original_height);

        (binary_mask, polygon)
    }

    /// Bilinear upsample a 2D float array.
    fn bilinear_upsample(
        data: &[f32],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Vec<f32> {
        let mut result = vec![0.0f32; dst_w * dst_h];

        for dst_y in 0..dst_h {
            for dst_x in 0..dst_w {
                // Map destination to source coordinates (align_corners=False style)
                let src_x = (dst_x as f32 + 0.5) * (src_w as f32 / dst_w as f32) - 0.5;
                let src_y = (dst_y as f32 + 0.5) * (src_h as f32 / dst_h as f32) - 0.5;

                result[dst_y * dst_w + dst_x] =
                    Self::bilinear_sample(data, src_w, src_h, src_x, src_y);
            }
        }

        result
    }

    /// Sample a value from a 2D array using bilinear interpolation.
    fn bilinear_sample(data: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
        let x0 = (x.floor() as isize).clamp(0, width as isize - 1) as usize;
        let y0 = (y.floor() as isize).clamp(0, height as isize - 1) as usize;
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);

        let fx = (x - x.floor()).clamp(0.0, 1.0);
        let fy = (y - y.floor()).clamp(0.0, 1.0);

        let v00 = data[y0 * width + x0];
        let v10 = data[y0 * width + x1];
        let v01 = data[y1 * width + x0];
        let v11 = data[y1 * width + x1];

        let v0 = v00 * (1.0 - fx) + v10 * fx;
        let v1 = v01 * (1.0 - fx) + v11 * fx;

        v0 * (1.0 - fy) + v1 * fy
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

        // Run encoding in blocking task
        tokio::task::spawn_blocking(move || {
            // Preprocess image
            let input_data = Self::preprocess_image(&image_rgb, width, height)?;

            // Create tensor with shape (1, 3, 1024, 1024)
            let size = SAM_INPUT_SIZE as i64;
            let input_tensor = Tensor::from_array(([1i64, 3, size, size], input_data))?;

            // Lock the encoder session
            let mut encoder_guard = encoder
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock encoder: {}", e))?;

            // Run encoder with named inputs
            let outputs = encoder_guard.run(ort::inputs![
                "image" => input_tensor
            ])?;

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

            Ok(EncoderOutput {
                image_embed: image_embed_data.to_vec(),
                high_res_feats_0: high_res_0_data.to_vec(),
                high_res_feats_1: high_res_1_data.to_vec(),
            })
        })
        .await?
    }

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

        tokio::task::spawn_blocking(move || {
            // Prepare point inputs (including box prompt as label 2/3 points)
            let (point_coords, point_labels) =
                Self::scale_points_and_box(&points, box_prompt, original_width, original_height);

            let num_points = point_labels.len() as i64;

            // Create image embedding tensor (1, 256, 64, 64)
            let image_embed_tensor =
                Tensor::from_array(([1i64, 256, 64, 64], encoder_output.image_embed))?;

            // Create high resolution feature tensors
            // high_res_feats_0: (1, 32, 256, 256)
            let high_res_0_tensor =
                Tensor::from_array(([1i64, 32, 256, 256], encoder_output.high_res_feats_0))?;

            // high_res_feats_1: (1, 64, 128, 128)
            let high_res_1_tensor =
                Tensor::from_array(([1i64, 64, 128, 128], encoder_output.high_res_feats_1))?;

            // Create point tensors
            let coords_tensor = Tensor::from_array(([1i64, num_points, 2], point_coords))?;
            let labels_tensor = Tensor::from_array(([1i64, num_points], point_labels))?;

            // Prepare mask input (no previous mask)
            let mask_input = vec![0.0f32; 1 * 1 * 256 * 256];
            let mask_tensor = Tensor::from_array(([1i64, 1, 256, 256], mask_input))?;

            let has_mask = vec![0.0f32];
            let has_mask_tensor = Tensor::from_array(([1i64], has_mask))?;

            // Lock the decoder session
            let mut decoder_guard = decoder
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock decoder: {}", e))?;

            // Run decoder with all required inputs
            let outputs = decoder_guard.run(ort::inputs![
                "image_embed" => image_embed_tensor,
                "high_res_feats_0" => high_res_0_tensor,
                "high_res_feats_1" => high_res_1_tensor,
                "point_coords" => coords_tensor,
                "point_labels" => labels_tensor,
                "mask_input" => mask_tensor,
                "has_mask_input" => has_mask_tensor
            ])?;

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
                "Decoder output: {} masks of size {}x{}",
                num_masks,
                mask_w,
                mask_h
            );

            let mask_pixels = mask_h * mask_w;

            // Process each mask prediction
            let mut result_masks = Vec::new();
            let mut result_polygons = Vec::new();
            let mut result_scores = Vec::new();

            for i in 0..num_masks {
                let mask_offset = i * mask_pixels;
                let mask_slice = &masks_data[mask_offset..mask_offset + mask_pixels];

                let (binary_mask, polygon) = Self::postprocess_mask(
                    mask_slice,
                    mask_w,
                    mask_h,
                    original_width,
                    original_height,
                );

                result_masks.push(binary_mask);
                result_polygons.push(polygon);
                result_scores.push(scores_data[i]);
            }

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
    // For now, just return CPU since GPU features require cargo features
    // In the future, we can add runtime detection
    log::info!("Using CPU execution provider (GPU requires feature flags)");
    ExecutionProvider::Cpu
}
