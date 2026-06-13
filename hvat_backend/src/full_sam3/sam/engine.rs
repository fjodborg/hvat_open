//! SAM3 compatibility engine wrapper.
//!
//! This wrapper gives `full_sam3` its own backend type and module boundary,
//! while delegating inference to the proven SAM2 ONNX engine implementation.

use std::path::Path;

use async_trait::async_trait;

use crate::full_sam::sam::{
    EncoderOutput, ExecutionProvider, OnnxSamEngine, SamBackend, SamMaskResult, SamPoint,
    SamVariant,
};

/// Temporary SAM3 backend engine that proxies to the SAM2 ONNX runtime.
pub struct Sam3CompatOnnxEngine {
    inner: OnnxSamEngine,
}

impl Sam3CompatOnnxEngine {
    pub fn new(
        model_dir: &Path,
        variant: SamVariant,
        provider: ExecutionProvider,
    ) -> anyhow::Result<Self> {
        let inner = OnnxSamEngine::new(model_dir, variant, provider)?;
        Ok(Self { inner })
    }
}

#[async_trait]
impl SamBackend for Sam3CompatOnnxEngine {
    fn name(&self) -> &'static str {
        "sam3-compat-onnx"
    }

    fn provider(&self) -> ExecutionProvider {
        self.inner.provider()
    }

    async fn encode_image(
        &self,
        image_rgb: &[u8],
        width: u32,
        height: u32,
    ) -> anyhow::Result<EncoderOutput> {
        self.inner.encode_image(image_rgb, width, height).await
    }

    async fn decode_mask(
        &self,
        encoder_output: &EncoderOutput,
        original_width: u32,
        original_height: u32,
        points: &[SamPoint],
        box_prompt: Option<[f32; 4]>,
    ) -> anyhow::Result<SamMaskResult> {
        self.inner
            .decode_mask(
                encoder_output,
                original_width,
                original_height,
                points,
                box_prompt,
            )
            .await
    }

    fn is_available(&self) -> bool {
        self.inner.is_available()
    }
}

/// Benchmark the SAM3 backend across execution providers.
///
/// Mirrors `full_sam::sam::engine::real_model_tests::benchmark_sam_providers`,
/// but drives the real SAM3 backend type (`Sam3CompatOnnxEngine`) so the SAM3
/// inference path is exercised end-to-end. Ignored by default: requires the SAM
/// Tiny ONNX models in `.cache/models` (the compat runtime reuses the SAM2
/// artifacts) and is slow on CPU.
///
/// Run (one GPU provider per build — WebGPU and CUDA prebuilts are mutually
/// exclusive):
///
/// ```text
/// cargo test -p hvat_backend --profile release-dev --features sam-webgpu --lib \
///     full_sam3::sam::engine::sam3_bench::benchmark_sam3_providers \
///     -- --ignored --nocapture
/// ```
#[cfg(test)]
mod sam3_bench {
    use super::*;
    use std::path::Path;

    /// Deterministic synthetic RGB image (no external file deps).
    fn synth_image(width: u32, height: u32, seed: u8) -> Vec<u8> {
        let mut data = vec![0u8; (width * height * 3) as usize];
        for (i, px) in data.chunks_mut(3).enumerate() {
            px[0] = (i as u32).wrapping_add(seed as u32) as u8;
            px[1] = (i as u32 * 2).wrapping_add(seed as u32) as u8;
            px[2] = (i as u32 * 3).wrapping_add(seed as u32) as u8;
        }
        data
    }

    /// Min / median / mean of millisecond samples.
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

    struct ProviderBench {
        encode_ms: Vec<f64>,
        decode_ms: Vec<f64>,
    }

    /// Benchmark a single provider through the SAM3 backend.
    ///
    /// Returns `Err(reason)` when the provider is unavailable — not compiled into
    /// this build, or it failed to initialize on the host. Detected via a
    /// provider mismatch (the engine falls back to CPU internally), so we never
    /// silently benchmark CPU under a GPU label.
    async fn bench_provider(
        model_dir: &Path,
        provider: ExecutionProvider,
        warmup: usize,
        iters: usize,
    ) -> Result<ProviderBench, String> {
        let engine = Sam3CompatOnnxEngine::new(model_dir, SamVariant::Tiny, provider)
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

        let mut encode_ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = std::time::Instant::now();
            embed = engine
                .encode_image(&rgb, width, height)
                .await
                .map_err(|e| format!("encode failed: {e:#}"))?;
            encode_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }

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

    /// Benchmark CPU, WebGPU, and CUDA on the SAM3 backend (SAM2 Tiny artifacts).
    ///
    /// All three are attempted; CPU is always available, WebGPU and CUDA print an
    /// explicit `UNAVAILABLE` line (instead of failing) when not compiled in or
    /// unusable on the host. `scripts/bench_sam_providers.sh` runs both feature
    /// builds to cover all three.
    #[tokio::test]
    #[ignore = "requires real SAM ONNX models in .cache/models and is slow"]
    async fn benchmark_sam3_providers() {
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
            "\n=== SAM3 provider benchmark (SAM2 Tiny artifacts, 800x600, warmup={warmup}, iters={iters}) ==="
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

        assert!(cpu_ok, "CPU provider benchmark did not run");
    }
}
