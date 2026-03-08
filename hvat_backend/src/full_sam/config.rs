//! Server configuration.

use std::ops::Deref;
use std::path::PathBuf;

use crate::common::config::{BaseCliArgs, ServerConfig as BaseServerConfig};
use clap::Parser;

use crate::sam::{ExecutionProvider, SamVariant};

/// HVAT Axum Server CLI arguments.
///
/// The shared backend options are flattened from `BaseCliArgs`, while this
/// crate adds axum-specific options (pyramid pregen + SAM integration).
#[derive(Parser, Debug, Clone)]
#[command(name = "hvat-server")]
#[command(about = "HVAT API server for hyperspectral image streaming")]
#[command(version)]
pub struct CliArgs {
    #[command(flatten)]
    pub base: BaseCliArgs,

    /// Maximum memory per user session in MB
    #[arg(long, default_value = "500", env = "HVAT_MAX_USER_MB")]
    pub max_user_mb: u64,

    /// Maximum concurrent pyramid pre-generation jobs at startup
    #[arg(long, default_value = "2", env = "HVAT_PYRAMID_CONCURRENCY")]
    pub pyramid_concurrency: usize,

    // --- SAM Configuration ---
    /// Enable SAM (Segment Anything Model) integration.
    #[arg(long, env = "HVAT_SAM_ENABLED", default_value = "false")]
    pub sam_enabled: bool,

    /// SAM model directory. Defaults to ./.cache/models
    #[arg(long, env = "HVAT_SAM_MODEL_DIR")]
    pub sam_model_dir: Option<PathBuf>,

    /// SAM model variant: tiny, small, base-plus, large
    #[arg(long, env = "HVAT_SAM_VARIANT", default_value = "tiny")]
    pub sam_variant: String,

    /// SAM execution provider: auto, cpu, cuda, rocm, directml, coreml
    #[arg(long, env = "HVAT_SAM_PROVIDER", default_value = "auto")]
    pub sam_provider: String,

    /// Maximum cached image embeddings for SAM
    #[arg(long, env = "HVAT_SAM_CACHE_SIZE", default_value = "50")]
    pub sam_cache_size: usize,
}

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Shared backend configuration.
    pub base: BaseServerConfig,

    /// Maximum memory per user session in bytes
    pub max_user_memory: u64,

    /// Maximum concurrent pyramid pre-generation jobs at startup
    pub pyramid_concurrency: usize,

    // --- SAM Configuration ---
    /// Enable SAM integration
    pub sam_enabled: bool,

    /// SAM model directory
    pub sam_model_dir: PathBuf,

    /// SAM model variant
    pub sam_variant: SamVariant,

    /// SAM execution provider
    pub sam_provider: ExecutionProvider,

    /// Maximum cached embeddings
    pub sam_cache_size: usize,
}

impl Deref for ServerConfig {
    type Target = BaseServerConfig;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            base: BaseServerConfig::default(),
            max_user_memory: 500 * 1024 * 1024, // 500MB
            pyramid_concurrency: 2,
            // SAM defaults
            sam_enabled: false,
            sam_model_dir: crate::sam::models::default_model_dir(),
            sam_variant: SamVariant::default(),
            sam_provider: ExecutionProvider::default(),
            sam_cache_size: 50,
        }
    }
}

impl ServerConfig {
    /// Create configuration from CLI arguments.
    ///
    /// # Panics
    ///
    /// Panics if SAM-specific flags are used without `--sam-enabled`.
    pub fn from_cli(args: CliArgs) -> Self {
        Self::validate_sam_flags(&args);

        let base = BaseServerConfig::from_base_cli(args.base);

        let sam_variant = args.sam_variant.parse().unwrap_or_default();
        let sam_provider = if args.sam_provider == "auto" {
            crate::sam::engine::detect_best_provider()
        } else {
            args.sam_provider.parse().unwrap_or_default()
        };
        let sam_model_dir = args
            .sam_model_dir
            .unwrap_or_else(crate::sam::models::default_model_dir);

        Self {
            base,
            max_user_memory: args.max_user_mb * 1024 * 1024,
            pyramid_concurrency: args.pyramid_concurrency.max(1),
            sam_enabled: args.sam_enabled,
            sam_model_dir,
            sam_variant,
            sam_provider,
            sam_cache_size: args.sam_cache_size,
        }
    }

    /// Validate SAM-related CLI flags.
    ///
    /// Panics if SAM-specific flags are used without `--sam-enabled`.
    fn validate_sam_flags(args: &CliArgs) {
        if args.sam_enabled {
            return;
        }

        let has_custom_variant = args.sam_variant != "tiny";
        let has_custom_provider = args.sam_provider != "auto";
        let has_custom_model_dir = args.sam_model_dir.is_some();
        let has_custom_cache_size = args.sam_cache_size != 50;

        let mut issues = Vec::new();
        if has_custom_variant {
            issues.push(format!("--sam-variant={}", args.sam_variant));
        }
        if has_custom_provider {
            issues.push(format!("--sam-provider={}", args.sam_provider));
        }
        if has_custom_model_dir {
            issues.push(format!(
                "--sam-model-dir={}",
                args.sam_model_dir.as_ref().unwrap().display()
            ));
        }
        if has_custom_cache_size {
            issues.push(format!("--sam-cache-size={}", args.sam_cache_size));
        }

        if !issues.is_empty() {
            panic!(
                "\n\nERROR: SAM flags specified without --sam-enabled!\n\n\
                 The following SAM flags were set:\n  {}\n\n\
                 But --sam-enabled was not specified (defaults to false).\n\n\
                 To enable SAM, add --sam-enabled to your command:\n  \
                 cargo run -p hvat_backend -- --sam-enabled {}\n\n",
                issues.join("\n  "),
                issues.join(" ")
            );
        }
    }

    /// Load configuration from environment variables.
    pub fn from_env() -> Self {
        Self::from_cli(CliArgs::parse())
    }
}
