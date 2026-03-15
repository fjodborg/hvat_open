//! SAM3 backend configuration.

use std::fmt;
use std::ops::Deref;
use std::path::PathBuf;
use std::str::FromStr;

use crate::common::config::{BaseCliArgs, ServerConfig as BaseServerConfig};
use crate::full_sam::sam::{ExecutionProvider, SamVariant};
use clap::Parser;

/// Runtime family used by the SAM3 backend executable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Sam3RuntimeKind {
    /// Current compatibility path: SAM2 ONNX artifacts through SAM3 wiring.
    #[default]
    CompatOnnx,
    /// Native SAM3 runtime path using SAM3 checkpoint + config artifacts.
    Native,
}

/// Error returned when parsing an invalid SAM3 runtime kind string.
#[derive(Debug, Clone)]
pub struct ParseSam3RuntimeKindError(String);

impl fmt::Display for ParseSam3RuntimeKindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown SAM3 runtime '{}', expected one of: compat, compat-onnx, native",
            self.0
        )
    }
}

impl std::error::Error for ParseSam3RuntimeKindError {}

impl FromStr for Sam3RuntimeKind {
    type Err = ParseSam3RuntimeKindError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "compat" | "compat-onnx" | "compat_onnx" => Ok(Self::CompatOnnx),
            "native" => Ok(Self::Native),
            _ => Err(ParseSam3RuntimeKindError(s.to_string())),
        }
    }
}

impl Sam3RuntimeKind {
    /// Human-readable runtime name.
    pub fn name(self) -> &'static str {
        match self {
            Self::CompatOnnx => "compat-onnx",
            Self::Native => "native",
        }
    }
}

/// HVAT Axum SAM3 backend CLI arguments.
#[derive(Parser, Debug, Clone)]
#[command(name = "hvat-backend-sam3")]
#[command(about = "HVAT API server for hyperspectral image streaming (SAM3 backend)")]
#[command(version)]
pub struct CliArgs {
    #[command(flatten)]
    pub base: BaseCliArgs,

    /// Maximum memory per user session in MB.
    #[arg(long, default_value = "500", env = "HVAT_MAX_USER_MB")]
    pub max_user_mb: u64,

    /// Maximum concurrent pyramid pre-generation jobs at startup.
    #[arg(long, default_value = "2", env = "HVAT_PYRAMID_CONCURRENCY")]
    pub pyramid_concurrency: usize,

    // --- SAM Runtime Configuration ---
    /// Enable SAM integration in this backend.
    #[arg(long, env = "HVAT_SAM_ENABLED", default_value = "false")]
    pub sam_enabled: bool,

    /// SAM model directory (used by compat-onnx runtime).
    #[arg(long, env = "HVAT_SAM_MODEL_DIR")]
    pub sam_model_dir: Option<PathBuf>,

    /// SAM model variant (used by compat-onnx runtime).
    #[arg(long, env = "HVAT_SAM_VARIANT", default_value = "tiny")]
    pub sam_variant: String,

    /// SAM execution provider.
    #[arg(long, env = "HVAT_SAM_PROVIDER", default_value = "auto")]
    pub sam_provider: String,

    /// Maximum cached image embeddings for SAM.
    #[arg(long, env = "HVAT_SAM_CACHE_SIZE", default_value = "50")]
    pub sam_cache_size: usize,

    /// SAM3 runtime kind.
    #[arg(long, env = "HVAT_SAM3_RUNTIME", default_value = "compat")]
    pub sam3_runtime: String,

    /// Local SAM3 checkpoint path (required for native runtime).
    #[arg(long, env = "HVAT_SAM3_CHECKPOINT")]
    pub sam3_checkpoint: Option<PathBuf>,

    /// Local SAM3 config path (required for native runtime).
    #[arg(long, env = "HVAT_SAM3_CONFIG")]
    pub sam3_config: Option<PathBuf>,
}

/// Configuration parsing and validation errors.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("SAM-specific flags require --sam-enabled: {0}")]
    SamFlagsWithoutEnable(String),

    #[error("Invalid SAM variant '{value}': {reason}")]
    InvalidSamVariant { value: String, reason: String },

    #[error("Invalid SAM execution provider '{value}': {reason}")]
    InvalidSamProvider { value: String, reason: String },

    #[error("Invalid SAM3 runtime '{value}': {reason}")]
    InvalidSam3Runtime { value: String, reason: String },

    #[error("Native SAM3 runtime requires both --sam3-checkpoint and --sam3-config")]
    NativeSam3RequiresArtifacts,

    #[error("SAM3 checkpoint/config flags require --sam3-runtime native")]
    Sam3ArtifactsRequireNativeRuntime,
}

/// SAM3 backend server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Shared backend configuration.
    pub base: BaseServerConfig,

    /// Maximum memory per user session in bytes.
    pub max_user_memory: u64,

    /// Maximum concurrent pyramid pre-generation jobs at startup.
    pub pyramid_concurrency: usize,

    /// Enable SAM integration.
    pub sam_enabled: bool,

    /// SAM model directory.
    pub sam_model_dir: PathBuf,

    /// SAM model variant for compat runtime.
    pub sam_variant: SamVariant,

    /// SAM execution provider.
    pub sam_provider: ExecutionProvider,

    /// Maximum cached embeddings.
    pub sam_cache_size: usize,

    /// Selected SAM3 runtime.
    pub sam3_runtime: Sam3RuntimeKind,

    /// SAM3 checkpoint artifact path (native runtime).
    pub sam3_checkpoint: Option<PathBuf>,

    /// SAM3 config artifact path (native runtime).
    pub sam3_config: Option<PathBuf>,
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
            max_user_memory: 500 * 1024 * 1024,
            pyramid_concurrency: 2,
            sam_enabled: false,
            sam_model_dir: crate::full_sam::sam::models::default_model_dir(),
            sam_variant: SamVariant::default(),
            sam_provider: ExecutionProvider::default(),
            sam_cache_size: 50,
            sam3_runtime: Sam3RuntimeKind::default(),
            sam3_checkpoint: None,
            sam3_config: None,
        }
    }
}

impl ServerConfig {
    /// Create configuration from CLI arguments.
    pub fn from_cli(args: CliArgs) -> std::result::Result<Self, ConfigError> {
        Self::validate_sam_flags(&args)?;

        let base = BaseServerConfig::from_base_cli(args.base);

        let sam_variant = args.sam_variant.parse().map_err(
            |e: crate::full_sam::sam::models::ParseSamVariantError| {
                ConfigError::InvalidSamVariant {
                    value: args.sam_variant.clone(),
                    reason: e.to_string(),
                }
            },
        )?;
        let sam_provider = if args.sam_provider == "auto" {
            crate::full_sam::sam::engine::detect_best_provider()
        } else {
            args.sam_provider.parse().map_err(
                |e: crate::full_sam::sam::backend::ParseExecutionProviderError| {
                    ConfigError::InvalidSamProvider {
                        value: args.sam_provider.clone(),
                        reason: e.to_string(),
                    }
                },
            )?
        };
        let sam3_runtime = args
            .sam3_runtime
            .parse()
            .map_err(
                |e: ParseSam3RuntimeKindError| ConfigError::InvalidSam3Runtime {
                    value: args.sam3_runtime.clone(),
                    reason: e.to_string(),
                },
            )?;

        let sam_model_dir = args
            .sam_model_dir
            .unwrap_or_else(crate::full_sam::sam::models::default_model_dir);

        Ok(Self {
            base,
            max_user_memory: args.max_user_mb * 1024 * 1024,
            pyramid_concurrency: args.pyramid_concurrency.max(1),
            sam_enabled: args.sam_enabled,
            sam_model_dir,
            sam_variant,
            sam_provider,
            sam_cache_size: args.sam_cache_size,
            sam3_runtime,
            sam3_checkpoint: args.sam3_checkpoint,
            sam3_config: args.sam3_config,
        })
    }

    /// Validate SAM-related CLI flags.
    fn validate_sam_flags(args: &CliArgs) -> std::result::Result<(), ConfigError> {
        let has_custom_variant = args.sam_variant != "tiny";
        let has_custom_provider = args.sam_provider != "auto";
        let has_custom_cache_size = args.sam_cache_size != 50;
        let has_custom_runtime = args.sam3_runtime != "compat";
        let has_checkpoint = args.sam3_checkpoint.is_some();
        let has_config = args.sam3_config.is_some();

        if !args.sam_enabled {
            let mut issues = Vec::new();
            if has_custom_variant {
                issues.push(format!("--sam-variant={}", args.sam_variant));
            }
            if has_custom_provider {
                issues.push(format!("--sam-provider={}", args.sam_provider));
            }
            if let Some(model_dir) = args.sam_model_dir.as_ref() {
                issues.push(format!("--sam-model-dir={}", model_dir.display()));
            }
            if has_custom_cache_size {
                issues.push(format!("--sam-cache-size={}", args.sam_cache_size));
            }
            if has_custom_runtime {
                issues.push(format!("--sam3-runtime={}", args.sam3_runtime));
            }
            if let Some(checkpoint) = args.sam3_checkpoint.as_ref() {
                issues.push(format!("--sam3-checkpoint={}", checkpoint.display()));
            }
            if let Some(config) = args.sam3_config.as_ref() {
                issues.push(format!("--sam3-config={}", config.display()));
            }

            if !issues.is_empty() {
                return Err(ConfigError::SamFlagsWithoutEnable(format!(
                    "set flags [{}], but SAM is disabled",
                    issues.join(", ")
                )));
            }
            return Ok(());
        }

        let runtime = args.sam3_runtime.parse::<Sam3RuntimeKind>().map_err(|e| {
            ConfigError::InvalidSam3Runtime {
                value: args.sam3_runtime.clone(),
                reason: e.to_string(),
            }
        })?;

        match runtime {
            Sam3RuntimeKind::CompatOnnx => {
                if has_checkpoint || has_config {
                    return Err(ConfigError::Sam3ArtifactsRequireNativeRuntime);
                }
            }
            Sam3RuntimeKind::Native => {
                if !has_checkpoint || !has_config {
                    return Err(ConfigError::NativeSam3RequiresArtifacts);
                }
            }
        }

        Ok(())
    }

    /// Load configuration from environment variables.
    pub fn from_env() -> anyhow::Result<Self> {
        Self::from_cli(CliArgs::parse()).map_err(Into::into)
    }

    /// Convert to the shared full backend config shape used by `AppState`.
    pub fn to_full_sam_config(&self) -> crate::full_sam::config::ServerConfig {
        crate::full_sam::config::ServerConfig {
            base: self.base.clone(),
            max_user_memory: self.max_user_memory,
            pyramid_concurrency: self.pyramid_concurrency,
            sam_enabled: self.sam_enabled,
            sam_model_dir: self.sam_model_dir.clone(),
            sam_variant: self.sam_variant,
            sam_provider: self.sam_provider,
            sam_cache_size: self.sam_cache_size,
        }
    }
}

#[cfg(test)]
#[path = "config.test.rs"]
mod tests;
