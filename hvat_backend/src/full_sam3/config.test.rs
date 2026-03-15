use super::*;

use clap::Parser;

#[test]
fn defaults_to_compat_runtime() {
    let args = CliArgs::try_parse_from(["hvat_backend_sam3"]).expect("cli parse should succeed");
    let config = ServerConfig::from_cli(args).expect("config parse should succeed");

    assert_eq!(config.sam3_runtime, Sam3RuntimeKind::CompatOnnx);
    assert!(config.sam3_checkpoint.is_none());
    assert!(config.sam3_config.is_none());
}

#[test]
fn native_runtime_requires_checkpoint_and_config() {
    let args = CliArgs::try_parse_from([
        "hvat_backend_sam3",
        "--sam-enabled",
        "--sam3-runtime",
        "native",
    ])
    .expect("cli parse should succeed");

    let err = ServerConfig::from_cli(args).expect_err("config parse should fail");
    assert!(matches!(err, ConfigError::NativeSam3RequiresArtifacts));
}

#[test]
fn sam3_artifacts_require_native_runtime() {
    let args = CliArgs::try_parse_from([
        "hvat_backend_sam3",
        "--sam-enabled",
        "--sam3-checkpoint",
        "/tmp/sam3.pt",
        "--sam3-config",
        "/tmp/config.json",
    ])
    .expect("cli parse should succeed");

    let err = ServerConfig::from_cli(args).expect_err("config parse should fail");
    assert!(matches!(
        err,
        ConfigError::Sam3ArtifactsRequireNativeRuntime
    ));
}

#[test]
fn native_runtime_accepts_explicit_artifacts() {
    let args = CliArgs::try_parse_from([
        "hvat_backend_sam3",
        "--sam-enabled",
        "--sam3-runtime",
        "native",
        "--sam3-checkpoint",
        "/tmp/sam3.pt",
        "--sam3-config",
        "/tmp/config.json",
    ])
    .expect("cli parse should succeed");

    let config = ServerConfig::from_cli(args).expect("config parse should succeed");
    assert_eq!(config.sam3_runtime, Sam3RuntimeKind::Native);
    assert_eq!(
        config.sam3_checkpoint,
        Some(std::path::PathBuf::from("/tmp/sam3.pt"))
    );
    assert_eq!(
        config.sam3_config,
        Some(std::path::PathBuf::from("/tmp/config.json"))
    );
}
