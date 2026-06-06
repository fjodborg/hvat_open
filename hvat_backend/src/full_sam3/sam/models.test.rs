use super::*;

use crate::full_sam3::config::Sam3RuntimeKind;
use tempfile::tempdir;

#[test]
fn compat_spec_uses_configured_variant() {
    let config = ServerConfig {
        sam_variant: SamVariant::Large,
        ..ServerConfig::default()
    };

    let spec = model_spec_from_config(&config).expect("spec resolution should succeed");
    assert_eq!(
        spec,
        Sam3RuntimeModelSpec::CompatOnnx {
            variant: SamVariant::Large
        }
    );
}

#[test]
fn native_spec_requires_checkpoint() {
    let config = ServerConfig {
        sam3_runtime: Sam3RuntimeKind::Native,
        sam3_config: Some("/tmp/config.json".into()),
        ..ServerConfig::default()
    };

    let err = model_spec_from_config(&config).expect_err("missing checkpoint should fail");
    assert_eq!(err, Sam3ModelSpecError::MissingNativeCheckpoint);
}

#[test]
fn native_spec_requires_config() {
    let config = ServerConfig {
        sam3_runtime: Sam3RuntimeKind::Native,
        sam3_checkpoint: Some("/tmp/sam3.pt".into()),
        ..ServerConfig::default()
    };

    let err = model_spec_from_config(&config).expect_err("missing config should fail");
    assert_eq!(err, Sam3ModelSpecError::MissingNativeConfig);
}

#[test]
fn expected_native_files_include_artifact_paths() {
    let spec = Sam3RuntimeModelSpec::NativeArtifacts {
        checkpoint_path: "/tmp/sam3.pt".into(),
        config_path: "/tmp/config.json".into(),
    };
    let expected = expected_model_files(&spec);

    assert_eq!(expected.len(), 2);
    assert_eq!(expected[0], "/tmp/sam3.pt");
    assert_eq!(expected[1], "/tmp/config.json");
}

#[test]
fn native_spec_requires_existing_checkpoint_file() {
    let temp = tempdir().expect("tempdir");
    let config_path = temp.path().join("config.json");
    std::fs::write(&config_path, "{}").expect("write config");

    let config = ServerConfig {
        sam3_runtime: Sam3RuntimeKind::Native,
        sam3_checkpoint: Some(temp.path().join("missing.pt")),
        sam3_config: Some(config_path),
        ..ServerConfig::default()
    };

    let err = model_spec_from_config(&config).expect_err("missing checkpoint should fail");
    assert!(matches!(
        err,
        Sam3ModelSpecError::NativeCheckpointNotFound { .. }
    ));
}

#[test]
fn native_spec_requires_existing_config_file() {
    let temp = tempdir().expect("tempdir");
    let checkpoint_path = temp.path().join("sam3.pt");
    std::fs::write(&checkpoint_path, "fake-checkpoint").expect("write checkpoint");

    let config = ServerConfig {
        sam3_runtime: Sam3RuntimeKind::Native,
        sam3_checkpoint: Some(checkpoint_path),
        sam3_config: Some(temp.path().join("missing.json")),
        ..ServerConfig::default()
    };

    let err = model_spec_from_config(&config).expect_err("missing config should fail");
    assert!(matches!(
        err,
        Sam3ModelSpecError::NativeConfigNotFound { .. }
    ));
}

#[test]
fn native_spec_requires_checkpoint_to_be_a_file() {
    let temp = tempdir().expect("tempdir");
    let checkpoint_dir = temp.path().join("checkpoint_dir");
    std::fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
    let config_path = temp.path().join("config.json");
    std::fs::write(&config_path, "{}").expect("write config");

    let config = ServerConfig {
        sam3_runtime: Sam3RuntimeKind::Native,
        sam3_checkpoint: Some(checkpoint_dir),
        sam3_config: Some(config_path),
        ..ServerConfig::default()
    };

    let err = model_spec_from_config(&config).expect_err("directory checkpoint should fail");
    assert!(matches!(
        err,
        Sam3ModelSpecError::NativeCheckpointNotFile { .. }
    ));
}

#[test]
fn native_spec_requires_config_to_be_a_file() {
    let temp = tempdir().expect("tempdir");
    let checkpoint_path = temp.path().join("sam3.pt");
    std::fs::write(&checkpoint_path, "fake-checkpoint").expect("write checkpoint");
    let config_dir = temp.path().join("config_dir");
    std::fs::create_dir_all(&config_dir).expect("create config dir");

    let config = ServerConfig {
        sam3_runtime: Sam3RuntimeKind::Native,
        sam3_checkpoint: Some(checkpoint_path),
        sam3_config: Some(config_dir),
        ..ServerConfig::default()
    };

    let err = model_spec_from_config(&config).expect_err("directory config should fail");
    assert!(matches!(
        err,
        Sam3ModelSpecError::NativeConfigNotFile { .. }
    ));
}

#[test]
fn native_spec_requires_non_empty_checkpoint_file() {
    let temp = tempdir().expect("tempdir");
    let checkpoint_path = temp.path().join("sam3.pt");
    let config_path = temp.path().join("config.json");
    std::fs::write(&checkpoint_path, "").expect("write empty checkpoint");
    std::fs::write(&config_path, "{}").expect("write config");

    let config = ServerConfig {
        sam3_runtime: Sam3RuntimeKind::Native,
        sam3_checkpoint: Some(checkpoint_path),
        sam3_config: Some(config_path),
        ..ServerConfig::default()
    };

    let err = model_spec_from_config(&config).expect_err("empty checkpoint should fail");
    assert!(matches!(
        err,
        Sam3ModelSpecError::NativeCheckpointEmpty { .. }
    ));
}

#[test]
fn native_spec_requires_json_config_file() {
    let temp = tempdir().expect("tempdir");
    let checkpoint_path = temp.path().join("sam3.pt");
    let config_path = temp.path().join("config.json");
    std::fs::write(&checkpoint_path, "fake-checkpoint").expect("write checkpoint");
    std::fs::write(&config_path, "not-json").expect("write invalid config");

    let config = ServerConfig {
        sam3_runtime: Sam3RuntimeKind::Native,
        sam3_checkpoint: Some(checkpoint_path),
        sam3_config: Some(config_path),
        ..ServerConfig::default()
    };

    let err = model_spec_from_config(&config).expect_err("invalid config json should fail");
    assert!(matches!(
        err,
        Sam3ModelSpecError::InvalidNativeConfigJson { .. }
    ));
}
