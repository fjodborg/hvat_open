use super::*;

use crate::full_sam3::config::Sam3RuntimeKind;
use tempfile::tempdir;

#[test]
fn compat_spec_uses_configured_variant() {
    let mut config = ServerConfig::default();
    config.sam_variant = SamVariant::Large;

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
    let mut config = ServerConfig::default();
    config.sam3_runtime = Sam3RuntimeKind::Native;
    config.sam3_config = Some("/tmp/config.json".into());

    let err = model_spec_from_config(&config).expect_err("missing checkpoint should fail");
    assert_eq!(err, Sam3ModelSpecError::MissingNativeCheckpoint);
}

#[test]
fn native_spec_requires_config() {
    let mut config = ServerConfig::default();
    config.sam3_runtime = Sam3RuntimeKind::Native;
    config.sam3_checkpoint = Some("/tmp/sam3.pt".into());

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

    let mut config = ServerConfig::default();
    config.sam3_runtime = Sam3RuntimeKind::Native;
    config.sam3_checkpoint = Some(temp.path().join("missing.pt"));
    config.sam3_config = Some(config_path);

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

    let mut config = ServerConfig::default();
    config.sam3_runtime = Sam3RuntimeKind::Native;
    config.sam3_checkpoint = Some(checkpoint_path);
    config.sam3_config = Some(temp.path().join("missing.json"));

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

    let mut config = ServerConfig::default();
    config.sam3_runtime = Sam3RuntimeKind::Native;
    config.sam3_checkpoint = Some(checkpoint_dir);
    config.sam3_config = Some(config_path);

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

    let mut config = ServerConfig::default();
    config.sam3_runtime = Sam3RuntimeKind::Native;
    config.sam3_checkpoint = Some(checkpoint_path);
    config.sam3_config = Some(config_dir);

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

    let mut config = ServerConfig::default();
    config.sam3_runtime = Sam3RuntimeKind::Native;
    config.sam3_checkpoint = Some(checkpoint_path);
    config.sam3_config = Some(config_path);

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

    let mut config = ServerConfig::default();
    config.sam3_runtime = Sam3RuntimeKind::Native;
    config.sam3_checkpoint = Some(checkpoint_path);
    config.sam3_config = Some(config_path);

    let err = model_spec_from_config(&config).expect_err("invalid config json should fail");
    assert!(matches!(
        err,
        Sam3ModelSpecError::InvalidNativeConfigJson { .. }
    ));
}
