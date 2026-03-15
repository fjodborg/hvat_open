use super::*;

use crate::full_sam3::config::Sam3RuntimeKind;

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
