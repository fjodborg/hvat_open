use super::*;
use crate::full_sam3::config::Sam3RuntimeKind;

#[test]
fn sam3_compat_wiring_uses_distinct_model_identity() {
    let sam2 = SamBackendWiring::sam2_default();
    let sam3 = sam3_compat_wiring();

    assert_ne!(sam2.backend_name, sam3.backend_name);
    assert_ne!(sam2.model_id, sam3.model_id);
    assert_eq!(sam3.model_id, SAM3_COMPAT_MODEL_ID);
}

#[test]
fn native_runtime_init_surfaces_artifacts_in_error_context() {
    let mut config = ServerConfig {
        sam_enabled: true,
        ..ServerConfig::default()
    };
    config.sam3_runtime = Sam3RuntimeKind::Native;
    config.sam3_checkpoint = Some(std::path::PathBuf::from("/tmp/sam3.pt"));
    config.sam3_config = Some(std::path::PathBuf::from("/tmp/config.json"));

    let err = match build_app_state(config) {
        Ok(_) => panic!("native runtime should not initialize yet"),
        Err(err) => err,
    };
    match err {
        StateInitError::SamInit {
            backend_name,
            expected_files,
            source,
            ..
        } => {
            assert_eq!(backend_name, "sam3-native");
            assert!(expected_files.contains("/tmp/sam3.pt"));
            assert!(expected_files.contains("/tmp/config.json"));
            assert!(source.to_string().contains("not implemented yet"));
        }
    }
}
