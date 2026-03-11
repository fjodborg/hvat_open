use super::*;

#[test]
fn sam3_compat_wiring_uses_distinct_model_identity() {
    let sam2 = SamBackendWiring::sam2_default();
    let sam3 = sam3_compat_wiring();

    assert_ne!(sam2.backend_name, sam3.backend_name);
    assert_ne!(sam2.model_id, sam3.model_id);
    assert_eq!(sam3.model_id, SAM3_COMPAT_MODEL_ID);
}
