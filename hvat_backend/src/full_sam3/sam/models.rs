//! SAM3 model-variant mapping for the current compatibility runtime.
//!
//! Until SAM3-native artifacts are wired in, the SAM3 backend uses SAM2 ONNX
//! variant files through a compat path.

/// Compatibility variant type reused from the SAM2 ONNX runtime.
pub type Sam3CompatVariant = crate::full_sam::sam::SamVariant;

/// Return expected ONNX filenames for the current SAM3 compat runtime.
pub fn expected_onnx_files(variant: Sam3CompatVariant) -> [String; 2] {
    [
        variant.encoder_filename().to_string(),
        variant.decoder_filename().to_string(),
    ]
}
