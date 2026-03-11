use super::*;

#[test]
fn test_variant_parsing() {
    assert_eq!("tiny".parse::<SamVariant>().ok(), Some(SamVariant::Tiny));
    assert_eq!("SMALL".parse::<SamVariant>().ok(), Some(SamVariant::Small));
    assert_eq!(
        "base-plus".parse::<SamVariant>().ok(),
        Some(SamVariant::BasePlus)
    );
    assert_eq!("large".parse::<SamVariant>().ok(), Some(SamVariant::Large));
    assert!("invalid".parse::<SamVariant>().is_err());
}

#[test]
fn test_urls() {
    let tiny = SamVariant::Tiny;
    assert!(tiny.encoder_url().contains("huggingface.co"));
    assert!(tiny.encoder_url().contains("tiny.encoder.onnx"));
    assert!(tiny.decoder_url().contains("tiny.decoder.onnx"));
}
