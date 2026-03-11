use super::*;
use crate::protocol::PROTOCOL_VERSION;

#[test]
fn test_error_encode_decode_roundtrip() {
    let error =
        ProtocolError::retryable(ErrorCode::PyramidNotReady, "Pyramid is building", 5000)
            .with_context(ErrorContext::Pyramid {
                image_id: "test.png".to_string(),
                progress: Some(0.5),
                eta_seconds: Some(10),
            });

    let encoded = error.encode(PROTOCOL_VERSION);
    let decoded = ProtocolError::decode(&encoded[2..]).unwrap();

    assert_eq!(decoded.code as u16, error.code as u16);
    assert_eq!(decoded.severity as u8, error.severity as u8);
    assert_eq!(decoded.retryable, error.retryable);
    assert_eq!(decoded.retry_after_ms, error.retry_after_ms);
    assert_eq!(decoded.message, error.message);
}

#[test]
fn test_error_without_context() {
    let error = ProtocolError::error(ErrorCode::ImageNotFound, "Image not found");

    let encoded = error.encode(PROTOCOL_VERSION);
    let decoded = ProtocolError::decode(&encoded[2..]).unwrap();

    assert!(matches!(decoded.context, ErrorContext::None));
}

#[test]
fn test_fatal_error() {
    let error = ProtocolError::fatal(ErrorCode::ProtocolMismatch, "Incompatible version");

    assert_eq!(error.severity, Severity::Fatal);
    assert!(!error.retryable);
}
