use super::*;

#[test]
fn encodes_header_and_payload() {
    let bytes = FrameBuilder::new()
        .with_header(2, 0x01, 42)
        .push_u16_le(7)
        .push_u32_le(99)
        .finish();

    assert_eq!(bytes[0], 2);
    assert_eq!(bytes[1], 0x01);
    assert_eq!(
        u32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]),
        42
    );
    assert_eq!(u16::from_le_bytes([bytes[6], bytes[7]]), 7);
    assert_eq!(
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        99
    );
}
