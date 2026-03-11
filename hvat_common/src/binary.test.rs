use super::*;

#[test]
fn test_read_u8() {
    let data = [0x42, 0xFF];
    let mut reader = BinaryReader::new(&data);

    assert_eq!(reader.read_u8(), Some(0x42));
    assert_eq!(reader.read_u8(), Some(0xFF));
    assert_eq!(reader.read_u8(), None);
}

#[test]
fn test_read_u16() {
    let data = [0x01, 0x02, 0x03];
    let mut reader = BinaryReader::new(&data);

    assert_eq!(reader.read_u16(), Some(0x0201)); // Little-endian
    assert_eq!(reader.read_u16(), None); // Only 1 byte left
}

#[test]
fn test_read_u32() {
    let data = 42u32.to_le_bytes();
    let mut reader = BinaryReader::new(&data);

    assert_eq!(reader.read_u32(), Some(42));
    assert_eq!(reader.read_u32(), None);
}

#[test]
fn test_read_bytes() {
    let data = [1, 2, 3, 4, 5];
    let mut reader = BinaryReader::new(&data);

    assert_eq!(reader.read_bytes(3), Some(&[1, 2, 3][..]));
    assert_eq!(reader.read_bytes(3), None); // Only 2 bytes left
    assert_eq!(reader.read_bytes(2), Some(&[4, 5][..]));
}

#[test]
fn test_read_str() {
    let data = b"hello\xFF";
    let mut reader = BinaryReader::new(data);

    assert_eq!(reader.read_str(5), Some("hello"));
    assert_eq!(reader.read_str(1), None); // 0xFF is not valid UTF-8
}

#[test]
fn test_remaining_and_position() {
    let data = [1, 2, 3, 4];
    let mut reader = BinaryReader::new(&data);

    assert_eq!(reader.remaining(), 4);
    assert_eq!(reader.position(), 0);

    reader.read_u16();
    assert_eq!(reader.remaining(), 2);
    assert_eq!(reader.position(), 2);
}

#[test]
fn test_skip() {
    let data = [1, 2, 3, 4];
    let mut reader = BinaryReader::new(&data);

    assert!(reader.skip(2).is_some());
    assert_eq!(reader.read_u8(), Some(3));
    assert!(reader.skip(5).is_none()); // Only 1 byte left
}

#[test]
fn test_empty_reader() {
    let data: [u8; 0] = [];
    let mut reader = BinaryReader::new(&data);

    assert!(reader.is_empty());
    assert_eq!(reader.read_u8(), None);
}
