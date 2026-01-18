//! Safe binary reading utilities for protocol parsing.
//!
//! Provides a cursor-based reader that prevents out-of-bounds access
//! at compile time by returning `Option` for all read operations.

/// A cursor-based binary reader for safe protocol parsing.
///
/// All read operations return `Option`, making it impossible to
/// accidentally read past the buffer bounds. The cursor advances
/// automatically after each successful read.
///
/// # Example
///
/// ```
/// use hvat_common::BinaryReader;
///
/// let data = [0x01, 0x00, 0x00, 0x00, 0xFF, 0x00];
/// let mut reader = BinaryReader::new(&data);
///
/// assert_eq!(reader.read_u32(), Some(1));
/// assert_eq!(reader.read_u16(), Some(255));
/// assert_eq!(reader.read_u8(), None); // No more data
/// ```
#[derive(Debug, Clone)]
pub struct BinaryReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BinaryReader<'a> {
    /// Create a new reader over the given byte slice.
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Returns the number of bytes remaining to be read.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Returns true if there are no more bytes to read.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pos >= self.data.len()
    }

    /// Returns the current position in the buffer.
    #[inline]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Read a single byte, advancing the cursor.
    #[inline]
    pub fn read_u8(&mut self) -> Option<u8> {
        let byte = *self.data.get(self.pos)?;
        self.pos += 1;
        Some(byte)
    }

    /// Read a little-endian u16, advancing the cursor.
    #[inline]
    pub fn read_u16(&mut self) -> Option<u16> {
        let bytes = self.data.get(self.pos..self.pos + 2)?;
        self.pos += 2;
        Some(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    /// Read a little-endian u32, advancing the cursor.
    #[inline]
    pub fn read_u32(&mut self) -> Option<u32> {
        let bytes = self.data.get(self.pos..self.pos + 4)?;
        self.pos += 4;
        Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Read a little-endian u64, advancing the cursor.
    #[inline]
    pub fn read_u64(&mut self) -> Option<u64> {
        let bytes = self.data.get(self.pos..self.pos + 8)?;
        self.pos += 8;
        Some(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Read a little-endian f32, advancing the cursor.
    #[inline]
    pub fn read_f32(&mut self) -> Option<f32> {
        self.read_u32().map(f32::from_bits)
    }

    /// Read exactly `len` bytes as a slice, advancing the cursor.
    ///
    /// Returns `None` if fewer than `len` bytes remain.
    #[inline]
    pub fn read_bytes(&mut self, len: usize) -> Option<&'a [u8]> {
        let bytes = self.data.get(self.pos..self.pos + len)?;
        self.pos += len;
        Some(bytes)
    }

    /// Read remaining bytes as a UTF-8 string of the given length.
    ///
    /// Returns `None` if fewer than `len` bytes remain or if the bytes
    /// are not valid UTF-8.
    #[inline]
    pub fn read_str(&mut self, len: usize) -> Option<&'a str> {
        let bytes = self.read_bytes(len)?;
        std::str::from_utf8(bytes).ok()
    }

    /// Read remaining bytes as a UTF-8 string, using lossy conversion.
    ///
    /// Invalid UTF-8 sequences are replaced with the Unicode replacement character.
    #[inline]
    pub fn read_str_lossy(&mut self, len: usize) -> Option<String> {
        let bytes = self.read_bytes(len)?;
        Some(String::from_utf8_lossy(bytes).into_owned())
    }

    /// Read all remaining bytes without advancing (peek).
    #[inline]
    pub fn peek_remaining(&self) -> &'a [u8] {
        self.data.get(self.pos..).unwrap_or(&[])
    }

    /// Skip `n` bytes, advancing the cursor.
    ///
    /// Returns `None` if fewer than `n` bytes remain.
    #[inline]
    pub fn skip(&mut self, n: usize) -> Option<()> {
        if self.pos + n <= self.data.len() {
            self.pos += n;
            Some(())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
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
}
