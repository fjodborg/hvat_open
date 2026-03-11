use std::marker::PhantomData;

/// Builder state: header has not been written yet.
pub struct NoHeader;

/// Builder state: header has been written and payload can be appended.
pub struct WithHeader;

/// Typestate builder for `[version:u8][type:u8][request_id:u32][payload...]` frames.
pub struct FrameBuilder<State> {
    buf: Vec<u8>,
    _state: PhantomData<State>,
}

impl FrameBuilder<NoHeader> {
    /// Create a new builder without a header.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            _state: PhantomData,
        }
    }

    /// Write the fixed frame header and transition to `WithHeader`.
    #[must_use]
    pub fn with_header(
        mut self,
        version: u8,
        message_type: u8,
        request_id: u32,
    ) -> FrameBuilder<WithHeader> {
        self.buf.reserve(6);
        self.buf.push(version);
        self.buf.push(message_type);
        self.buf.extend_from_slice(&request_id.to_le_bytes());
        FrameBuilder {
            buf: self.buf,
            _state: PhantomData,
        }
    }
}

impl Default for FrameBuilder<NoHeader> {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameBuilder<WithHeader> {
    /// Reserve additional payload capacity.
    #[must_use]
    pub fn reserve_payload(mut self, payload_capacity: usize) -> Self {
        self.buf.reserve(payload_capacity);
        self
    }

    /// Append one `u8` payload value.
    #[must_use]
    pub fn push_u8(mut self, value: u8) -> Self {
        self.buf.push(value);
        self
    }

    /// Append one `u16` payload value (little-endian).
    #[must_use]
    pub fn push_u16_le(mut self, value: u16) -> Self {
        self.buf.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// Append one `u32` payload value (little-endian).
    #[must_use]
    pub fn push_u32_le(mut self, value: u32) -> Self {
        self.buf.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// Append one `u64` payload value (little-endian).
    #[must_use]
    pub fn push_u64_le(mut self, value: u64) -> Self {
        self.buf.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// Append raw payload bytes.
    #[must_use]
    pub fn extend_bytes(mut self, bytes: &[u8]) -> Self {
        self.buf.extend_from_slice(bytes);
        self
    }

    /// Finalize and return encoded frame bytes.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }
}

#[cfg(test)]
#[path = "frame.test.rs"]
mod tests;
