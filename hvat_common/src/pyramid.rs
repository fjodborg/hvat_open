//! Pyramid-related types with compile-time safety guarantees.

use std::fmt;

/// Maximum pyramid level (0 = full resolution, higher = smaller).
///
/// Level 7 would be 1/128th resolution, which is sufficient for
/// any practical image size.
pub const MAX_PYRAMID_LEVEL: u8 = 7;

/// A validated pyramid level.
///
/// Pyramid levels represent different resolutions of an image:
/// - Level 0: Full resolution
/// - Level 1: 1/2 resolution (half width and height)
/// - Level 2: 1/4 resolution
/// - Level N: 1/(2^N) resolution
///
/// This type guarantees the level is within valid bounds (0 to [`MAX_PYRAMID_LEVEL`]).
///
/// # Example
///
/// ```
/// use hvat_common::PyramidLevel;
///
/// // Create from validated input
/// let level = PyramidLevel::new(2).expect("level 2 is valid");
/// assert_eq!(level.as_u8(), 2);
///
/// // Invalid levels return None
/// assert!(PyramidLevel::new(100).is_none());
///
/// // Use predefined constants
/// let full_res = PyramidLevel::FULL_RES;
/// let thumbnail = PyramidLevel::THUMBNAIL;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PyramidLevel(u8);

impl PyramidLevel {
    /// Full resolution (level 0).
    pub const FULL_RES: Self = Self(0);

    /// Typical thumbnail level (level 3 = 1/8 resolution).
    pub const THUMBNAIL: Self = Self(3);

    /// Create a new pyramid level if the value is valid.
    ///
    /// Returns `None` if `level > MAX_PYRAMID_LEVEL`.
    #[inline]
    pub fn new(level: u8) -> Option<Self> {
        if level <= MAX_PYRAMID_LEVEL {
            Some(Self(level))
        } else {
            None
        }
    }

    /// Create a pyramid level, clamping to the maximum if necessary.
    ///
    /// This is useful when you want to ensure a valid level without
    /// error handling, accepting that out-of-range values become the max.
    #[inline]
    pub fn new_clamped(level: u8) -> Self {
        Self(level.min(MAX_PYRAMID_LEVEL))
    }

    /// Create a pyramid level from a u32, clamping to valid range.
    ///
    /// Useful for parsing client requests where level comes as u32.
    #[inline]
    pub fn from_u32_clamped(level: u32) -> Self {
        Self::new_clamped(level.min(u8::MAX as u32) as u8)
    }

    /// Create a pyramid level, clamping to a specific maximum.
    ///
    /// Useful when the actual image has fewer levels than [`MAX_PYRAMID_LEVEL`].
    #[inline]
    pub fn new_clamped_to(level: u8, max_level: u8) -> Self {
        Self(level.min(max_level).min(MAX_PYRAMID_LEVEL))
    }

    /// Get the raw level value.
    #[inline]
    pub fn as_u8(self) -> u8 {
        self.0
    }

    /// Get the level as usize for indexing.
    #[inline]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Get the level as u32.
    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0 as u32
    }

    /// Returns true if this is full resolution (level 0).
    #[inline]
    pub fn is_full_res(self) -> bool {
        self.0 == 0
    }

    /// Calculate the scale factor for this level.
    ///
    /// Level 0 = 1, Level 1 = 2, Level 2 = 4, etc.
    #[inline]
    pub fn scale_factor(self) -> u32 {
        1u32 << self.0
    }

    /// Calculate the dimensions at this pyramid level.
    ///
    /// Returns `(width, height)` for the given full-resolution dimensions.
    #[inline]
    pub fn dimensions_for(self, full_width: u32, full_height: u32) -> (u32, u32) {
        let scale = self.scale_factor();
        ((full_width / scale).max(1), (full_height / scale).max(1))
    }

    /// Iterate from this level down to full resolution (level 0).
    ///
    /// Useful for progressive loading where you start from thumbnail
    /// and refine to full resolution.
    #[inline]
    pub fn iter_to_full_res(self) -> impl Iterator<Item = PyramidLevel> {
        (0..=self.0).rev().map(|l| PyramidLevel(l))
    }

    /// Iterate from full resolution (level 0) up to this level.
    #[inline]
    pub fn iter_from_full_res(self) -> impl Iterator<Item = PyramidLevel> {
        (0..=self.0).map(|l| PyramidLevel(l))
    }
}

impl fmt::Display for PyramidLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "level {}", self.0)
    }
}

impl Default for PyramidLevel {
    /// Default to full resolution.
    fn default() -> Self {
        Self::FULL_RES
    }
}

impl From<PyramidLevel> for u8 {
    fn from(level: PyramidLevel) -> u8 {
        level.0
    }
}

impl From<PyramidLevel> for u32 {
    fn from(level: PyramidLevel) -> u32 {
        level.0 as u32
    }
}

impl From<PyramidLevel> for usize {
    fn from(level: PyramidLevel) -> usize {
        level.0 as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid() {
        assert!(PyramidLevel::new(0).is_some());
        assert!(PyramidLevel::new(MAX_PYRAMID_LEVEL).is_some());
    }

    #[test]
    fn test_new_invalid() {
        assert!(PyramidLevel::new(MAX_PYRAMID_LEVEL + 1).is_none());
        assert!(PyramidLevel::new(255).is_none());
    }

    #[test]
    fn test_new_clamped() {
        assert_eq!(PyramidLevel::new_clamped(0).as_u8(), 0);
        assert_eq!(PyramidLevel::new_clamped(100).as_u8(), MAX_PYRAMID_LEVEL);
    }

    #[test]
    fn test_from_u32_clamped() {
        assert_eq!(PyramidLevel::from_u32_clamped(0).as_u8(), 0);
        assert_eq!(PyramidLevel::from_u32_clamped(3).as_u8(), 3);
        assert_eq!(
            PyramidLevel::from_u32_clamped(1000).as_u8(),
            MAX_PYRAMID_LEVEL
        );
    }

    #[test]
    fn test_scale_factor() {
        assert_eq!(PyramidLevel::new(0).unwrap().scale_factor(), 1);
        assert_eq!(PyramidLevel::new(1).unwrap().scale_factor(), 2);
        assert_eq!(PyramidLevel::new(2).unwrap().scale_factor(), 4);
        assert_eq!(PyramidLevel::new(3).unwrap().scale_factor(), 8);
    }

    #[test]
    fn test_dimensions_for() {
        let level = PyramidLevel::new(2).unwrap();
        assert_eq!(level.dimensions_for(1024, 768), (256, 192));

        // Minimum size is 1x1
        let level = PyramidLevel::new(7).unwrap();
        assert_eq!(level.dimensions_for(64, 64), (1, 1));
    }

    #[test]
    fn test_iter_to_full_res() {
        let level = PyramidLevel::new(3).unwrap();
        let levels: Vec<u8> = level.iter_to_full_res().map(|l| l.as_u8()).collect();
        assert_eq!(levels, vec![3, 2, 1, 0]);
    }

    #[test]
    fn test_constants() {
        assert_eq!(PyramidLevel::FULL_RES.as_u8(), 0);
        assert_eq!(PyramidLevel::THUMBNAIL.as_u8(), 3);
    }

    #[test]
    fn test_is_full_res() {
        assert!(PyramidLevel::FULL_RES.is_full_res());
        assert!(!PyramidLevel::THUMBNAIL.is_full_res());
    }
}
