//! RGBA band packing for GPU-ready data.
//!
//! Re-exports shared packing and downsampling logic from `hvat_backend_helper`
//! and provides the `BandPacker` trait for the pyramid builder.

use crate::loaders::BandData;

pub use hvat_backend_helper::packer::{
    BANDS_PER_LAYER, MIN_TEXTURE_LAYERS, downsample_bands, pack_bands_to_rgba_layers,
};

/// Trait for packing band data into GPU-ready formats.
pub trait BandPacker: Send + Sync {
    /// Pack f32 bands into RGBA u8 layers (4 bands per layer).
    fn pack_to_rgba(&self, bands: &BandData) -> Vec<(u32, Vec<u8>)>;

    /// Downsample bands to target resolution.
    fn downsample(&self, bands: &BandData, target_width: u32, target_height: u32) -> BandData;
}

/// Default band packer implementation.
pub struct DefaultBandPacker;

impl DefaultBandPacker {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DefaultBandPacker {
    fn default() -> Self {
        Self::new()
    }
}

impl BandPacker for DefaultBandPacker {
    fn pack_to_rgba(&self, bands: &BandData) -> Vec<(u32, Vec<u8>)> {
        pack_bands_to_rgba_layers(&bands.bands, bands.width, bands.height)
    }

    fn downsample(&self, bands: &BandData, target_width: u32, target_height: u32) -> BandData {
        downsample_bands(bands, target_width, target_height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downsample() {
        let bands = BandData {
            width: 4,
            height: 4,
            bands: vec![vec![
                1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
            ]],
        };

        let downsampled = downsample_bands(&bands, 2, 2);

        assert_eq!(downsampled.width, 2);
        assert_eq!(downsampled.height, 2);
        assert_eq!(downsampled.bands.len(), 1);

        let band = &downsampled.bands[0];
        assert!(band[0] > 0.5);
        assert!(band[1] < 0.5);
        assert!(band[2] < 0.5);
        assert!(band[3] > 0.5);
    }
}
