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

#[test]
fn test_calculate_level_size() {
    assert_eq!(calculate_level_size(1024, 1024, 0), (1024, 1024));
    assert_eq!(calculate_level_size(1024, 1024, 1), (512, 512));
    assert_eq!(calculate_level_size(1024, 1024, 2), (256, 256));
}
