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
