use super::*;

#[test]
fn test_pack_single_band() {
    let bands: Vec<Vec<f32>> = vec![vec![0.0, 0.5, 1.0, 0.25]];
    let layers = pack_bands_to_rgba_layers(&bands, 2, 2);

    // Should produce at least 2 layers (WebGL2 requirement)
    assert_eq!(layers.len(), 2);

    // First layer should have our band in R channel
    let (idx, data) = &layers[0];
    assert_eq!(*idx, 0);
    assert_eq!(data[0], 0); // pixel 0, R
    assert_eq!(data[3], 255); // pixel 0, A (opaque - no 4th band)
    assert_eq!(data[4], 127); // pixel 1, R (0.5 * 255 ≈ 127)
    assert_eq!(data[7], 255); // pixel 1, A (opaque)
    assert_eq!(data[8], 255); // pixel 2, R
    assert_eq!(data[12], 63); // pixel 3, R (0.25 * 255 ≈ 63)
    assert_eq!(data[15], 255); // pixel 3, A (opaque)
}

#[test]
fn test_pack_three_bands_rgb() {
    // Standard RGB image - 3 bands, no alpha
    let bands: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0], // R: fully red, then black
        vec![0.0, 1.0], // G: black, then fully green
        vec![0.0, 0.0], // B: all black
    ];
    let layers = pack_bands_to_rgba_layers(&bands, 2, 1);

    assert_eq!(layers.len(), 2);

    let (_, data) = &layers[0];
    // Pixel 0: red (RGB=255,0,0)
    assert_eq!(data[0], 255); // R
    assert_eq!(data[1], 0); // G
    assert_eq!(data[2], 0); // B
    assert_eq!(data[3], 255); // A (opaque - no 4th band)
    // Pixel 1: green (RGB=0,255,0)
    assert_eq!(data[4], 0); // R
    assert_eq!(data[5], 255); // G
    assert_eq!(data[6], 0); // B
    assert_eq!(data[7], 255); // A (opaque - no 4th band)

    // Layer 1 should also have opaque alpha
    let (_, layer1_data) = &layers[1];
    assert_eq!(layer1_data[3], 255);
    assert_eq!(layer1_data[7], 255);
}

#[test]
fn test_pack_four_bands() {
    let bands: Vec<Vec<f32>> = vec![
        vec![1.0, 1.0],   // R
        vec![0.5, 0.5],   // G
        vec![0.0, 0.0],   // B
        vec![0.25, 0.25], // A (actual alpha band)
    ];
    let layers = pack_bands_to_rgba_layers(&bands, 2, 1);

    assert_eq!(layers.len(), 2);

    let (_, data) = &layers[0];
    // Pixel 0: RGBA
    assert_eq!(data[0], 255); // R
    assert_eq!(data[1], 127); // G (0.5 * 255 ≈ 127)
    assert_eq!(data[2], 0); // B
    assert_eq!(data[3], 63); // A (from band, not forced to 255)
}
