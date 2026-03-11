use super::*;

#[test]
fn test_bilinear_sample_corners() {
    // 2x2 grid with distinct values
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let width = 2;
    let height = 2;

    // Exact corners
    assert!((bilinear_sample(&data, width, height, 0.0, 0.0) - 1.0).abs() < 1e-6);
    assert!((bilinear_sample(&data, width, height, 1.0, 0.0) - 2.0).abs() < 1e-6);
    assert!((bilinear_sample(&data, width, height, 0.0, 1.0) - 3.0).abs() < 1e-6);
    assert!((bilinear_sample(&data, width, height, 1.0, 1.0) - 4.0).abs() < 1e-6);
}

#[test]
fn test_bilinear_sample_center() {
    // 2x2 grid with distinct values
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let width = 2;
    let height = 2;

    // Center should be average of all four
    let center = bilinear_sample(&data, width, height, 0.5, 0.5);
    assert!((center - 2.5).abs() < 1e-6);
}

#[test]
fn test_bilinear_sample_clamping() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let width = 2;
    let height = 2;

    // Out of bounds should clamp
    let oob = bilinear_sample(&data, width, height, -1.0, -1.0);
    assert!((oob - 1.0).abs() < 1e-6); // Should clamp to corner

    let oob2 = bilinear_sample(&data, width, height, 10.0, 10.0);
    assert!((oob2 - 4.0).abs() < 1e-6); // Should clamp to opposite corner
}
