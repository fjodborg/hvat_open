use super::*;

#[test]
fn test_calculate_num_levels() {
    // Small image - just 1 level
    assert_eq!(calculate_num_levels(256, 256), 1);
    assert_eq!(calculate_num_levels(128, 128), 1);

    // 512x512 -> 256x256 = 2 levels
    assert_eq!(calculate_num_levels(512, 512), 2);

    // 1024x1024 -> 512 -> 256 = 3 levels
    assert_eq!(calculate_num_levels(1024, 1024), 3);

    // 4096x4096 -> 2048 -> 1024 -> 512 -> 256 = 5 levels
    assert_eq!(calculate_num_levels(4096, 4096), 5);
}

#[test]
fn test_calculate_level_dimensions() {
    // For a 1024x1024 image:
    // Level 0: 1024x1024 (full)
    // Level 1: 512x512
    // Level 2: 256x256 (thumbnail)

    let (w, h) = calculate_level_dimensions(1024, 1024, 0);
    assert_eq!((w, h), (1024, 1024));

    let (w, h) = calculate_level_dimensions(1024, 1024, 1);
    assert_eq!((w, h), (512, 512));

    let (w, h) = calculate_level_dimensions(1024, 1024, 2);
    assert_eq!((w, h), (256, 256));
}
