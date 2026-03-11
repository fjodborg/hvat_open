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
