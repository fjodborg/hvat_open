use super::*;

fn dataset_with_mixed_geometries() -> AnnotationDataset {
    AnnotationDataset {
        info: DatasetInfo::default(),
        images: vec![ImageRecord {
            id: 1,
            file_name: "img.png".to_string(),
            width: 100,
            height: 50,
            source_index: None,
        }],
        categories: vec![CategoryRecord {
            id: 10,
            name: "cat".to_string(),
            color: None,
        }],
        annotations: vec![
            AnnotationRecord {
                id: 1,
                image_id: 1,
                category_id: 10,
                geometry: Geometry::BoundingBox {
                    x: 10.0,
                    y: 5.0,
                    w: 20.0,
                    h: 10.0,
                },
                score: None,
                attributes: BTreeMap::new(),
            },
            AnnotationRecord {
                id: 2,
                image_id: 1,
                category_id: 10,
                geometry: Geometry::Polygon {
                    points: vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
                },
                score: None,
                attributes: BTreeMap::new(),
            },
            AnnotationRecord {
                id: 3,
                image_id: 1,
                category_id: 10,
                geometry: Geometry::Point { x: 30.0, y: 15.0 },
                score: None,
                attributes: BTreeMap::new(),
            },
        ],
    }
}

#[test]
fn strict_rejects_non_bbox() {
    let fmt = YoloDetectFormat::new();
    let dataset = dataset_with_mixed_geometries();
    let result = fmt.export(
        &dataset,
        &ExportOptions {
            shape_policy: ShapeConversionPolicy::Strict,
            ..ExportOptions::default()
        },
    );
    assert!(result.is_err());
}

#[test]
fn lossy_round_trip() {
    let fmt = YoloDetectFormat::new();
    let dataset = dataset_with_mixed_geometries();
    let bundle = fmt
        .export(
            &dataset,
            &ExportOptions {
                shape_policy: ShapeConversionPolicy::AllowLossyWarn,
                include_empty_images: true,
                ..ExportOptions::default()
            },
        )
        .unwrap();
    let imported = fmt.import(
        &ImportBundle {
            files: bundle.files,
        },
        &ImportOptions::default(),
    );
    assert!(imported.is_ok());
    let imported = imported.unwrap();
    assert_eq!(imported.dataset.images.len(), 1);
    assert_eq!(imported.dataset.categories.len(), 1);
    assert_eq!(imported.dataset.annotations.len(), 3);
}

#[test]
fn import_supports_flattened_label_paths() {
    let fmt = YoloDetectFormat::new();
    let dataset = dataset_with_mixed_geometries();
    let mut bundle = fmt
        .export(
            &dataset,
            &ExportOptions {
                shape_policy: ShapeConversionPolicy::AllowLossyWarn,
                include_empty_images: true,
                ..ExportOptions::default()
            },
        )
        .unwrap();

    for file in &mut bundle.files {
        if file.path.contains('/') || file.path.contains('\\') {
            file.path = file.path.replace(['/', '\\'], "__");
        }
    }

    let imported = fmt.import(
        &ImportBundle {
            files: bundle.files,
        },
        &ImportOptions::default(),
    );
    assert!(imported.is_ok());
    let imported = imported.unwrap();
    assert_eq!(imported.dataset.annotations.len(), 3);
}
