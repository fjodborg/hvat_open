use super::*;

#[test]
fn datumaro_round_trip_mixed_shapes() {
    let fmt = DatumaroFormat::new();
    let dataset = AnnotationDataset {
        info: DatasetInfo::default(),
        images: vec![ImageRecord {
            id: 1,
            file_name: "img.png".to_string(),
            width: 100,
            height: 80,
            source_index: None,
        }],
        categories: vec![CategoryRecord {
            id: 1,
            name: "cat".to_string(),
            color: None,
        }],
        annotations: vec![
            AnnotationRecord {
                id: 1,
                image_id: 1,
                category_id: 1,
                geometry: Geometry::BoundingBox {
                    x: 1.0,
                    y: 2.0,
                    w: 3.0,
                    h: 4.0,
                },
                score: None,
                attributes: BTreeMap::new(),
            },
            AnnotationRecord {
                id: 2,
                image_id: 1,
                category_id: 1,
                geometry: Geometry::Polygon {
                    points: vec![(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)],
                },
                score: None,
                attributes: BTreeMap::new(),
            },
            AnnotationRecord {
                id: 3,
                image_id: 1,
                category_id: 1,
                geometry: Geometry::Point { x: 7.0, y: 8.0 },
                score: None,
                attributes: BTreeMap::new(),
            },
        ],
    };

    let bundle = fmt.export(&dataset, &ExportOptions::default()).unwrap();
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
