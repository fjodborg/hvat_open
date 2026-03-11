use std::collections::BTreeMap;

use super::*;
use crate::annotation_io::policy::{ExportOptions, ImportOptions};

#[test]
fn coco_round_trip_single_bbox() {
    let format = CocoFormat::new();
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
            id: 7,
            name: "vehicle".to_string(),
            color: Some([255, 0, 0]),
        }],
        annotations: vec![AnnotationRecord {
            id: 42,
            image_id: 1,
            category_id: 7,
            geometry: Geometry::BoundingBox {
                x: 10.0,
                y: 11.0,
                w: 20.0,
                h: 21.0,
            },
            score: None,
            attributes: BTreeMap::new(),
        }],
    };

    let bundle = format.export(&dataset, &ExportOptions::default()).unwrap();
    let imported = format
        .import(
            &ImportBundle {
                files: bundle.files,
            },
            &ImportOptions::default(),
        )
        .unwrap();

    assert_eq!(imported.dataset.images.len(), 1);
    assert_eq!(imported.dataset.categories.len(), 1);
    assert_eq!(imported.dataset.annotations.len(), 1);
}

#[test]
fn invalid_polygon_is_skipped_with_warning() {
    let format = CocoFormat::new();
    let bundle = ImportBundle {
        files: vec![BundleFile {
            path: "annotations.json".to_string(),
            mime_type: Some("application/json".to_string()),
            bytes: br#"{
                "images":[{"id":1,"file_name":"a.png","width":1,"height":1}],
                "categories":[{"id":1,"name":"cat","supercategory":"none"}],
                "annotations":[{"id":9,"image_id":1,"category_id":1,"segmentation":[[0,0,1]],"area":0,"iscrowd":0}]
            }"#
                .to_vec(),
        }],
    };

    let imported = format.import(&bundle, &ImportOptions::default()).unwrap();
    assert_eq!(imported.dataset.annotations.len(), 0);
    assert_eq!(imported.report.skipped_annotations, 1);
    assert_eq!(imported.report.warnings.len(), 1);
    assert_eq!(
        imported.report.warnings[0].code,
        IoWarningCode::InvalidRecord
    );
}

#[test]
fn export_writes_image_file_names() {
    let format = CocoFormat::new();
    let dataset = AnnotationDataset {
        info: DatasetInfo::default(),
        images: vec![
            ImageRecord {
                id: 1,
                file_name: "images/a.png".to_string(),
                width: 100,
                height: 80,
                source_index: None,
            },
            ImageRecord {
                id: 2,
                file_name: "nested/folder/b.jpg".to_string(),
                width: 200,
                height: 160,
                source_index: None,
            },
        ],
        categories: vec![CategoryRecord {
            id: 1,
            name: "cat".to_string(),
            color: None,
        }],
        annotations: vec![],
    };

    let bundle = format.export(&dataset, &ExportOptions::default()).unwrap();
    let payload = std::str::from_utf8(&bundle.files[0].bytes).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(payload).unwrap();

    let exported = parsed
        .get("images")
        .and_then(|value| value.as_array())
        .unwrap();
    assert_eq!(exported.len(), 2);
    assert_eq!(exported[0]["file_name"], "images/a.png");
    assert_eq!(exported[1]["file_name"], "nested/folder/b.jpg");
}

#[test]
fn import_accepts_rle_segmentation_and_uses_bbox() {
    let format = CocoFormat::new();
    let bundle = ImportBundle {
        files: vec![BundleFile {
            path: "annotations.json".to_string(),
            mime_type: Some("application/json".to_string()),
            bytes: br#"{
                "images":[{"id":1,"file_name":"a.png","width":10,"height":10}],
                "categories":[{"id":1,"name":"cat","supercategory":"none"}],
                "annotations":[
                    {
                        "id":9,
                        "image_id":1,
                        "category_id":1,
                        "segmentation":{"size":[10,10],"counts":"A1B2"},
                        "bbox":[1,2,3,4],
                        "area":12,
                        "iscrowd":1
                    }
                ]
            }"#
            .to_vec(),
        }],
    };

    let imported = format.import(&bundle, &ImportOptions::default()).unwrap();
    assert_eq!(imported.dataset.annotations.len(), 1);
    assert_eq!(
        imported.dataset.annotations[0].geometry,
        Geometry::BoundingBox {
            x: 1.0,
            y: 2.0,
            w: 3.0,
            h: 4.0
        }
    );
}
