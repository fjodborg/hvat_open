//! Runtime registry for annotation import/export adapters.

use std::collections::HashMap;
use std::sync::Arc;

use super::error::AnnotationIoError;
use super::policy::{ExportOptions, ImportOptions};
use super::traits::{AnnotationExporter, AnnotationFormat, AnnotationImporter};
use super::types::{
    AnnotationDataset, ExportBundle, FormatCapabilities, FormatId, ImportBundle, ImportResult,
};

/// Registry of importers and exporters keyed by format.
#[derive(Default)]
pub struct FormatRegistry {
    exporters: HashMap<FormatId, Arc<dyn AnnotationExporter>>,
    importers: HashMap<FormatId, Arc<dyn AnnotationImporter>>,
}

impl FormatRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an exporter for a format.
    pub fn register_exporter(&mut self, exporter: Arc<dyn AnnotationExporter>) {
        self.exporters.insert(exporter.format_id(), exporter);
    }

    /// Register an importer for a format.
    pub fn register_importer(&mut self, importer: Arc<dyn AnnotationImporter>) {
        self.importers.insert(importer.format_id(), importer);
    }

    /// Register one adapter that supports both import and export.
    pub fn register_format<T>(&mut self, format: T)
    where
        T: AnnotationFormat + 'static,
    {
        let adapter = Arc::new(format);
        self.register_exporter(adapter.clone());
        self.register_importer(adapter);
    }

    /// Dispatch export to the format adapter.
    pub fn export(
        &self,
        format_id: FormatId,
        dataset: &AnnotationDataset,
        options: &ExportOptions,
    ) -> Result<ExportBundle, AnnotationIoError> {
        let exporter = self
            .exporters
            .get(&format_id)
            .ok_or(AnnotationIoError::UnsupportedFormat(format_id))?;
        exporter.export(dataset, options)
    }

    /// Dispatch import to the format adapter.
    pub fn import(
        &self,
        format_id: FormatId,
        bundle: &ImportBundle,
        options: &ImportOptions,
    ) -> Result<ImportResult, AnnotationIoError> {
        let importer = self
            .importers
            .get(&format_id)
            .ok_or(AnnotationIoError::UnsupportedFormat(format_id))?;
        importer.import(bundle, options)
    }

    /// Return exportable formats.
    pub fn export_formats(&self) -> Vec<FormatId> {
        let mut ids = self.exporters.keys().copied().collect::<Vec<_>>();
        ids.sort_by_key(|id| *id as u8);
        ids
    }

    /// Return importable formats.
    pub fn import_formats(&self) -> Vec<FormatId> {
        let mut ids = self.importers.keys().copied().collect::<Vec<_>>();
        ids.sort_by_key(|id| *id as u8);
        ids
    }

    /// Return merged capabilities for a format if any adapter is registered.
    pub fn capabilities(&self, format_id: FormatId) -> Option<FormatCapabilities> {
        let exporter = self.exporters.get(&format_id).map(|e| e.capabilities());
        let importer = self.importers.get(&format_id).map(|i| i.capabilities());

        match (exporter, importer) {
            (None, None) => None,
            (Some(cap), None) | (None, Some(cap)) => Some(cap),
            (Some(a), Some(b)) => Some(FormatCapabilities {
                supports_multi_image: a.supports_multi_image || b.supports_multi_image,
                supports_bbox: a.supports_bbox || b.supports_bbox,
                supports_polygon: a.supports_polygon || b.supports_polygon,
                supports_point: a.supports_point || b.supports_point,
                supports_attributes: a.supports_attributes || b.supports_attributes,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use super::*;
    use crate::annotation_io::formats::coco::CocoFormat;
    use crate::annotation_io::formats::datumaro::DatumaroFormat;
    use crate::annotation_io::formats::yolo_detect::YoloDetectFormat;
    use crate::annotation_io::formats::yolo_seg::YoloSegFormat;
    use crate::annotation_io::policy::{ExportOptions, ImportOptions};
    use crate::annotation_io::types::{
        AnnotationRecord, CategoryRecord, DatasetInfo, Geometry, ImageRecord, ImportBundle,
    };

    const EPSILON: f32 = 1e-3;

    fn registry_with_formats() -> FormatRegistry {
        let mut registry = FormatRegistry::new();
        registry.register_format(CocoFormat::new());
        registry.register_format(DatumaroFormat::new());
        registry.register_format(YoloDetectFormat::new());
        registry.register_format(YoloSegFormat::new());
        registry
    }

    fn sort_dataset(dataset: &mut AnnotationDataset) {
        dataset.images.sort_by_key(|image| image.id);
        dataset.categories.sort_by_key(|category| category.id);
        dataset
            .annotations
            .sort_by_key(|annotation| (annotation.image_id, annotation.id));
    }

    fn assert_f32_close(expected: f32, actual: f32, label: &str) {
        let delta = (expected - actual).abs();
        assert!(
            delta <= EPSILON,
            "{label} mismatch: expected {expected}, actual {actual}, |delta|={delta}"
        );
    }

    fn assert_geometry_close(expected: &Geometry, actual: &Geometry) {
        match (expected, actual) {
            (
                Geometry::BoundingBox {
                    x: ex,
                    y: ey,
                    w: ew,
                    h: eh,
                },
                Geometry::BoundingBox {
                    x: ax,
                    y: ay,
                    w: aw,
                    h: ah,
                },
            ) => {
                assert_f32_close(*ex, *ax, "bbox.x");
                assert_f32_close(*ey, *ay, "bbox.y");
                assert_f32_close(*ew, *aw, "bbox.w");
                assert_f32_close(*eh, *ah, "bbox.h");
            }
            (Geometry::Polygon { points: expected }, Geometry::Polygon { points: actual }) => {
                assert_eq!(
                    expected.len(),
                    actual.len(),
                    "polygon vertex count mismatch"
                );
                for ((ex, ey), (ax, ay)) in expected.iter().zip(actual.iter()) {
                    assert_f32_close(*ex, *ax, "polygon.x");
                    assert_f32_close(*ey, *ay, "polygon.y");
                }
            }
            (Geometry::Point { x: ex, y: ey }, Geometry::Point { x: ax, y: ay }) => {
                assert_f32_close(*ex, *ax, "point.x");
                assert_f32_close(*ey, *ay, "point.y");
            }
            (expected, actual) => {
                panic!("geometry type mismatch: expected {expected:?}, got {actual:?}")
            }
        }
    }

    fn assert_round_trip(
        registry: &FormatRegistry,
        format_id: FormatId,
        dataset: AnnotationDataset,
    ) {
        let bundle = registry
            .export(format_id, &dataset, &ExportOptions::default())
            .expect("export should succeed");
        let imported = registry
            .import(
                format_id,
                &ImportBundle {
                    files: bundle.files,
                },
                &ImportOptions::default(),
            )
            .expect("import should succeed");

        assert_eq!(
            imported.report.skipped_annotations, 0,
            "round-trip should not skip annotations for {format_id:?}"
        );
        assert_eq!(
            imported.report.imported_annotations,
            dataset.annotations.len(),
            "imported annotation count mismatch in report for {format_id:?}"
        );

        let mut expected = dataset;
        let mut actual = imported.dataset;
        sort_dataset(&mut expected);
        sort_dataset(&mut actual);

        let expected_images = normalize_images(&expected.images);
        let actual_images = normalize_images(&actual.images);
        assert_eq!(
            expected_images, actual_images,
            "images mismatch for {format_id:?}"
        );
        let mut expected_categories = expected
            .categories
            .iter()
            .map(|category| category.name.clone())
            .collect::<Vec<_>>();
        expected_categories.sort_unstable();
        let mut actual_categories = actual
            .categories
            .iter()
            .map(|category| category.name.clone())
            .collect::<Vec<_>>();
        actual_categories.sort_unstable();
        assert_eq!(
            expected_categories, actual_categories,
            "categories mismatch for {format_id:?}"
        );

        let expected_annotations = semantic_annotations(&expected);
        let actual_annotations = semantic_annotations(&actual);
        assert_eq!(
            expected_annotations.len(),
            actual_annotations.len(),
            "annotation count mismatch for {format_id:?}"
        );

        for (expected, actual) in expected_annotations.iter().zip(actual_annotations.iter()) {
            assert_eq!(
                expected.image_file_name, actual.image_file_name,
                "annotation image mapping mismatch for {format_id:?}"
            );
            assert_eq!(
                expected.category_name, actual.category_name,
                "annotation category mapping mismatch for {format_id:?}"
            );
            assert_geometry_close(&expected.geometry, &actual.geometry);
        }
    }

    fn normalize_images(images: &[ImageRecord]) -> Vec<(String, u32, u32)> {
        let mut normalized = images
            .iter()
            .map(|image| (image.file_name.clone(), image.width, image.height))
            .collect::<Vec<_>>();
        normalized.sort_unstable();
        normalized
    }

    #[derive(Debug, Clone)]
    struct SemanticAnnotation {
        image_file_name: String,
        category_name: String,
        geometry: Geometry,
    }

    fn semantic_annotations(dataset: &AnnotationDataset) -> Vec<SemanticAnnotation> {
        let image_by_id = dataset
            .images
            .iter()
            .map(|image| (image.id, image.file_name.clone()))
            .collect::<HashMap<_, _>>();
        let category_by_id = dataset
            .categories
            .iter()
            .map(|category| (category.id, category.name.clone()))
            .collect::<HashMap<_, _>>();

        let mut annotations = dataset
            .annotations
            .iter()
            .map(|annotation| SemanticAnnotation {
                image_file_name: image_by_id
                    .get(&annotation.image_id)
                    .cloned()
                    .unwrap_or_else(|| format!("image#{}", annotation.image_id)),
                category_name: category_by_id
                    .get(&annotation.category_id)
                    .cloned()
                    .unwrap_or_else(|| format!("category#{}", annotation.category_id)),
                geometry: annotation.geometry.clone(),
            })
            .collect::<Vec<_>>();

        annotations.sort_by(|left, right| {
            let left_key = semantic_annotation_key(left);
            let right_key = semantic_annotation_key(right);
            left_key.cmp(&right_key)
        });
        annotations
    }

    fn semantic_annotation_key(annotation: &SemanticAnnotation) -> String {
        format!(
            "{}|{}|{}",
            annotation.image_file_name,
            annotation.category_name,
            geometry_key(&annotation.geometry)
        )
    }

    fn geometry_key(geometry: &Geometry) -> String {
        match geometry {
            Geometry::BoundingBox { x, y, w, h } => format!("bbox:{x:.3}:{y:.3}:{w:.3}:{h:.3}"),
            Geometry::Point { x, y } => format!("point:{x:.3}:{y:.3}"),
            Geometry::Polygon { points } => {
                let coords = points
                    .iter()
                    .map(|(x, y)| format!("{x:.3}:{y:.3}"))
                    .collect::<Vec<_>>()
                    .join(";");
                format!("poly:{coords}")
            }
        }
    }

    fn mixed_shape_dataset() -> AnnotationDataset {
        AnnotationDataset {
            info: DatasetInfo::default(),
            images: vec![ImageRecord {
                id: 1,
                file_name: "images/frame_0001.png".to_string(),
                width: 640,
                height: 480,
                source_index: None,
            }],
            categories: vec![CategoryRecord {
                id: 1,
                name: "object".to_string(),
                color: None,
            }],
            annotations: vec![
                AnnotationRecord {
                    id: 1,
                    image_id: 1,
                    category_id: 1,
                    geometry: Geometry::BoundingBox {
                        x: 12.0,
                        y: 24.0,
                        w: 100.0,
                        h: 80.0,
                    },
                    score: None,
                    attributes: BTreeMap::new(),
                },
                AnnotationRecord {
                    id: 2,
                    image_id: 1,
                    category_id: 1,
                    geometry: Geometry::Polygon {
                        points: vec![(10.0, 10.0), (40.0, 10.0), (40.0, 30.0), (10.0, 30.0)],
                    },
                    score: None,
                    attributes: BTreeMap::new(),
                },
                AnnotationRecord {
                    id: 3,
                    image_id: 1,
                    category_id: 1,
                    geometry: Geometry::Point { x: 55.0, y: 65.0 },
                    score: None,
                    attributes: BTreeMap::new(),
                },
            ],
        }
    }

    fn yolo_detect_dataset() -> AnnotationDataset {
        AnnotationDataset {
            info: DatasetInfo::default(),
            images: vec![ImageRecord {
                id: 1,
                file_name: "img_detect.png".to_string(),
                width: 100,
                height: 100,
                source_index: None,
            }],
            categories: vec![
                CategoryRecord {
                    id: 0,
                    name: "car".to_string(),
                    color: None,
                },
                CategoryRecord {
                    id: 1,
                    name: "person".to_string(),
                    color: None,
                },
            ],
            annotations: vec![
                AnnotationRecord {
                    id: 1,
                    image_id: 1,
                    category_id: 0,
                    geometry: Geometry::BoundingBox {
                        x: 10.0,
                        y: 15.0,
                        w: 20.0,
                        h: 25.0,
                    },
                    score: None,
                    attributes: BTreeMap::new(),
                },
                AnnotationRecord {
                    id: 2,
                    image_id: 1,
                    category_id: 1,
                    geometry: Geometry::BoundingBox {
                        x: 50.0,
                        y: 60.0,
                        w: 10.0,
                        h: 12.0,
                    },
                    score: None,
                    attributes: BTreeMap::new(),
                },
            ],
        }
    }

    fn yolo_seg_dataset() -> AnnotationDataset {
        AnnotationDataset {
            info: DatasetInfo::default(),
            images: vec![ImageRecord {
                id: 1,
                file_name: "img_seg.png".to_string(),
                width: 100,
                height: 100,
                source_index: None,
            }],
            categories: vec![CategoryRecord {
                id: 0,
                name: "shape".to_string(),
                color: None,
            }],
            annotations: vec![AnnotationRecord {
                id: 1,
                image_id: 1,
                category_id: 0,
                geometry: Geometry::Polygon {
                    points: vec![(10.0, 10.0), (60.0, 10.0), (60.0, 40.0), (10.0, 40.0)],
                },
                score: None,
                attributes: BTreeMap::new(),
            }],
        }
    }

    #[test]
    fn registry_round_trip_coco() {
        let registry = registry_with_formats();
        assert_round_trip(&registry, FormatId::Coco, mixed_shape_dataset());
    }

    #[test]
    fn registry_round_trip_datumaro() {
        let registry = registry_with_formats();
        assert_round_trip(&registry, FormatId::Datumaro, mixed_shape_dataset());
    }

    #[test]
    fn registry_round_trip_yolo_detect() {
        let registry = registry_with_formats();
        assert_round_trip(&registry, FormatId::YoloDetect, yolo_detect_dataset());
    }

    #[test]
    fn registry_round_trip_yolo_seg() {
        let registry = registry_with_formats();
        assert_round_trip(&registry, FormatId::YoloSeg, yolo_seg_dataset());
    }
}
