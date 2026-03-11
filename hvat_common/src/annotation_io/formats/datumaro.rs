//! Datumaro-style JSON import/export adapter.
//!
//! This adapter targets a practical subset of Datumaro's dataset format:
//! `annotations/default.json` with label categories and item annotations.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::annotation_io::error::AnnotationIoError;
use crate::annotation_io::policy::{ExportOptions, ImportOptions, ShapeConversionPolicy};
use crate::annotation_io::traits::{AnnotationExporter, AnnotationImporter};
use crate::annotation_io::types::{
    AnnotationDataset, AnnotationRecord, BundleFile, CategoryRecord, DatasetInfo, ExportBundle,
    FormatCapabilities, FormatId, Geometry, ImageRecord, ImportBundle, ImportResult, IoReport,
    IoWarning, IoWarningCode,
};

const DATUMARO_DEFAULT_FILE: &str = "annotations/default.json";

/// Datumaro adapter.
#[derive(Debug, Default, Clone, Copy)]
pub struct DatumaroFormat;

impl DatumaroFormat {
    pub fn new() -> Self {
        Self
    }
}

impl AnnotationExporter for DatumaroFormat {
    fn format_id(&self) -> FormatId {
        FormatId::Datumaro
    }

    fn export(
        &self,
        dataset: &AnnotationDataset,
        _options: &ExportOptions,
    ) -> Result<ExportBundle, AnnotationIoError> {
        let mut categories_sorted = dataset.categories.clone();
        categories_sorted.sort_by_key(|cat| cat.id);

        let label_categories = categories_sorted
            .iter()
            .map(|cat| DatumaroLabelCategory {
                name: cat.name.clone(),
                parent: String::new(),
                attributes: Vec::new(),
            })
            .collect::<Vec<_>>();
        let cat_to_label = categories_sorted
            .iter()
            .enumerate()
            .map(|(idx, cat)| (cat.id, idx as u64))
            .collect::<HashMap<_, _>>();

        let mut anns_by_image: HashMap<u64, Vec<&AnnotationRecord>> = HashMap::new();
        for ann in &dataset.annotations {
            anns_by_image.entry(ann.image_id).or_default().push(ann);
        }

        let mut items = Vec::new();
        for image in &dataset.images {
            let mut annotations = Vec::new();
            if let Some(records) = anns_by_image.get(&image.id) {
                for record in records {
                    let Some(label) = cat_to_label.get(&record.category_id).copied() else {
                        return Err(AnnotationIoError::ValidationError(format!(
                            "Unknown category_id {} for annotation {}",
                            record.category_id, record.id
                        )));
                    };

                    let kind = match &record.geometry {
                        Geometry::BoundingBox { x, y, w, h } => DatumaroAnnotationKind::BBox {
                            bbox: [*x, *y, *w, *h],
                        },
                        Geometry::Polygon { points } => {
                            let flat = points
                                .iter()
                                .flat_map(|(x, y)| [*x, *y])
                                .collect::<Vec<_>>();
                            DatumaroAnnotationKind::Polygon { points: flat }
                        }
                        Geometry::Point { x, y } => DatumaroAnnotationKind::Points {
                            points: vec![*x, *y],
                        },
                    };

                    annotations.push(DatumaroAnnotation {
                        id: Some(record.id),
                        label: Some(label),
                        group: 0,
                        attributes: record.attributes.clone(),
                        kind,
                    });
                }
            }

            items.push(DatumaroItem {
                id: image.id.to_string(),
                image: Some(DatumaroImageInfo {
                    path: image.file_name.clone(),
                    size: Some([image.height, image.width]),
                }),
                annotations,
                attributes: BTreeMap::new(),
            });
        }

        let root = DatumaroRoot {
            info: DatumaroInfo {
                description: dataset.info.description.clone(),
            },
            categories: DatumaroCategories {
                label: label_categories,
            },
            items,
        };

        let bytes = serde_json::to_vec(&root).map_err(|err| {
            AnnotationIoError::parse(
                FormatId::Datumaro,
                format!("failed to serialize JSON: {err}"),
            )
        })?;

        Ok(ExportBundle {
            files: vec![BundleFile {
                path: DATUMARO_DEFAULT_FILE.to_string(),
                bytes,
                mime_type: Some("application/json".to_string()),
            }],
        })
    }

    fn capabilities(&self) -> FormatCapabilities {
        FormatCapabilities {
            supports_multi_image: true,
            supports_bbox: true,
            supports_polygon: true,
            supports_point: true,
            supports_attributes: true,
        }
    }
}

impl AnnotationImporter for DatumaroFormat {
    fn format_id(&self) -> FormatId {
        FormatId::Datumaro
    }

    #[allow(clippy::too_many_lines)]
    fn import(
        &self,
        bundle: &ImportBundle,
        options: &ImportOptions,
    ) -> Result<ImportResult, AnnotationIoError> {
        let file = bundle
            .file_by_path(DATUMARO_DEFAULT_FILE)
            .or_else(|| bundle.first_by_suffix(".json"))
            .ok_or_else(|| {
                AnnotationIoError::InvalidBundle(
                    "Datumaro import expects annotations/default.json or a JSON file".to_string(),
                )
            })?;

        let root = serde_json::from_slice::<DatumaroRoot>(&file.bytes)
            .map_err(|err| AnnotationIoError::parse(FormatId::Datumaro, err.to_string()))?;

        let categories = root
            .categories
            .label
            .iter()
            .enumerate()
            .map(|(idx, label)| CategoryRecord {
                id: idx as u64,
                name: label.name.clone(),
                color: None,
            })
            .collect::<Vec<_>>();

        let mut report = IoReport {
            imported_categories: categories.len(),
            ..IoReport::default()
        };
        let mut images = Vec::new();
        let mut annotations = Vec::new();
        let mut next_image_id = 1u64;
        let mut next_ann_id = 1u64;

        for item in root.items {
            let image_id = item.id.parse::<u64>().ok().unwrap_or_else(|| {
                let id = next_image_id;
                next_image_id += 1;
                id
            });

            let (file_name, width, height) = match item.image {
                Some(info) => {
                    let file_name = info.path;
                    let (w, h) = match info.size {
                        Some([hh, ww]) => (ww, hh),
                        None => {
                            report.warnings.push(IoWarning::new(
                                IoWarningCode::MissingImageDimensions,
                                format!("Item {} missing image size; defaulting to 1x1", item.id),
                            ));
                            (1, 1)
                        }
                    };
                    (file_name, w.max(1), h.max(1))
                }
                None => {
                    report.warnings.push(IoWarning::new(
                        IoWarningCode::MissingImageDimensions,
                        format!("Item {} missing image info; defaulting to 1x1", item.id),
                    ));
                    (format!("{}.png", item.id), 1, 1)
                }
            };

            images.push(ImageRecord {
                id: image_id,
                file_name,
                width,
                height,
                source_index: None,
            });

            for ann in item.annotations {
                let category_id = ann.label.unwrap_or(0);
                if category_id as usize >= categories.len() {
                    report.skipped_annotations += 1;
                    report.warnings.push(IoWarning::new(
                        IoWarningCode::InvalidRecord,
                        format!(
                            "Item {} annotation {:?} references invalid label {}",
                            item.id, ann.id, category_id
                        ),
                    ));
                    continue;
                }

                let geometry = match annotation_to_geometry(&ann.kind, options.shape_policy) {
                    Ok(Some(geometry)) => geometry,
                    Ok(None) => {
                        report.skipped_annotations += 1;
                        continue;
                    }
                    Err(err) => return Err(err),
                };

                annotations.push(AnnotationRecord {
                    id: ann.id.unwrap_or_else(|| {
                        let id = next_ann_id;
                        next_ann_id += 1;
                        id
                    }),
                    image_id,
                    category_id,
                    geometry,
                    score: None,
                    attributes: ann.attributes,
                });
            }
        }

        report.imported_images = images.len();
        report.imported_annotations = annotations.len();

        Ok(ImportResult {
            dataset: AnnotationDataset {
                info: DatasetInfo {
                    description: root.info.description,
                    ..DatasetInfo::default()
                },
                images,
                categories,
                annotations,
            },
            report,
        })
    }

    fn capabilities(&self) -> FormatCapabilities {
        FormatCapabilities {
            supports_multi_image: true,
            supports_bbox: true,
            supports_polygon: true,
            supports_point: true,
            supports_attributes: true,
        }
    }
}

fn annotation_to_geometry(
    annotation: &DatumaroAnnotationKind,
    policy: ShapeConversionPolicy,
) -> Result<Option<Geometry>, AnnotationIoError> {
    match annotation {
        DatumaroAnnotationKind::BBox { bbox } => Ok(Some(Geometry::BoundingBox {
            x: bbox[0],
            y: bbox[1],
            w: bbox[2],
            h: bbox[3],
        })),
        DatumaroAnnotationKind::Polygon { points } => {
            if points.len() < 6 || points.len() % 2 != 0 {
                return Err(AnnotationIoError::ValidationError(
                    "Datumaro polygon must have an even number of values >= 6".to_string(),
                ));
            }
            let pairs = points
                .chunks(2)
                .map(|chunk| (chunk[0], chunk[1]))
                .collect::<Vec<_>>();
            Ok(Some(Geometry::Polygon { points: pairs }))
        }
        DatumaroAnnotationKind::Points { points } => {
            if points.len() < 2 {
                return Err(AnnotationIoError::ValidationError(
                    "Datumaro points must contain at least one point".to_string(),
                ));
            }

            if points.len() == 2 {
                return Ok(Some(Geometry::Point {
                    x: points[0],
                    y: points[1],
                }));
            }

            match policy {
                ShapeConversionPolicy::Strict => Err(AnnotationIoError::StrictPolicyViolation(
                    "Datumaro multi-point annotation cannot map to a single Point".to_string(),
                )),
                ShapeConversionPolicy::AllowLossyWarn => Ok(Some(Geometry::Point {
                    x: points[0],
                    y: points[1],
                })),
                ShapeConversionPolicy::SkipUnsupported => Ok(None),
            }
        }
        DatumaroAnnotationKind::Unknown => match policy {
            ShapeConversionPolicy::Strict => Err(AnnotationIoError::StrictPolicyViolation(
                "Unknown Datumaro annotation type".to_string(),
            )),
            ShapeConversionPolicy::AllowLossyWarn | ShapeConversionPolicy::SkipUnsupported => {
                Ok(None)
            }
        },
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct DatumaroRoot {
    #[serde(default)]
    info: DatumaroInfo,
    categories: DatumaroCategories,
    #[serde(default)]
    items: Vec<DatumaroItem>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct DatumaroInfo {
    #[serde(default)]
    description: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct DatumaroCategories {
    #[serde(default)]
    label: Vec<DatumaroLabelCategory>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DatumaroLabelCategory {
    name: String,
    #[serde(default)]
    parent: String,
    #[serde(default)]
    attributes: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DatumaroItem {
    id: String,
    #[serde(default)]
    image: Option<DatumaroImageInfo>,
    #[serde(default)]
    annotations: Vec<DatumaroAnnotation>,
    #[serde(default)]
    attributes: BTreeMap<String, Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DatumaroImageInfo {
    path: String,
    /// Datumaro uses [height, width]
    #[serde(default)]
    size: Option<[u32; 2]>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DatumaroAnnotation {
    #[serde(default)]
    id: Option<u64>,
    #[serde(default)]
    label: Option<u64>,
    #[serde(default)]
    group: u64,
    #[serde(default)]
    attributes: BTreeMap<String, Value>,
    #[serde(flatten)]
    kind: DatumaroAnnotationKind,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum DatumaroAnnotationKind {
    BBox {
        bbox: [f32; 4],
    },
    Polygon {
        points: Vec<f32>,
    },
    Points {
        points: Vec<f32>,
    },
    #[serde(other)]
    Unknown,
}

#[cfg(test)]
#[path = "datumaro.test.rs"]
mod tests;
