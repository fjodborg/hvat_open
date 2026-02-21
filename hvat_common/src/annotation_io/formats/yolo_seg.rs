//! YOLO segmentation import/export adapter.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::annotation_io::error::AnnotationIoError;
use crate::annotation_io::policy::{ExportOptions, ImportOptions, ShapeConversionPolicy};
use crate::annotation_io::traits::{AnnotationExporter, AnnotationImporter};
use crate::annotation_io::types::{
    AnnotationDataset, AnnotationRecord, BundleFile, CategoryRecord, DatasetInfo, ExportBundle,
    FormatCapabilities, FormatId, Geometry, ImageRecord, ImportBundle, ImportResult, IoReport,
    IoWarning, IoWarningCode,
};

const POINT_POLY_SIZE: f32 = 1.0;
const META_FILE: &str = "hvat_yolo_seg_images.json";
const CLASSES_FILE: &str = "classes.txt";

/// YOLO segmentation adapter.
#[derive(Debug, Default, Clone, Copy)]
pub struct YoloSegFormat;

impl YoloSegFormat {
    pub fn new() -> Self {
        Self
    }
}

impl AnnotationExporter for YoloSegFormat {
    fn format_id(&self) -> FormatId {
        FormatId::YoloSeg
    }

    fn export(
        &self,
        dataset: &AnnotationDataset,
        options: &ExportOptions,
    ) -> Result<ExportBundle, AnnotationIoError> {
        let mut categories = dataset.categories.clone();
        categories.sort_by_key(|cat| cat.id);
        let class_to_index = categories
            .iter()
            .enumerate()
            .map(|(index, cat)| (cat.id, index))
            .collect::<HashMap<_, _>>();

        let classes_payload = categories
            .iter()
            .map(|cat| cat.name.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let mut files = vec![BundleFile {
            path: CLASSES_FILE.to_string(),
            bytes: classes_payload.into_bytes(),
            mime_type: Some("text/plain".to_string()),
        }];

        let mut annotations_per_image: HashMap<u64, Vec<&AnnotationRecord>> = HashMap::new();
        for ann in &dataset.annotations {
            annotations_per_image
                .entry(ann.image_id)
                .or_default()
                .push(ann);
        }

        let mut meta = Vec::new();
        for image in &dataset.images {
            let label_path = format!("labels/{}.txt", image.id);
            let mut lines = Vec::new();

            if let Some(image_annotations) = annotations_per_image.get(&image.id) {
                for ann in image_annotations {
                    let class_idx = class_to_index.get(&ann.category_id).ok_or_else(|| {
                        AnnotationIoError::ValidationError(format!(
                            "Unknown category_id {} for annotation {}",
                            ann.category_id, ann.id
                        ))
                    })?;

                    let Some(mut points) = to_polygon(ann.id, &ann.geometry, options.shape_policy)?
                    else {
                        continue;
                    };

                    if image.width == 0 || image.height == 0 {
                        return Err(AnnotationIoError::MissingImageContext(format!(
                            "Image '{}' has zero dimensions",
                            image.file_name
                        )));
                    }

                    if options.clamp_to_image_bounds {
                        for (x, y) in &mut points {
                            *x = x.max(0.0).min(image.width as f32);
                            *y = y.max(0.0).min(image.height as f32);
                        }
                    }

                    if points.len() < 3 {
                        return Err(AnnotationIoError::ValidationError(format!(
                            "annotation {} resolved to polygon with fewer than 3 points",
                            ann.id
                        )));
                    }

                    let mut line = class_idx.to_string();
                    for (x, y) in points {
                        line.push(' ');
                        line.push_str(&format!(
                            "{:.6} {:.6}",
                            x / image.width as f32,
                            y / image.height as f32
                        ));
                    }
                    lines.push(line);
                }
            }

            if options.include_empty_images || !lines.is_empty() {
                let mut payload = lines.join("\n").into_bytes();
                if !payload.is_empty() {
                    payload.push(b'\n');
                }
                files.push(BundleFile {
                    path: label_path.clone(),
                    bytes: payload,
                    mime_type: Some("text/plain".to_string()),
                });
            }

            meta.push(YoloImageMeta {
                id: image.id,
                file_name: image.file_name.clone(),
                width: image.width,
                height: image.height,
                label_path,
            });
        }

        let meta_bytes = serde_json::to_vec(&meta).map_err(|err| {
            AnnotationIoError::Internal(format!("failed to serialize YOLO seg metadata: {err}"))
        })?;
        files.push(BundleFile {
            path: META_FILE.to_string(),
            bytes: meta_bytes,
            mime_type: Some("application/json".to_string()),
        });

        Ok(ExportBundle { files })
    }

    fn capabilities(&self) -> FormatCapabilities {
        FormatCapabilities {
            supports_multi_image: true,
            supports_bbox: false,
            supports_polygon: true,
            supports_point: false,
            supports_attributes: false,
        }
    }
}

impl AnnotationImporter for YoloSegFormat {
    fn format_id(&self) -> FormatId {
        FormatId::YoloSeg
    }

    #[allow(clippy::too_many_lines)]
    fn import(
        &self,
        bundle: &ImportBundle,
        _options: &ImportOptions,
    ) -> Result<ImportResult, AnnotationIoError> {
        let classes_file = bundle
            .file_by_path(CLASSES_FILE)
            .ok_or_else(|| AnnotationIoError::InvalidBundle("Missing classes.txt".to_string()))?;
        let class_names = std::str::from_utf8(&classes_file.bytes)
            .map_err(|err| AnnotationIoError::InvalidBundle(err.to_string()))?
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();

        let categories = class_names
            .iter()
            .enumerate()
            .map(|(idx, name)| CategoryRecord {
                id: idx as u64,
                name: name.clone(),
                color: None,
            })
            .collect::<Vec<_>>();

        let image_meta = if let Some(meta_file) = bundle.file_by_path(META_FILE) {
            serde_json::from_slice::<Vec<YoloImageMeta>>(&meta_file.bytes).map_err(|err| {
                AnnotationIoError::InvalidBundle(format!("Invalid {META_FILE}: {err}"))
            })?
        } else {
            let label_files = bundle
                .files
                .iter()
                .filter(|file| file.path.ends_with(".txt") && file.path != CLASSES_FILE)
                .collect::<Vec<_>>();
            label_files
                .iter()
                .enumerate()
                .map(|(idx, file)| YoloImageMeta {
                    id: idx as u64 + 1,
                    file_name: file.path.replace("labels/", "").replace(".txt", ".png"),
                    width: 1,
                    height: 1,
                    label_path: file.path.clone(),
                })
                .collect::<Vec<_>>()
        };

        let images = image_meta
            .iter()
            .map(|meta| ImageRecord {
                id: meta.id,
                file_name: meta.file_name.clone(),
                width: meta.width.max(1),
                height: meta.height.max(1),
                source_index: None,
            })
            .collect::<Vec<_>>();

        let mut report = IoReport {
            imported_images: images.len(),
            imported_categories: categories.len(),
            ..IoReport::default()
        };
        if bundle.file_by_path(META_FILE).is_none() {
            report.warnings.push(IoWarning::new(
                IoWarningCode::MissingImageDimensions,
                "Missing YOLO metadata file; imported coordinates remain normalized",
            ));
        }

        let mut annotations = Vec::new();
        let mut next_ann_id = 1u64;

        for meta in &image_meta {
            let Some(label_file) = resolve_label_file(bundle, &meta.label_path) else {
                report.skipped_annotations += 1;
                report.warnings.push(IoWarning::new(
                    IoWarningCode::InvalidRecord,
                    format!("Missing label file '{}'", meta.label_path),
                ));
                continue;
            };
            let text = std::str::from_utf8(&label_file.bytes)
                .map_err(|err| AnnotationIoError::InvalidBundle(err.to_string()))?;

            for (line_idx, line) in text.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                match parse_seg_line(line) {
                    Ok((class_idx, points)) => {
                        if class_idx >= categories.len() {
                            report.skipped_annotations += 1;
                            report.warnings.push(IoWarning::new(
                                IoWarningCode::InvalidRecord,
                                format!(
                                    "Line {} in '{}' references out-of-range class {}",
                                    line_idx + 1,
                                    meta.label_path,
                                    class_idx
                                ),
                            ));
                            continue;
                        }

                        let width = meta.width.max(1) as f32;
                        let height = meta.height.max(1) as f32;
                        let denorm = points
                            .iter()
                            .map(|(x, y)| (x * width, y * height))
                            .collect::<Vec<_>>();

                        annotations.push(AnnotationRecord {
                            id: next_ann_id,
                            image_id: meta.id,
                            category_id: class_idx as u64,
                            geometry: Geometry::Polygon { points: denorm },
                            score: None,
                            attributes: BTreeMap::new(),
                        });
                        next_ann_id += 1;
                    }
                    Err(err) => {
                        report.skipped_annotations += 1;
                        report.warnings.push(IoWarning::new(
                            IoWarningCode::InvalidRecord,
                            format!(
                                "Skipped invalid line {} in '{}': {}",
                                line_idx + 1,
                                meta.label_path,
                                err
                            ),
                        ));
                    }
                }
            }
        }

        report.imported_annotations = annotations.len();

        Ok(ImportResult {
            dataset: AnnotationDataset {
                info: DatasetInfo {
                    description: "YOLO Seg Import".to_string(),
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
            supports_bbox: false,
            supports_polygon: true,
            supports_point: false,
            supports_attributes: false,
        }
    }
}

fn resolve_label_file<'a>(bundle: &'a ImportBundle, label_path: &str) -> Option<&'a BundleFile> {
    bundle
        .file_by_path(label_path)
        .or_else(|| bundle.file_by_path(&label_path.replace(['/', '\\'], "__")))
        .or_else(|| {
            let basename = label_path.rsplit(['/', '\\']).next().unwrap_or(label_path);
            bundle
                .files
                .iter()
                .find(|file| file.path.ends_with(basename))
        })
}

fn parse_seg_line(line: &str) -> Result<(usize, Vec<(f32, f32)>), String> {
    let values = line
        .split_whitespace()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    if values.len() < 7 {
        return Err("expected class + at least 3 xy pairs".to_string());
    }
    if values.len() % 2 == 0 {
        return Err("expected odd number of values".to_string());
    }

    let class_idx = values[0]
        .parse::<usize>()
        .map_err(|e| format!("invalid class index: {e}"))?;
    let mut points = Vec::new();
    let mut i = 1usize;
    while i < values.len() {
        let x = values[i]
            .parse::<f32>()
            .map_err(|e| format!("invalid x at {}: {e}", i + 1))?;
        let y = values[i + 1]
            .parse::<f32>()
            .map_err(|e| format!("invalid y at {}: {e}", i + 2))?;
        points.push((x, y));
        i += 2;
    }
    Ok((class_idx, points))
}

fn to_polygon(
    annotation_id: u64,
    geometry: &Geometry,
    policy: ShapeConversionPolicy,
) -> Result<Option<Vec<(f32, f32)>>, AnnotationIoError> {
    match geometry {
        Geometry::Polygon { points } => Ok(Some(points.clone())),
        Geometry::BoundingBox { x, y, w, h } => match policy {
            ShapeConversionPolicy::Strict => Err(AnnotationIoError::StrictPolicyViolation(
                format!("annotation {} has bbox geometry in YOLO seg", annotation_id),
            )),
            ShapeConversionPolicy::AllowLossyWarn => Ok(Some(vec![
                (*x, *y),
                (*x + *w, *y),
                (*x + *w, *y + *h),
                (*x, *y + *h),
            ])),
            ShapeConversionPolicy::SkipUnsupported => Ok(None),
        },
        Geometry::Point { x, y } => match policy {
            ShapeConversionPolicy::Strict => {
                Err(AnnotationIoError::StrictPolicyViolation(format!(
                    "annotation {} has point geometry in YOLO seg",
                    annotation_id
                )))
            }
            ShapeConversionPolicy::AllowLossyWarn => {
                let half = POINT_POLY_SIZE / 2.0;
                Ok(Some(vec![
                    (*x - half, *y - half),
                    (*x + half, *y - half),
                    (*x + half, *y + half),
                    (*x - half, *y + half),
                ]))
            }
            ShapeConversionPolicy::SkipUnsupported => Ok(None),
        },
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct YoloImageMeta {
    id: u64,
    file_name: String,
    width: u32,
    height: u32,
    label_path: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset_with_mixed_geometries() -> AnnotationDataset {
        AnnotationDataset {
            info: DatasetInfo::default(),
            images: vec![ImageRecord {
                id: 1,
                file_name: "img.png".to_string(),
                width: 120,
                height: 60,
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
                    geometry: Geometry::Polygon {
                        points: vec![(0.0, 0.0), (12.0, 0.0), (12.0, 6.0), (0.0, 6.0)],
                    },
                    score: None,
                    attributes: BTreeMap::new(),
                },
                AnnotationRecord {
                    id: 2,
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
    fn strict_rejects_bbox() {
        let fmt = YoloSegFormat::new();
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
        let fmt = YoloSegFormat::new();
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
        let fmt = YoloSegFormat::new();
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
}
