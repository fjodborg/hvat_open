//! YOLO detection import/export adapter.

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

const POINT_BOX_SIZE: f32 = 1.0;
const META_FILE: &str = "hvat_yolo_detect_images.json";
const CLASSES_FILE: &str = "classes.txt";

/// YOLO detection adapter.
#[derive(Debug, Default, Clone, Copy)]
pub struct YoloDetectFormat;

impl YoloDetectFormat {
    pub fn new() -> Self {
        Self
    }
}

impl AnnotationExporter for YoloDetectFormat {
    fn format_id(&self) -> FormatId {
        FormatId::YoloDetect
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

                    let Some([x, y, w, h]) = to_bbox(ann.id, &ann.geometry, options.shape_policy)?
                    else {
                        continue;
                    };

                    if image.width == 0 || image.height == 0 {
                        return Err(AnnotationIoError::MissingImageContext(format!(
                            "Image '{}' has zero dimensions",
                            image.file_name
                        )));
                    }

                    let [x, y, w, h] = if options.clamp_to_image_bounds {
                        clamp_bbox_to_image(x, y, w, h, image.width as f32, image.height as f32)
                    } else {
                        [x, y, w, h]
                    };

                    let line = format!(
                        "{} {:.6} {:.6} {:.6} {:.6}",
                        class_idx,
                        (x + w / 2.0) / image.width as f32,
                        (y + h / 2.0) / image.height as f32,
                        w / image.width as f32,
                        h / image.height as f32
                    );
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
            AnnotationIoError::Internal(format!("failed to serialize YOLO metadata: {err}"))
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
            supports_bbox: true,
            supports_polygon: false,
            supports_point: false,
            supports_attributes: false,
        }
    }
}

impl AnnotationImporter for YoloDetectFormat {
    fn format_id(&self) -> FormatId {
        FormatId::YoloDetect
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

                match parse_detect_line(line) {
                    Ok((class_idx, cx, cy, w, h)) => {
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
                        let abs_w = w * width;
                        let abs_h = h * height;
                        let abs_x = cx * width - abs_w / 2.0;
                        let abs_y = cy * height - abs_h / 2.0;

                        annotations.push(AnnotationRecord {
                            id: next_ann_id,
                            image_id: meta.id,
                            category_id: class_idx as u64,
                            geometry: Geometry::BoundingBox {
                                x: abs_x,
                                y: abs_y,
                                w: abs_w,
                                h: abs_h,
                            },
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
                    description: "YOLO Detect Import".to_string(),
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
            supports_polygon: false,
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

fn parse_detect_line(line: &str) -> Result<(usize, f32, f32, f32, f32), String> {
    let values = line
        .split_whitespace()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    if values.len() != 5 {
        return Err("expected 5 values: class cx cy w h".to_string());
    }
    let class_idx = values[0]
        .parse::<usize>()
        .map_err(|e| format!("invalid class index: {e}"))?;
    let cx = values[1]
        .parse::<f32>()
        .map_err(|e| format!("invalid cx: {e}"))?;
    let cy = values[2]
        .parse::<f32>()
        .map_err(|e| format!("invalid cy: {e}"))?;
    let w = values[3]
        .parse::<f32>()
        .map_err(|e| format!("invalid w: {e}"))?;
    let h = values[4]
        .parse::<f32>()
        .map_err(|e| format!("invalid h: {e}"))?;
    Ok((class_idx, cx, cy, w, h))
}

fn to_bbox(
    annotation_id: u64,
    geometry: &Geometry,
    policy: ShapeConversionPolicy,
) -> Result<Option<[f32; 4]>, AnnotationIoError> {
    match geometry {
        Geometry::BoundingBox { x, y, w, h } => Ok(Some([*x, *y, *w, *h])),
        Geometry::Polygon { points } => match policy {
            ShapeConversionPolicy::Strict => {
                Err(AnnotationIoError::StrictPolicyViolation(format!(
                    "annotation {} has polygon geometry in YOLO detect",
                    annotation_id
                )))
            }
            ShapeConversionPolicy::AllowLossyWarn => Ok(Some(polygon_bbox(points))),
            ShapeConversionPolicy::SkipUnsupported => Ok(None),
        },
        Geometry::Point { x, y } => match policy {
            ShapeConversionPolicy::Strict => {
                Err(AnnotationIoError::StrictPolicyViolation(format!(
                    "annotation {} has point geometry in YOLO detect",
                    annotation_id
                )))
            }
            ShapeConversionPolicy::AllowLossyWarn => Ok(Some([
                *x - POINT_BOX_SIZE / 2.0,
                *y - POINT_BOX_SIZE / 2.0,
                POINT_BOX_SIZE,
                POINT_BOX_SIZE,
            ])),
            ShapeConversionPolicy::SkipUnsupported => Ok(None),
        },
    }
}

fn polygon_bbox(points: &[(f32, f32)]) -> [f32; 4] {
    if points.is_empty() {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for (x, y) in points {
        min_x = min_x.min(*x);
        min_y = min_y.min(*y);
        max_x = max_x.max(*x);
        max_y = max_y.max(*y);
    }
    [min_x, min_y, max_x - min_x, max_y - min_y]
}

fn clamp_bbox_to_image(x: f32, y: f32, w: f32, h: f32, image_w: f32, image_h: f32) -> [f32; 4] {
    let x1 = x.max(0.0).min(image_w);
    let y1 = y.max(0.0).min(image_h);
    let x2 = (x + w).max(0.0).min(image_w);
    let y2 = (y + h).max(0.0).min(image_h);
    [x1, y1, (x2 - x1).max(0.0), (y2 - y1).max(0.0)]
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
#[path = "yolo_detect.test.rs"]
mod tests;
