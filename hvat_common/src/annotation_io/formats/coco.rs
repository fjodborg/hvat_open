//! COCO format adapter.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::annotation_io::error::AnnotationIoError;
use crate::annotation_io::policy::{ExportOptions, ImportOptions};
use crate::annotation_io::traits::{AnnotationExporter, AnnotationImporter};
use crate::annotation_io::types::{
    AnnotationDataset, AnnotationRecord, BundleFile, CategoryRecord, DatasetInfo, ExportBundle,
    FormatCapabilities, FormatId, Geometry, ImageRecord, ImportBundle, ImportResult, IoReport,
    IoWarning, IoWarningCode,
};

const POINT_BOX_SIZE: f32 = 1.0;
type CocoGeometryFields = (
    Option<[f32; 4]>,
    Option<CocoSegmentation>,
    f32,
    Option<[f32; 2]>,
);

/// COCO import/export implementation.
#[derive(Debug, Default, Clone, Copy)]
pub struct CocoFormat;

impl CocoFormat {
    pub fn new() -> Self {
        Self
    }

    fn find_json_file<'a>(
        &self,
        bundle: &'a ImportBundle,
    ) -> Result<&'a BundleFile, AnnotationIoError> {
        if bundle.files.is_empty() {
            return Err(AnnotationIoError::InvalidBundle(
                "COCO import expects at least one file".to_string(),
            ));
        }

        bundle
            .first_by_suffix(".json")
            .or_else(|| bundle.files.first())
            .ok_or_else(|| {
                AnnotationIoError::InvalidBundle(
                    "COCO import bundle had no readable files".to_string(),
                )
            })
    }

    fn map_geometry_to_coco(geometry: &Geometry) -> Result<CocoGeometryFields, AnnotationIoError> {
        match geometry {
            Geometry::BoundingBox { x, y, w, h } => {
                Ok((Some([*x, *y, *w, *h]), None, *w * *h, None))
            }
            Geometry::Polygon { points } => {
                if points.len() < 3 {
                    return Err(AnnotationIoError::ValidationError(
                        "Polygon must contain at least 3 points".to_string(),
                    ));
                }
                let segmentation: Vec<f32> = points.iter().flat_map(|(x, y)| [*x, *y]).collect();
                let bbox = polygon_bbox(points);
                let area = polygon_area(points);
                Ok((
                    Some(bbox),
                    Some(CocoSegmentation::Polygons(vec![segmentation])),
                    area,
                    None,
                ))
            }
            Geometry::Point { x, y } => Ok((
                Some([
                    *x - POINT_BOX_SIZE / 2.0,
                    *y - POINT_BOX_SIZE / 2.0,
                    POINT_BOX_SIZE,
                    POINT_BOX_SIZE,
                ]),
                None,
                1.0,
                Some([*x, *y]),
            )),
        }
    }

    fn map_coco_to_geometry(annotation: &CocoAnnotation) -> Result<Geometry, AnnotationIoError> {
        if let Some([x, y]) = annotation.hvat_point {
            return Ok(Geometry::Point { x, y });
        }

        if let Some(segmentation) = &annotation.segmentation
            && let CocoSegmentation::Polygons(polygons) = segmentation
            && let Some(first_segment) = polygons.first()
        {
            // Legacy HVAT export used 2-coordinate segments for points.
            if first_segment.len() == 2 {
                return Ok(Geometry::Point {
                    x: first_segment[0],
                    y: first_segment[1],
                });
            }

            if first_segment.len() >= 6 {
                if first_segment.len() % 2 != 0 {
                    return Err(AnnotationIoError::ValidationError(
                        "Segmentation polygon has odd number of coordinates".to_string(),
                    ));
                }
                let points = first_segment
                    .chunks(2)
                    .map(|chunk| (chunk[0], chunk[1]))
                    .collect::<Vec<_>>();
                return Ok(Geometry::Polygon { points });
            }
        }

        if let Some([x, y, w, h]) = annotation.bbox {
            return Ok(Geometry::BoundingBox { x, y, w, h });
        }

        Err(AnnotationIoError::ValidationError(
            "COCO annotation did not contain bbox or usable segmentation".to_string(),
        ))
    }
}

impl AnnotationExporter for CocoFormat {
    fn format_id(&self) -> FormatId {
        FormatId::Coco
    }

    fn export(
        &self,
        dataset: &AnnotationDataset,
        _options: &ExportOptions,
    ) -> Result<ExportBundle, AnnotationIoError> {
        let images = dataset
            .images
            .iter()
            .map(|img| CocoImage {
                id: img.id,
                file_name: img.file_name.clone(),
                width: img.width,
                height: img.height,
                license: None,
                coco_url: None,
                flickr_url: None,
                date_captured: None,
            })
            .collect();

        let categories = dataset
            .categories
            .iter()
            .map(|cat| CocoCategory {
                id: cat.id,
                name: cat.name.clone(),
                supercategory: "none".to_string(),
            })
            .collect();

        let mut annotations = Vec::with_capacity(dataset.annotations.len());
        for ann in &dataset.annotations {
            let (bbox, segmentation, area, hvat_point) = Self::map_geometry_to_coco(&ann.geometry)?;
            annotations.push(CocoAnnotation {
                id: ann.id,
                image_id: ann.image_id,
                category_id: ann.category_id,
                bbox,
                segmentation,
                area,
                iscrowd: 0,
                hvat_point,
            });
        }

        let coco = CocoDataset {
            info: CocoInfo {
                year: dataset.info.year.unwrap_or_default(),
                version: dataset.info.version.clone(),
                description: dataset.info.description.clone(),
                contributor: dataset.info.contributor.clone(),
                date_created: dataset.info.date_created.clone(),
                ..CocoInfo::default()
            },
            images,
            annotations,
            categories,
            licenses: Vec::new(),
        };

        let bytes = serde_json::to_vec(&coco).map_err(|err| {
            AnnotationIoError::parse(
                FormatId::Coco,
                format!("failed to serialize COCO JSON: {err}"),
            )
        })?;

        Ok(ExportBundle {
            files: vec![BundleFile {
                path: "annotations.json".to_string(),
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
            supports_attributes: false,
        }
    }
}

impl AnnotationImporter for CocoFormat {
    fn format_id(&self) -> FormatId {
        FormatId::Coco
    }

    fn import(
        &self,
        bundle: &ImportBundle,
        _options: &ImportOptions,
    ) -> Result<ImportResult, AnnotationIoError> {
        let file = self.find_json_file(bundle)?;
        let text = std::str::from_utf8(&file.bytes).map_err(|err| {
            AnnotationIoError::InvalidBundle(format!("COCO file is not UTF-8: {err}"))
        })?;
        let coco: CocoDataset = serde_json::from_str(text)
            .map_err(|err| AnnotationIoError::parse(FormatId::Coco, err.to_string()))?;

        let images = coco
            .images
            .into_iter()
            .map(|img| ImageRecord {
                id: img.id,
                file_name: img.file_name,
                width: img.width,
                height: img.height,
                source_index: None,
            })
            .collect::<Vec<_>>();

        let categories = coco
            .categories
            .into_iter()
            .map(|cat| CategoryRecord {
                id: cat.id,
                name: cat.name,
                color: None,
            })
            .collect::<Vec<_>>();

        let mut annotations = Vec::new();
        let mut warnings = Vec::new();
        let mut skipped_annotations = 0usize;

        for ann in coco.annotations {
            match Self::map_coco_to_geometry(&ann) {
                Ok(geometry) => {
                    annotations.push(AnnotationRecord {
                        id: ann.id,
                        image_id: ann.image_id,
                        category_id: ann.category_id,
                        geometry,
                        score: None,
                        attributes: BTreeMap::new(),
                    });
                }
                Err(err) => {
                    skipped_annotations += 1;
                    let mut warning = IoWarning::new(
                        IoWarningCode::InvalidRecord,
                        format!("Skipped invalid COCO annotation {}: {}", ann.id, err),
                    );
                    warning
                        .context
                        .insert("annotation_id".to_string(), serde_json::Value::from(ann.id));
                    warnings.push(warning);
                }
            }
        }

        let dataset = AnnotationDataset {
            info: DatasetInfo {
                description: coco.info.description,
                version: coco.info.version,
                year: Some(coco.info.year),
                contributor: coco.info.contributor,
                date_created: coco.info.date_created,
            },
            images,
            categories,
            annotations,
        };

        let report = IoReport {
            imported_images: dataset.images.len(),
            imported_annotations: dataset.annotations.len(),
            imported_categories: dataset.categories.len(),
            skipped_annotations,
            warnings,
        };

        Ok(ImportResult { dataset, report })
    }

    fn capabilities(&self) -> FormatCapabilities {
        FormatCapabilities {
            supports_multi_image: true,
            supports_bbox: true,
            supports_polygon: true,
            supports_point: true,
            supports_attributes: false,
        }
    }
}

fn polygon_bbox(vertices: &[(f32, f32)]) -> [f32; 4] {
    if vertices.is_empty() {
        return [0.0, 0.0, 0.0, 0.0];
    }

    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    for (x, y) in vertices {
        min_x = min_x.min(*x);
        min_y = min_y.min(*y);
        max_x = max_x.max(*x);
        max_y = max_y.max(*y);
    }

    [min_x, min_y, max_x - min_x, max_y - min_y]
}

fn polygon_area(vertices: &[(f32, f32)]) -> f32 {
    if vertices.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0f32;
    for i in 0..vertices.len() {
        let j = (i + 1) % vertices.len();
        area += vertices[i].0 * vertices[j].1;
        area -= vertices[j].0 * vertices[i].1;
    }

    (area / 2.0).abs()
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct CocoDataset {
    #[serde(default)]
    info: CocoInfo,
    #[serde(default)]
    images: Vec<CocoImage>,
    #[serde(default)]
    annotations: Vec<CocoAnnotation>,
    #[serde(default)]
    categories: Vec<CocoCategory>,
    #[serde(default)]
    licenses: Vec<CocoLicense>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct CocoInfo {
    #[serde(default)]
    year: u32,
    #[serde(default)]
    version: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    contributor: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    date_created: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct CocoImage {
    id: u64,
    file_name: String,
    width: u32,
    height: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    license: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    coco_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    flickr_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    date_captured: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CocoAnnotation {
    id: u64,
    image_id: u64,
    category_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    bbox: Option<[f32; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    segmentation: Option<CocoSegmentation>,
    area: f32,
    iscrowd: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    hvat_point: Option<[f32; 2]>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum CocoSegmentation {
    Polygons(Vec<Vec<f32>>),
    Rle(CocoRle),
}

#[derive(Debug, Serialize, Deserialize)]
struct CocoRle {
    #[serde(default)]
    size: Vec<u32>,
    #[serde(default)]
    counts: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct CocoCategory {
    id: u64,
    name: String,
    supercategory: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct CocoLicense {
    id: u32,
    name: String,
    url: String,
}

#[cfg(test)]
mod tests {
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
}
