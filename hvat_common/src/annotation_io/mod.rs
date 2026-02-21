//! Annotation import/export abstractions and shared format adapters.

pub mod error;
pub mod formats;
pub mod policy;
pub mod registry;
pub mod traits;
pub mod types;

pub use error::AnnotationIoError;
pub use policy::{
    AnnotationMergePolicy, CategoryMergePolicy, ExportOptions, ExportScope, ImportOptions,
    ImportTargetMode, ShapeConversionPolicy,
};
pub use registry::FormatRegistry;
pub use traits::{AnnotationExporter, AnnotationFormat, AnnotationImporter};
pub use types::{
    AnnotationDataset, AnnotationRecord, BundleFile, CategoryRecord, DatasetAnnotationId,
    DatasetCategoryId, DatasetImageId, DatasetInfo, ExportBundle, FormatCapabilities, FormatId,
    Geometry, ImageRecord, ImportBundle, ImportResult, IoReport, IoWarning, IoWarningCode,
};
