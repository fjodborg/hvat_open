//! HVAT Axum API Server
//!
//! Backend server for hyperspectral image streaming, annotation persistence,
//! and AI-assisted segmentation (SAM integration).
//!
//! # Architecture
//!
//! The server uses a trait-based architecture for extensibility:
//! - [`loaders::ImageLoader`] - Load different hyperspectral file formats
//! - [`pyramid::PyramidStorage`] - Cache pyramid tiles to disk/memory
//! - [`packer::BandPacker`] - Pack bands into GPU-ready RGBA format
//!
//! # API Endpoints
//!
//! - `GET /api/projects` - List available projects (folders)
//! - `GET /api/projects/:id/images` - List images in a project
//! - `GET /api/images/:id/meta` - Get image metadata and pyramid status
//! - `WS /api/images/:id/stream` - Binary WebSocket for progressive streaming

pub mod config;
pub mod error;
pub mod loaders;
pub mod packer;
pub mod protocol;
pub mod pyramid;
pub mod routes;
pub mod sam;
pub mod startup;
pub mod state;
pub mod utils;

pub use config::ServerConfig;
pub use error::{Error, Result};
pub use startup::pregenerate_pyramids;
pub use state::AppState;
