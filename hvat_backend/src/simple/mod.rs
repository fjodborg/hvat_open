//! HVAT simple backend mode within `hvat_backend`.
//!
//! Streams images directly from source files (no pyramid caching).
//! Supports PNG, JPEG, TIFF, NPY, and other standard formats.

pub mod routes;
pub mod state;

pub use crate::common::config::{CliArgs, ServerConfig, parse_cli_args};
pub use crate::common::error::{Error, Result};
pub use state::AppState;
