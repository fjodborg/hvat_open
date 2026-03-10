//! SAM3 backend module scaffold.
//!
//! This module currently reuses the stable SAM2 backend internals while
//! providing an independent module/binary path for incremental SAM3 rollout.

pub mod config;
pub mod routes;
pub mod startup;
pub mod state;

pub use crate::full_sam::error::{Error, Result};
pub use config::{CliArgs, ServerConfig};
pub use startup::pregenerate_pyramids;
pub use state::{AppState, build_app_state};
