//! Full HVAT backend mode with REST image loading and SAM inference.

pub mod config;
pub mod error;
pub mod inference;
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
