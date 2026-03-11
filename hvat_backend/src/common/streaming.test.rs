use super::*;
use crate::common::config::ServerConfig;

#[test]
fn default_capabilities_use_single_zip_download_mode() {
    let caps = build_default_capabilities(&ServerConfig::default());
    assert!(caps.features.downloads);
    assert_eq!(caps.download_mode, hvat_common::DownloadMode::SingleZip);
    assert!(!caps.supports_chunked_downloads());
}
