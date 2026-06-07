//! REST SAM inference routes.

use std::sync::Arc;

use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::error::Error;
use crate::full_sam::inference::ImageContext;
use crate::full_sam::routes::sam_service::{
    SamBands, load_image_rgb_with_bands, resolve_sam_bands,
};
use crate::state::AppState;

#[derive(Debug, Deserialize)]
struct SamInferRequest {
    model_id: String,
    image_id: String,
    #[serde(default)]
    bands: SamBandsRequest,
    inputs: SamInputsRequest,
}

#[derive(Debug, Deserialize)]
struct SamWarmRequest {
    model_id: String,
    image_id: String,
    #[serde(default)]
    bands: SamBandsRequest,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct SamBandsRequest {
    #[serde(default)]
    r: u32,
    #[serde(default = "default_green")]
    g: u32,
    #[serde(default = "default_blue")]
    b: u32,
}

const fn default_green() -> u32 {
    1
}

const fn default_blue() -> u32 {
    2
}

impl Default for SamBandsRequest {
    fn default() -> Self {
        Self { r: 0, g: 1, b: 2 }
    }
}

impl From<SamBandsRequest> for SamBands {
    fn from(value: SamBandsRequest) -> Self {
        Self {
            red: value.r,
            green: value.g,
            blue: value.b,
        }
    }
}

#[derive(Debug, Deserialize)]
struct SamInputsRequest {
    #[serde(default)]
    points: Value,
    #[serde(default)]
    labels: Option<Vec<i32>>,
    #[serde(default, rename = "box")]
    box_prompt: Option<[f32; 4]>,
}

#[derive(Debug, Serialize)]
struct SamInferResponse {
    masks: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    scores: Option<Value>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    code: &'static str,
}

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/infer", post(post_infer))
        .route("/warm", post(post_warm))
}

async fn post_infer(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SamInferRequest>,
) -> Result<Json<SamInferResponse>, (StatusCode, Json<ErrorResponse>)> {
    let started = std::time::Instant::now();
    let outcome: Result<_, (StatusCode, Json<ErrorResponse>)> = async {
        let registry = state.model_registry.as_ref().ok_or_else(|| {
            api_err(
                StatusCode::SERVICE_UNAVAILABLE,
                "No models available on this server",
                "model_unavailable",
            )
        })?;

        let backend = registry.get(&request.model_id).ok_or_else(|| {
            api_err(
                StatusCode::BAD_REQUEST,
                format!("Model '{}' not found", request.model_id),
                "model_not_found",
            )
        })?;

        let normalized_inputs = normalize_inputs(&request.inputs)
            .map_err(|message| api_err(StatusCode::BAD_REQUEST, message, "invalid_inputs"))?;
        backend.validate_inputs(&normalized_inputs).map_err(|e| {
            api_err(
                StatusCode::BAD_REQUEST,
                format!("Invalid inputs: {}", e),
                "invalid_inputs",
            )
        })?;

        let requested_bands = SamBands::from(request.bands);
        let (resolved_bands, width, height) =
            resolve_sam_bands(&state, &request.image_id, requested_bands)
                .await
                .map_err(map_image_error)?;
        let embedding_key = resolved_bands.embedding_key(&request.image_id);

        if backend.requires_embedding() && !backend.is_prepared(&embedding_key).await {
            let (rgb_data, prep_w, prep_h, _) = load_image_rgb_with_bands(
                &state,
                &request.image_id,
                resolved_bands.red,
                resolved_bands.green,
                resolved_bands.blue,
            )
            .await
            .map_err(map_image_error)?;

            let prepare_image = ImageContext {
                image_id: embedding_key.clone(),
                rgb_data,
                width: prep_w,
                height: prep_h,
            };

            backend.prepare(&prepare_image, None).await.map_err(|e| {
                api_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to prepare model: {}", e),
                    "model_prepare_failed",
                )
            })?;
        }

        let infer_image = ImageContext {
            image_id: if backend.requires_embedding() {
                embedding_key
            } else {
                request.image_id.clone()
            },
            rgb_data: Vec::new(),
            width,
            height,
        };
        let options = json!({
            "bands": {
                "red": resolved_bands.red,
                "green": resolved_bands.green,
                "blue": resolved_bands.blue
            }
        });

        let result = backend
            .infer(&infer_image, normalized_inputs, options, None)
            .await
            .map_err(|e| {
                api_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Inference failed: {}", e),
                    "inference_failed",
                )
            })?;

        let masks = result
            .outputs
            .get("masks")
            .cloned()
            .unwrap_or_else(|| Value::Array(Vec::new()));
        let scores = result.outputs.get("scores").cloned();

        Ok(Json(SamInferResponse { masks, scores }))
    }
    .await;

    let elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    state
        .rest_metrics
        .record_sam_infer(elapsed_ms, outcome.is_ok());
    outcome
}

async fn post_warm(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SamWarmRequest>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    let started = std::time::Instant::now();
    let outcome: Result<_, (StatusCode, Json<ErrorResponse>)> = async {
        let registry = state.model_registry.as_ref().ok_or_else(|| {
            api_err(
                StatusCode::SERVICE_UNAVAILABLE,
                "No models available on this server",
                "model_unavailable",
            )
        })?;

        let backend = registry.get(&request.model_id).ok_or_else(|| {
            api_err(
                StatusCode::BAD_REQUEST,
                format!("Model '{}' not found", request.model_id),
                "model_not_found",
            )
        })?;

        let requested_bands = SamBands::from(request.bands);
        let (resolved_bands, _, _) = resolve_sam_bands(&state, &request.image_id, requested_bands)
            .await
            .map_err(map_image_error)?;
        let embedding_key = resolved_bands.embedding_key(&request.image_id);

        if backend.requires_embedding() && !backend.is_prepared(&embedding_key).await {
            let (rgb_data, width, height, _) = load_image_rgb_with_bands(
                &state,
                &request.image_id,
                resolved_bands.red,
                resolved_bands.green,
                resolved_bands.blue,
            )
            .await
            .map_err(map_image_error)?;
            let prepare_image = ImageContext {
                image_id: embedding_key,
                rgb_data,
                width,
                height,
            };
            backend.prepare(&prepare_image, None).await.map_err(|e| {
                api_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to warm embedding: {}", e),
                    "model_prepare_failed",
                )
            })?;
        }

        Ok(StatusCode::ACCEPTED)
    }
    .await;

    let elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    state
        .rest_metrics
        .record_sam_warm(elapsed_ms, outcome.is_ok());
    outcome
}

fn normalize_inputs(inputs: &SamInputsRequest) -> Result<Value, String> {
    let points_value = &inputs.points;
    let mut points = Vec::<Value>::new();

    if let Some(array) = points_value.as_array() {
        if array.is_empty() {
            // handled by validate_inputs
        } else if array.first().and_then(Value::as_array).is_some() {
            let labels = inputs
                .labels
                .as_ref()
                .ok_or_else(|| "labels are required when points are [[x,y], ...]".to_string())?;
            if labels.len() != array.len() {
                return Err("points and labels length mismatch".to_string());
            }
            for (idx, pair) in array.iter().enumerate() {
                let Some(coords) = pair.as_array() else {
                    return Err("point entries must be [x, y]".to_string());
                };
                if coords.len() != 2 {
                    return Err("point entries must contain exactly 2 values".to_string());
                }
                let x = coords[0]
                    .as_f64()
                    .ok_or_else(|| "point x must be numeric".to_string())?;
                let y = coords[1]
                    .as_f64()
                    .ok_or_else(|| "point y must be numeric".to_string())?;
                points.push(json!({
                    "x": x,
                    "y": y,
                    "label": labels[idx],
                }));
            }
        } else if array.first().and_then(Value::as_object).is_some() {
            for entry in array {
                let x = entry
                    .get("x")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| "point object missing numeric x".to_string())?;
                let y = entry
                    .get("y")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| "point object missing numeric y".to_string())?;
                let label = entry
                    .get("label")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| "point object missing integer label".to_string())?;
                points.push(json!({
                    "x": x,
                    "y": y,
                    "label": label,
                }));
            }
        } else {
            return Err("points must be [] or [[x,y], ...] or [{x,y,label}, ...]".to_string());
        }
    } else if !points_value.is_null() {
        return Err("points must be an array".to_string());
    }

    Ok(json!({
        "points": points,
        "box": inputs.box_prompt,
    }))
}

fn map_image_error(err: Error) -> (StatusCode, Json<ErrorResponse>) {
    match err {
        Error::ImageNotFound(_) => {
            api_err(StatusCode::NOT_FOUND, err.to_string(), "image_not_found")
        }
        Error::UnsupportedFormat(_) => api_err(
            StatusCode::BAD_REQUEST,
            err.to_string(),
            "unsupported_format",
        ),
        _ => api_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            err.to_string(),
            "internal_error",
        ),
    }
}

fn api_err(
    status: StatusCode,
    message: impl Into<String>,
    code: &'static str,
) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: message.into(),
            code,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use async_trait::async_trait;
    use axum::{
        body::Body,
        http::{Method, Request},
    };
    use serde_json::json;
    use tower::ServiceExt as _;

    use crate::full_sam::config::ServerConfig;
    use crate::full_sam::sam::{
        EncoderOutput, ExecutionProvider, SamBackend, SamMaskResult, SamPoint,
    };
    use crate::full_sam::state::{AppState, SamBackendWiring};

    struct MockSamBackend;

    #[async_trait]
    impl SamBackend for MockSamBackend {
        fn name(&self) -> &'static str {
            "mock-sam"
        }

        fn provider(&self) -> ExecutionProvider {
            ExecutionProvider::Cpu
        }

        async fn encode_image(
            &self,
            _image_rgb: &[u8],
            _width: u32,
            _height: u32,
        ) -> anyhow::Result<EncoderOutput> {
            Ok(EncoderOutput::BackendSpecific {
                backend_name: "mock-sam".to_string(),
                format: "mock-v1".to_string(),
                data: vec![1],
            })
        }

        async fn decode_mask(
            &self,
            _encoder_output: &EncoderOutput,
            original_width: u32,
            original_height: u32,
            points: &[SamPoint],
            box_prompt: Option<[f32; 4]>,
        ) -> anyhow::Result<SamMaskResult> {
            let (x1, y1, x2, y2) = if let Some([x1, y1, x2, y2]) = box_prompt {
                (x1, y1, x2, y2)
            } else if let Some(point) = points.iter().find(|p| p.label == 1) {
                let radius = 24.0;
                (
                    (point.x - radius).max(0.0),
                    (point.y - radius).max(0.0),
                    (point.x + radius).min(original_width as f32),
                    (point.y + radius).min(original_height as f32),
                )
            } else {
                (0.0, 0.0, 32.0, 32.0)
            };

            Ok(SamMaskResult {
                masks: vec![vec![0u8; (original_width * original_height) as usize]],
                polygons: vec![vec![[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
                iou_scores: vec![0.93],
                width: original_width,
                height: original_height,
            })
        }

        fn is_available(&self) -> bool {
            true
        }
    }

    fn mock_model_name(_: &ServerConfig) -> String {
        "Mock SAM".to_string()
    }

    fn mock_expected_files(_: &ServerConfig) -> Vec<String> {
        vec!["mock".to_string()]
    }

    fn mock_factory(_: &ServerConfig) -> anyhow::Result<Arc<dyn SamBackend>> {
        Ok(Arc::new(MockSamBackend))
    }

    async fn make_router() -> (Router, String, tempfile::TempDir) {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let image_path = temp_dir.path().join("sample.png");
        let mut img = image::RgbImage::new(128, 96);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([128, 200, 64]);
        }
        img.save(&image_path).expect("save sample image");

        let config = ServerConfig {
            sam_enabled: true,
            ..ServerConfig::default()
        };
        let mut config = config;
        config.base.data_dir = temp_dir.path().to_path_buf();
        config.base.project_name = "sam-route-test".to_string();
        let state = AppState::new_with_sam_wiring(
            config,
            SamBackendWiring {
                backend_name: "mock-backend",
                model_id: "sam-mock",
                model_name: mock_model_name,
                expected_files: mock_expected_files,
                create_backend: mock_factory,
            },
        )
        .expect("state init");
        let first_image_id =
            crate::common::catalog::list_supported_images(&state.config.data_dir, |path| {
                state.loaders.find_loader(path).is_some()
            })
            .first()
            .map(|info| info.id.clone())
            .expect("at least one image in test data dir");

        (
            Router::new()
                .nest("/api/sam", super::router())
                .with_state(Arc::new(state)),
            first_image_id,
            temp_dir,
        )
    }

    #[tokio::test]
    async fn infer_accepts_points_and_labels_shape() {
        let (app, image_id, _temp_dir) = make_router().await;
        let request = Request::builder()
            .method(Method::POST)
            .uri("/api/sam/infer")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&json!({
                    "model_id": "sam-mock",
                    "image_id": image_id,
                    "bands": { "r": 0, "g": 1, "b": 2 },
                    "inputs": {
                        "points": [[100.0, 110.0], [140.0, 150.0]],
                        "labels": [1, 0]
                    }
                }))
                .expect("serialize"),
            ))
            .expect("request");

        let response = app.oneshot(request).await.expect("response");
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read body");
        let json: Value = serde_json::from_slice(&body).expect("json");
        assert!(json.get("masks").and_then(Value::as_array).is_some());
        assert!(json.get("scores").is_some());
    }

    #[tokio::test]
    async fn infer_rejects_mismatched_points_labels() {
        let (app, image_id, _temp_dir) = make_router().await;
        let request = Request::builder()
            .method(Method::POST)
            .uri("/api/sam/infer")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&json!({
                    "model_id": "sam-mock",
                    "image_id": image_id,
                    "inputs": {
                        "points": [[1.0, 2.0], [3.0, 4.0]],
                        "labels": [1]
                    }
                }))
                .expect("serialize"),
            ))
            .expect("request");

        let response = app.oneshot(request).await.expect("response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read body");
        let json: Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(json["code"], "invalid_inputs");
        assert!(
            json["error"]
                .as_str()
                .is_some_and(|m| m.contains("length mismatch"))
        );
    }

    #[tokio::test]
    async fn warm_endpoint_returns_accepted() {
        let (app, image_id, _temp_dir) = make_router().await;
        let request = Request::builder()
            .method(Method::POST)
            .uri("/api/sam/warm")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&json!({
                    "model_id": "sam-mock",
                    "image_id": image_id,
                    "bands": { "r": 0, "g": 1, "b": 2 }
                }))
                .expect("serialize"),
            ))
            .expect("request");

        let response = app.oneshot(request).await.expect("response");
        assert_eq!(response.status(), StatusCode::ACCEPTED);
    }
}
