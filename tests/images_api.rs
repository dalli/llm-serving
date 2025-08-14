use axum::{routing::post, Router};
use axum::http::{Request, StatusCode};
use axum::body::Body;
use tower::util::ServiceExt; // for `oneshot`
use serde_json::{json, Value};
use std::sync::Arc;

use llm_serving::{
    api::routes::{images_generations},
    engine::CoreEngine,
};

#[tokio::test]
async fn images_generations_returns_b64_list() {
    let engine = Arc::new(CoreEngine::new());
    let app = Router::new()
        .route("/v1/images/generations", post(images_generations))
        .with_state(engine);

    let payload = json!({
        "model": "dummy-image",
        "prompt": "a cute cat",
        "n": 2,
        "size": "256x256"
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/images/generations")
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&body_bytes).unwrap();
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert!(data[0]["b64_json"].as_str().is_some());
}
