use axum::{routing::post, Router};
use axum::http::{Request, StatusCode};
use axum::body::Body;
use tower::util::ServiceExt; // for `oneshot`
use serde_json::{json, Value};
use std::sync::Arc;

use llm_serving::{
    api::routes::{chat_completions, embeddings},
    engine::CoreEngine,
};

#[tokio::test]
async fn chat_completions_non_stream_returns_json() {
    let engine = Arc::new(CoreEngine::new());
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/admin/models", axum::routing::get(llm_serving::api::routes::admin_models_list))
        .route("/admin/models/load", post(llm_serving::api::routes::admin_models_load))
        .route("/admin/models/unload", post(llm_serving::api::routes::admin_models_unload))
        .with_state(engine);

    let payload = json!({
        "model": "dummy-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false,
        "max_tokens": 3
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&body_bytes).unwrap();

    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], "dummy-model");
    assert!(v["id"].as_str().is_some());
    let content = &v["choices"][0]["message"]["content"];
    let s = content.as_str().unwrap_or("");
    assert!(s.starts_with("Echo:"));
    // Should be short due to max_tokens
    assert!(s.len() <= "Echo: ".len() + 3);
}

#[tokio::test]
async fn chat_completions_stream_sends_sse_and_done() {
    let engine = Arc::new(CoreEngine::new());
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .with_state(engine);

    // Embeddings API smoke test
    let payload = json!({
        "model": "dummy-embedding",
        "input": ["hello", "world"],
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(v["model"], "dummy-embedding");
    assert!(v["data"].as_array().unwrap().len() == 2);

    let payload = json!({
        "model": "dummy-model",
        "messages": [{"role": "user", "content": "stream please"}],
        "stream": true
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response.headers().get("content-type").unwrap().to_str().unwrap();
    assert!(content_type.starts_with("text/event-stream"));

    // Collect the full SSE stream (finite in our implementation)
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body_text = String::from_utf8(body_bytes.to_vec()).unwrap();

    assert!(body_text.contains("chat.completion.chunk"));
    assert!(body_text.contains("[DONE]"));
}

#[tokio::test]
async fn chat_completions_with_image_url_routes_to_multimodal() {
    let engine = Arc::new(CoreEngine::new());
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(engine);

    let payload = json!({
        "model": "dummy-model",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
            ]
        }],
        "stream": false,
        "max_tokens": 50
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&body_bytes).unwrap();

    let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
    assert!(content.starts_with("Echo(Vision): "));
    assert!(content.contains("images=1"));
}

#[tokio::test]
async fn admin_can_load_and_unload_embedding_model() {
    let engine = Arc::new(CoreEngine::new());
    let app = Router::new()
        .route("/admin/models", axum::routing::get(llm_serving::api::routes::admin_models_list))
        .route("/admin/models/load", post(llm_serving::api::routes::admin_models_load))
        .route("/admin/models/unload", post(llm_serving::api::routes::admin_models_unload))
        .with_state(engine);

    // initial list
    let req = Request::builder().method("GET").uri("/admin/models").body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["embedding"].as_array().unwrap().iter().any(|m| m.as_str() == Some("dummy-embedding")));

    // load new embedding model (falls back to dummy if feature not enabled/path missing)
    let payload = json!({"model": "custom-embed", "kind": "embedding", "path": null});
    let req = Request::builder()
        .method("POST")
        .uri("/admin/models/load")
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // list should include custom-embed
    let req = Request::builder().method("GET").uri("/admin/models").body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["embedding"].as_array().unwrap().iter().any(|m| m.as_str() == Some("custom-embed")));

    // unload
    let payload = json!({"model": "custom-embed", "kind": "embedding"});
    let req = Request::builder()
        .method("POST")
        .uri("/admin/models/unload")
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // list should no longer include custom-embed
    let req = Request::builder().method("GET").uri("/admin/models").body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(!v["embedding"].as_array().unwrap().iter().any(|m| m.as_str() == Some("custom-embed")));
}

#[tokio::test]
async fn admin_can_load_and_unload_llm_model() {
    let engine = Arc::new(CoreEngine::new());
    let app = Router::new()
        .route("/admin/models", axum::routing::get(llm_serving::api::routes::admin_models_list))
        .route("/admin/models/load", post(llm_serving::api::routes::admin_models_load))
        .route("/admin/models/unload", post(llm_serving::api::routes::admin_models_unload))
        .with_state(engine);

    // load new llm model (falls back to dummy if llama feature not enabled/path missing)
    let payload = json!({"model": "custom-llm", "kind": "llm", "path": null});
    let req = Request::builder()
        .method("POST")
        .uri("/admin/models/load")
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // list should include custom-llm
    let req = Request::builder().method("GET").uri("/admin/models").body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["llm"].as_array().unwrap().iter().any(|m| m.as_str() == Some("custom-llm")));

    // unload
    let payload = json!({"model": "custom-llm", "kind": "llm"});
    let req = Request::builder()
        .method("POST")
        .uri("/admin/models/unload")
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}
