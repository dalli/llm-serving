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
