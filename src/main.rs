mod api;
mod engine;
mod runtime;

use axum::{routing::post, Router};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::engine::CoreEngine;

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "llm_serving=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let engine = Arc::new(CoreEngine::new());

    let app = Router::new()
        .route("/v1/chat/completions", post(api::routes::chat_completions))
        .route("/v1/embeddings", post(api::routes::embeddings))
        .route("/admin/models", axum::routing::get(api::routes::admin_models_list))
        .route("/admin/models/load", post(api::routes::admin_models_load))
        .route("/admin/models/unload", post(api::routes::admin_models_unload))
        .with_state(engine);

    let listener = TcpListener::bind("0.0.0.0:3000").await.unwrap();
    tracing::debug!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}