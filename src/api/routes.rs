use axum::{
    extract::State,
    response::{sse::Event, IntoResponse, Response, Sse},
    Json,
};
use futures::StreamExt;
use std::{convert::Infallible, sync::Arc};
use tokio::sync::mpsc;

use crate::api::{
    dto::{
        ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionRequest,
        EmbeddingsRequest, LoadModelRequest, UnloadModelRequest, ModelsListResponse,
        ImagesGenerationRequest, ImagesGenerationResponse, ImageDataObject,
    },
    error::AppError,
};
use crate::engine::CoreEngine; // Import the actual CoreEngine
use crate::api::auth::authorize_request;
use axum::http::HeaderMap;
use base64::Engine as _; // bring encode into scope

pub async fn chat_completions(
    headers: HeaderMap,
    State(engine): State<Arc<CoreEngine>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, AppError> {
    authorize_request(&headers).map_err(AppError::BadRequest)?;
    if request.stream.unwrap_or(false) {
        let (tx, rx) = mpsc::channel::<String>(100);

        // Use the actual CoreEngine's process_chat_request
        // Pass the sender to the engine for streaming
        let _ = engine.process_chat_request(request, Some(tx)).await;

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx).map(|data| {
            Ok::<_, Infallible>(Event::default().data(data)) // Wrap in Ok
        });

        Ok(Sse::new(stream).into_response())
    } else {
        // Use the actual CoreEngine's process_chat_request
        let response = engine.process_chat_request(request, None).await?;
        Ok(Json(response).into_response())
    }
}

pub async fn embeddings(
    headers: HeaderMap,
    State(engine): State<Arc<CoreEngine>>,
    Json(request): Json<EmbeddingsRequest>,
 ) -> Result<Response, AppError> {
    authorize_request(&headers).map_err(AppError::BadRequest)?;
    match engine.process_embedding_request(request).await {
        Ok(resp) => Ok(Json(resp).into_response()),
        Err(e) => Err(AppError::BadRequest(e)),
    }
 }

pub async fn images_generations(
    headers: HeaderMap,
    State(engine): State<Arc<CoreEngine>>,
    Json(request): Json<ImagesGenerationRequest>,
) -> Result<Response, AppError> {
    authorize_request(&headers).map_err(AppError::BadRequest)?;
    match engine.process_image_request(request).await {
        Ok(images) => {
            let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
            let data: Vec<ImageDataObject> = images.into_iter()
                .map(|bytes| ImageDataObject {
                    b64_json: Some(base64::engine::general_purpose::STANDARD.encode(bytes)),
                    url: None,
                    revised_prompt: None,
                })
                .collect();
            Ok(Json(ImagesGenerationResponse { created, data }).into_response())
        }
        Err(e) => Err(AppError::BadRequest(e)),
    }
}

pub async fn admin_models_list(
    headers: HeaderMap,
    State(engine): State<Arc<CoreEngine>>,
) -> Result<Response, AppError> {
    authorize_request(&headers).map_err(AppError::BadRequest)?;
    let (llm, embedding, multimodal) = engine.list_models().await;
    Ok(Json(ModelsListResponse { llm, embedding, multimodal }).into_response())
}

pub async fn admin_models_load(
    headers: HeaderMap,
    State(engine): State<Arc<CoreEngine>>,
    Json(req): Json<LoadModelRequest>,
) -> Result<Response, AppError> {
    authorize_request(&headers).map_err(AppError::BadRequest)?;
    engine.load_model(&req.kind, &req.model, req.path.as_deref()).await
        .map_err(AppError::BadRequest)?;
    Ok(Json(serde_json::json!({"status":"ok"})).into_response())
}

pub async fn admin_models_unload(
    headers: HeaderMap,
    State(engine): State<Arc<CoreEngine>>,
    Json(req): Json<UnloadModelRequest>,
) -> Result<Response, AppError> {
    authorize_request(&headers).map_err(AppError::BadRequest)?;
    engine.unload_model(&req.kind, &req.model).await
        .map_err(AppError::BadRequest)?;
    Ok(Json(serde_json::json!({"status":"ok"})).into_response())
}