use axum::{
    extract::State,
    response::{sse::Event, IntoResponse, Response, Sse},
    Json,
};
use futures::StreamExt; // Import StreamExt
use std::{convert::Infallible, sync::Arc, time::{SystemTime, UNIX_EPOCH}};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::api::{
    dto::{
        ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionRequest,
        ChatCompletionResponse, ChatCompletionChoice, ChatCompletionMessage, Delta,
        ResponseMessage, Usage, EmbeddingsRequest, EmbeddingsResponse, EmbeddingObject, EmbeddingUsage,
    },
    error::AppError,
};
use crate::engine::CoreEngine; // Import the actual CoreEngine

pub async fn chat_completions(
    State(engine): State<Arc<CoreEngine>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, AppError> {
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
    State(engine): State<Arc<CoreEngine>>,
    Json(request): Json<EmbeddingsRequest>,
 ) -> Result<Response, AppError> {
    match engine.process_embedding_request(request).await {
        Ok(resp) => Ok(Json(resp).into_response()),
        Err(e) => Err(AppError::BadRequest(e)),
    }
 }