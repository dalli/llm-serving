use std::{collections::HashMap, sync::Arc};
use tokio::sync::{mpsc, Semaphore};

use crate::{
    api::dto::{
        ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionRequest,
        ChatCompletionResponse, ChatCompletionChoice, Delta, ResponseMessage, Usage,
        EmbeddingsRequest, EmbeddingsResponse, EmbeddingObject, EmbeddingUsage,
    },
    runtime::{dummy::DummyRuntime, dummy_embedding::DummyEmbeddingRuntime, LlmRuntime, EmbeddingRuntime},
};
#[cfg(feature = "llama")]
use crate::runtime::llama_cpp::LlamaCppRuntime;

pub struct CoreEngine {
    llm_runtimes: HashMap<String, Arc<dyn LlmRuntime>>,
    embedding_runtimes: HashMap<String, Arc<dyn EmbeddingRuntime>>,
    request_sender: mpsc::Sender<EngineRequest>,
}

pub enum EngineRequest {
    ChatCompletion {
        request: ChatCompletionRequest,
        response_sender: Option<mpsc::Sender<Result<ChatCompletionResponse, String>>>,
        stream_sender: Option<mpsc::Sender<String>>,
    },
    Embeddings {
        request: EmbeddingsRequest,
        response_sender: mpsc::Sender<Result<EmbeddingsResponse, String>>,
    },
}

impl CoreEngine {
    pub fn new() -> Self {
        let (request_sender, request_receiver) = mpsc::channel(100); // Channel for incoming requests

        let mut llm_runtimes: HashMap<String, Arc<dyn LlmRuntime>> = HashMap::new();
        // Always have a fallback dummy runtime for development
        llm_runtimes.insert("dummy-model".to_string(), Arc::new(DummyRuntime::new()));
        // Attempt to load a real llama.cpp runtime if a valid path is provided via env
        #[cfg(feature = "llama")]
        {
            if let Ok(model_path) = std::env::var("LLAMA_MODEL_PATH") {
                if let Ok(llama_runtime) = LlamaCppRuntime::new(&model_path) {
                    llm_runtimes.insert("llama-cpp".to_string(), Arc::new(llama_runtime));
                } else {
                    eprintln!("Failed to load LlamaCppRuntime from LLAMA_MODEL_PATH; continuing with dummy-model.");
                }
            }
        }

        // Embedding runtimes
        let mut embedding_runtimes: HashMap<String, Arc<dyn EmbeddingRuntime>> = HashMap::new();
        embedding_runtimes.insert("dummy-embedding".to_string(), Arc::new(DummyEmbeddingRuntime::new(384)));

        // Clone runtimes for the worker pool and wrap in Arc for shared access
        let worker_llm = Arc::new(llm_runtimes.clone());
        let worker_embed = Arc::new(embedding_runtimes.clone());

        // Configure concurrency limit (ENV: ENGINE_WORKERS), default to available_parallelism or 4
        let workers: usize = std::env::var("ENGINE_WORKERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(4);
        let semaphore = Arc::new(Semaphore::new(workers));

        tokio::spawn(Self::worker_pool(worker_llm, worker_embed, request_receiver, semaphore));

        CoreEngine {
            llm_runtimes,
            embedding_runtimes,
            request_sender,
        }
    }

    async fn worker_pool(
        llm_runtimes: Arc<HashMap<String, Arc<dyn LlmRuntime>>>,
        embedding_runtimes: Arc<HashMap<String, Arc<dyn EmbeddingRuntime>>>,
        mut request_receiver: mpsc::Receiver<EngineRequest>,
        semaphore: Arc<Semaphore>,
    ) {
        while let Some(req) = request_receiver.recv().await {
            let llm_map = llm_runtimes.clone();
            let embed_map = embedding_runtimes.clone();
            let semaphore_clone = semaphore.clone();
            // Acquire a permit and process the request concurrently
            tokio::spawn(async move {
                let _permit = semaphore_clone.acquire_owned().await.expect("semaphore closed");
                match req {
                    EngineRequest::ChatCompletion { request, response_sender, stream_sender } => {
                        let model_name = request.model.clone();
                        if let Some(runtime) = llm_map.get(&model_name) {
                            let prompt = request.messages.last().map(|m| m.content.clone()).unwrap_or_default();
                            let max_tokens = 100; // TODO: make configurable

                            if let Some(stream_tx) = stream_sender {
                                // Stream role first
                                let id = uuid::Uuid::new_v4().to_string();
                                let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
                                let role_chunk = ChatCompletionChunk {
                                    id: id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model_name.clone(),
                                    choices: vec![ChatCompletionChunkChoice {
                                        index: 0,
                                        delta: Delta { role: Some("assistant".to_string()), content: None },
                                        finish_reason: None,
                                    }],
                                };
                                let _ = stream_tx.send(serde_json::to_string(&role_chunk).unwrap()).await;

                                // Generate full text (simple runtime API), then send in one content chunk
                                let generated = runtime.generate(&prompt, max_tokens).await.unwrap_or_else(|e| format!("[error: {}]", e));
                                let content_chunk = ChatCompletionChunk {
                                    id: id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model_name.clone(),
                                    choices: vec![ChatCompletionChunkChoice {
                                        index: 0,
                                        delta: Delta { role: None, content: Some(generated) },
                                        finish_reason: None,
                                    }],
                                };
                                let _ = stream_tx.send(serde_json::to_string(&content_chunk).unwrap()).await;

                                let done_chunk = ChatCompletionChunk {
                                    id: id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model_name.clone(),
                                    choices: vec![ChatCompletionChunkChoice {
                                        index: 0,
                                        delta: Delta { role: None, content: None },
                                        finish_reason: Some("stop".to_string()),
                                    }],
                                };
                                let _ = stream_tx.send(serde_json::to_string(&done_chunk).unwrap()).await;
                                // Optional: client often expects a [DONE] sentinel per OpenAI semantics
                                let _ = stream_tx.send("[DONE]".to_string()).await;
                            } else if let Some(resp_tx) = response_sender {
                                let generated = runtime.generate(&prompt, max_tokens).await.unwrap_or_default();
                                let response = ChatCompletionResponse {
                                    id: uuid::Uuid::new_v4().to_string(),
                                    object: "chat.completion".to_string(),
                                    created: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                                    model: model_name,
                                    choices: vec![ChatCompletionChoice {
                                        index: 0,
                                        message: ResponseMessage { role: "assistant".to_string(), content: generated.clone() },
                                        finish_reason: "stop".to_string(),
                                    }],
                                    usage: Usage {
                                        prompt_tokens: 0,
                                        completion_tokens: 0,
                                        total_tokens: 0,
                                    },
                                };
                                let _ = resp_tx.send(Ok(response)).await;
                            }
                        } else if let Some(resp_tx) = response_sender {
                            let _ = resp_tx.send(Err(format!("Model {} not found", model_name))).await;
                        }
                    }
                    EngineRequest::Embeddings { request, response_sender } => {
                        let model_name = request.model.clone();
                        if let Some(runtime) = embed_map.get(&model_name) {
                            let inputs = request.input.clone();
                            let result = runtime.embed(&inputs).await;
                            match result {
                                Ok(vectors) => {
                                    let data: Vec<EmbeddingObject> = vectors
                                        .into_iter()
                                        .enumerate()
                                        .map(|(i, v)| EmbeddingObject { object: "embedding".to_string(), index: i, embedding: v })
                                        .collect();
                                    let response = EmbeddingsResponse {
                                        data,
                                        model: model_name,
                                        object: "list".to_string(),
                                        usage: EmbeddingUsage { prompt_tokens: 0, total_tokens: 0 },
                                    };
                                    let _ = response_sender.send(Ok(response)).await;
                                }
                                Err(e) => { let _ = response_sender.send(Err(e)).await; }
                            }
                        } else {
                            let _ = response_sender.send(Err(format!("Model {} not found", model_name))).await;
                        }
                    }
                }
                // _permit dropped here, releasing capacity
            });
        }
    }

    pub async fn process_chat_request(
        &self,
        request: ChatCompletionRequest,
        stream_sender: Option<mpsc::Sender<String>>,
    ) -> Result<ChatCompletionResponse, String> {
        let (response_sender, mut response_receiver) = mpsc::channel(1);
        self.request_sender
            .send(EngineRequest::ChatCompletion {
                request,
                response_sender: if stream_sender.is_none() { Some(response_sender) } else { None },
                stream_sender: stream_sender.clone(), // Clone stream_sender
            })
            .await
            .map_err(|e| format!("Failed to send request to engine: {}", e))?;
        
        if stream_sender.is_none() {
            response_receiver
                .recv()
                .await
                .ok_or("Engine response channel closed".to_string())?
        } else {
            // For streaming, we don't return a ChatCompletionResponse directly
            // The response is sent via the stream_sender
            Err("Streaming response handled via sender".to_string())
        }
    }

    pub async fn process_embedding_request(
        &self,
        request: EmbeddingsRequest,
    ) -> Result<EmbeddingsResponse, String> {
        let (response_sender, mut response_receiver) = mpsc::channel(1);
        self.request_sender
            .send(EngineRequest::Embeddings { request, response_sender })
            .await
            .map_err(|e| format!("Failed to send request to engine: {}", e))?;

        response_receiver
            .recv()
            .await
            .ok_or("Engine response channel closed".to_string())?
    }
}