use std::{collections::HashMap, sync::Arc};
use tokio::sync::{mpsc, Semaphore, RwLock};
use moka::future::Cache;
use sha2::{Digest, Sha256};
use metrics::{counter, histogram};

use crate::{
    api::dto::{
        ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionRequest,
        ChatCompletionResponse, ChatCompletionChoice, Delta, ResponseMessage, Usage, ChatMessageContent, ContentPart,
        EmbeddingsRequest, EmbeddingsResponse, EmbeddingObject, EmbeddingUsage,
        ImagesGenerationRequest,
    },
    runtime::{dummy::DummyRuntime, dummy_embedding::DummyEmbeddingRuntime, LlmRuntime, EmbeddingRuntime, MultimodalRuntime, ImageGenRuntime, GenerationOptions},
};
#[cfg(feature = "llama")]
use crate::runtime::llama_cpp::LlamaCppRuntime;
#[cfg(feature = "onnx")]
use crate::runtime::onnx_embedding::OnnxEmbeddingRuntime;
#[cfg(feature = "llava")]
use crate::runtime::llava::LlavaRuntime;

pub struct CoreEngine {
    llm_runtimes: Arc<RwLock<HashMap<String, Arc<dyn LlmRuntime>>>>,
    embedding_runtimes: Arc<RwLock<HashMap<String, Arc<dyn EmbeddingRuntime>>>>,
    multimodal_runtimes: Arc<RwLock<HashMap<String, Arc<dyn MultimodalRuntime>>>>,
    image_runtimes: Arc<RwLock<HashMap<String, Arc<dyn ImageGenRuntime>>>>,
    request_sender: mpsc::Sender<EngineRequest>,
    response_cache: Cache<String, ChatCompletionResponse>,
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
    Images {
        request: ImagesGenerationRequest,
        response_sender: mpsc::Sender<Result<Vec<Vec<u8>>, String>>,
    },
}

impl CoreEngine {
    pub fn new() -> Self {
        let (request_sender, request_receiver) = mpsc::channel(100); // Channel for incoming requests

        let mut llm_map_init: HashMap<String, Arc<dyn LlmRuntime>> = HashMap::new();
        // Always have a fallback dummy runtime for development
        llm_map_init.insert("dummy-model".to_string(), Arc::new(DummyRuntime::new()));
        // Attempt to load a real llama.cpp runtime if a valid path is provided via env
        #[cfg(feature = "llama")]
        {
            if let Ok(model_path) = std::env::var("LLAMA_MODEL_PATH") {
                if let Ok(llama_runtime) = LlamaCppRuntime::new(&model_path) {
                    llm_map_init.insert("llama-cpp".to_string(), Arc::new(llama_runtime));
                } else {
                    eprintln!("Failed to load LlamaCppRuntime from LLAMA_MODEL_PATH; continuing with dummy-model.");
                }
            }
        }
        // Build multimodal runtimes map alongside LLM map
        let mut mm_map_init: HashMap<String, Arc<dyn MultimodalRuntime>> = HashMap::new();
        // Ensure dummy exists in both maps
        if !llm_map_init.contains_key("dummy-model") {
            llm_map_init.insert("dummy-model".to_string(), Arc::new(DummyRuntime::new()));
        }
        mm_map_init.insert("dummy-model".to_string(), Arc::new(DummyRuntime::new()));

        let llm_runtimes: Arc<RwLock<HashMap<String, Arc<dyn LlmRuntime>>>> = Arc::new(RwLock::new(llm_map_init));

        // Embedding runtimes
        let mut embed_map_init: HashMap<String, Arc<dyn EmbeddingRuntime>> = HashMap::new();
        embed_map_init.insert("dummy-embedding".to_string(), Arc::new(DummyEmbeddingRuntime::new(384)));
        #[cfg(feature = "onnx")]
        if let Ok(onnx_model) = std::env::var("ONNX_EMBEDDING_MODEL_PATH") {
            // Dimension should ideally be inferred; keep 384 default
            if let Ok(rt) = OnnxEmbeddingRuntime::new(&onnx_model, 384) {
                embed_map_init.insert("onnx-embedding".to_string(), Arc::new(rt));
            }
        }
        let embedding_runtimes: Arc<RwLock<HashMap<String, Arc<dyn EmbeddingRuntime>>>> = Arc::new(RwLock::new(embed_map_init));
        // Image runtimes (Phase 4 scaffold)
        let mut img_map_init: HashMap<String, Arc<dyn ImageGenRuntime>> = HashMap::new();
        img_map_init.insert("dummy-image".to_string(), Arc::new(crate::runtime::dummy_image::DummyImageRuntime::new()));
        let image_runtimes: Arc<RwLock<HashMap<String, Arc<dyn ImageGenRuntime>>>> = Arc::new(RwLock::new(img_map_init));
        #[cfg(feature = "llava")]
        {
            if let (Ok(vision), Ok(proj), Ok(llm)) = (
                std::env::var("LLAVA_VISION_MODEL_PATH"),
                std::env::var("LLAVA_PROJECTION_PATH"),
                std::env::var("LLAMA_MODEL_PATH"),
            ) {
                if let Ok(rt) = LlavaRuntime::new(&vision, &proj, &llm) {
                    mm_map_init.insert("llava".to_string(), Arc::new(rt));
                }
            }
        }
        let multimodal_runtimes: Arc<RwLock<HashMap<String, Arc<dyn MultimodalRuntime>>>> = Arc::new(RwLock::new(mm_map_init));

        // Clone runtimes for the worker pool and wrap in Arc for shared access
        let worker_llm = llm_runtimes.clone();
        let worker_embed = embedding_runtimes.clone();
        let worker_mm = multimodal_runtimes.clone();
        let worker_img = image_runtimes.clone();

        // Configure concurrency limit (ENV: ENGINE_WORKERS), default to available_parallelism or 4
        let workers: usize = std::env::var("ENGINE_WORKERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(4);
        let semaphore = Arc::new(Semaphore::new(workers));

        tokio::spawn(Self::worker_pool(worker_llm, worker_embed, worker_mm, worker_img, request_receiver, semaphore));

        CoreEngine {
            llm_runtimes,
            embedding_runtimes,
            multimodal_runtimes,
            image_runtimes,
            request_sender,
            response_cache: Cache::builder()
                .max_capacity(10_000)
                .time_to_live(std::time::Duration::from_secs(60))
                .build(),
        }
    }

    async fn worker_pool(
        llm_runtimes: Arc<RwLock<HashMap<String, Arc<dyn LlmRuntime>>>>,
        embedding_runtimes: Arc<RwLock<HashMap<String, Arc<dyn EmbeddingRuntime>>>>,
        multimodal_runtimes: Arc<RwLock<HashMap<String, Arc<dyn MultimodalRuntime>>>>,
        image_runtimes: Arc<RwLock<HashMap<String, Arc<dyn ImageGenRuntime>>>>,
        mut request_receiver: mpsc::Receiver<EngineRequest>,
        semaphore: Arc<Semaphore>,
    ) {
        while let Some(req) = request_receiver.recv().await {
            let llm_map = llm_runtimes.clone();
            let embed_map = embedding_runtimes.clone();
            let mm_map = multimodal_runtimes.clone();
            let img_map = image_runtimes.clone();
            let semaphore_clone = semaphore.clone();
            // Acquire a permit and process the request concurrently
            tokio::spawn(async move {
                let _permit = semaphore_clone.acquire_owned().await.expect("semaphore closed");
                match req {
                    EngineRequest::ChatCompletion { request, response_sender, stream_sender } => {
                        counter!("requests_total", 1, "endpoint" => "chat");
                        let model_name = request.model.clone();
                        // Lookup both runtimes (LLM and Multimodal) for the given model name
                        let (llm_runtime_opt, mm_runtime_opt) = {
                            let llm = llm_map.read().await;
                            let mm = mm_map.read().await;
                            (llm.get(&model_name).cloned(), mm.get(&model_name).cloned())
                        };
                        if llm_runtime_opt.is_some() || mm_runtime_opt.is_some() {
                            let (prompt, image_urls) = match request.messages.last().map(|m| m.content.clone()) {
                                Some(ChatMessageContent::Text(content)) => (content, Vec::new()),
                                Some(ChatMessageContent::Parts(parts)) => {
                                    let mut text_acc = String::new();
                                    let mut urls = Vec::new();
                                    for p in parts {
                                        match p {
                                            ContentPart::Text { text } => text_acc.push_str(&text),
                                            ContentPart::ImageUrl { image_url } => urls.push(image_url.url),
                                        }
                                    }
                                    (text_acc, urls)
                                }
                                None => (String::new(), Vec::new()),
                            };
                            let gen_opts = GenerationOptions::from_request(request.max_tokens, request.temperature, request.top_p);

                            if let Some(stream_tx) = stream_sender {
                                let start = std::time::Instant::now();
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
                                let generated = if image_urls.is_empty() {
                                    if let Some(ref llm_rt) = llm_runtime_opt {
                                        llm_rt.generate(&prompt, &gen_opts).await
                                    } else {
                                        Err("Model requires images".to_string())
                                    }
                                } else {
                                    if let Some(ref mm_rt) = mm_runtime_opt {
                                        mm_rt.generate_from_vision(&prompt, &image_urls, &gen_opts).await
                                    } else if let Some(ref llm_rt) = llm_runtime_opt {
                                        // Fallback: ignore images if only LLM exists for compatibility
                                        llm_rt.generate(&prompt, &gen_opts).await
                                    } else {
                                        Err("Model not available".to_string())
                                    }
                                }.unwrap_or_else(|e| format!("[error: {}]", e));
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
                                histogram!(
                                    "request_latency_ms",
                                    start.elapsed().as_millis() as f64,
                                    "endpoint" => "chat"
                                );
                            } else if let Some(resp_tx) = response_sender {
                                let start = std::time::Instant::now();
                                let generated = if image_urls.is_empty() {
                                    if let Some(ref llm_rt) = llm_runtime_opt {
                                        llm_rt.generate(&prompt, &gen_opts).await.unwrap_or_default()
                                    } else {
                                        String::from("[error: Model requires images]")
                                    }
                                } else {
                                    if let Some(ref mm_rt) = mm_runtime_opt {
                                        mm_rt.generate_from_vision(&prompt, &image_urls, &gen_opts).await.unwrap_or_default()
                                    } else if let Some(ref llm_rt) = llm_runtime_opt {
                                        llm_rt.generate(&prompt, &gen_opts).await.unwrap_or_default()
                                    } else {
                                        String::from("[error: Model not available]")
                                    }
                                };
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
                                histogram!(
                                    "request_latency_ms",
                                    start.elapsed().as_millis() as f64,
                                    "endpoint" => "chat"
                                );
                            }
                        } else if let Some(resp_tx) = response_sender {
                            let _ = resp_tx.send(Err(format!("Model {} not found", model_name))).await;
                        }
                    }
                    EngineRequest::Embeddings { request, response_sender } => {
                        counter!("requests_total", 1, "endpoint" => "embeddings");
                        let model_name = request.model.clone();
                        let runtime_opt = {
                            let map = embed_map.read().await;
                            map.get(&model_name).cloned()
                        };
                        if let Some(runtime) = runtime_opt {
                            let start = std::time::Instant::now();
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
                                histogram!(
                                    "request_latency_ms",
                                    start.elapsed().as_millis() as f64,
                                    "endpoint" => "embeddings"
                                );
                                }
                                Err(e) => { let _ = response_sender.send(Err(e)).await; }
                            }
                        } else {
                            let _ = response_sender.send(Err(format!("Model {} not found", model_name))).await;
                        }
                    }
                    EngineRequest::Images { request, response_sender } => {
                        counter!("requests_total", 1, "endpoint" => "images");
                        let model_name = request.model.clone();
                        let runtime_opt = {
                            let map = img_map.read().await;
                            map.get(&model_name).cloned()
                        };
                        if let Some(runtime) = runtime_opt {
                            let start = std::time::Instant::now();
                            let n = request.n;
                            let prompt = request.prompt.clone();
                            let size = request.size.clone();
                            let result = runtime.generate_images(&prompt, n, &size).await;
                            let _ = response_sender.send(result).await;
                            histogram!(
                                "request_latency_ms",
                                start.elapsed().as_millis() as f64,
                                "endpoint" => "images"
                            );
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
        // Cache only non-streaming responses
        let cache_key = if stream_sender.is_none() {
            Some(Self::hash_chat_request(&request))
        } else {
            None
        };

        if let Some(ref key) = cache_key {
            if let Some(resp) = self.response_cache.get(key).await {
                counter!("cache_hit_total", 1);
                return Ok(resp);
            }
            counter!("cache_miss_total", 1);
        }

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
            let result = response_receiver
                .recv()
                .await
                .ok_or("Engine response channel closed".to_string())?;
            if let (Some(key), Ok(resp)) = (cache_key, &result) {
                self.response_cache.insert(key, resp.clone()).await;
                counter!("cache_store_total", 1);
            }
            result
        } else {
            // For streaming, we don't return a ChatCompletionResponse directly
            // The response is sent via the stream_sender
            Err("Streaming response handled via sender".to_string())
        }
    }

    fn hash_chat_request(req: &ChatCompletionRequest) -> String {
        let mut hasher = Sha256::new();
        hasher.update(req.model.as_bytes());
        for m in &req.messages {
            hasher.update(m.role.as_bytes());
            match &m.content {
                ChatMessageContent::Text(content) => hasher.update(content.as_bytes()),
                ChatMessageContent::Parts(parts) => {
                    for p in parts {
                        match p {
                            ContentPart::Text { text } => hasher.update(text.as_bytes()),
                            ContentPart::ImageUrl { image_url } => hasher.update(image_url.url.as_bytes()),
                        }
                    }
                }
            }
        }
        if let Some(mt) = req.max_tokens { hasher.update(mt.to_le_bytes()); }
        if let Some(t) = req.temperature { hasher.update(t.to_le_bytes()); }
        if let Some(tp) = req.top_p { hasher.update(tp.to_le_bytes()); }
        format!("{:x}", hasher.finalize())
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

    pub async fn process_image_request(
        &self,
        request: ImagesGenerationRequest,
    ) -> Result<Vec<Vec<u8>>, String> {
        let (response_sender, mut response_receiver) = mpsc::channel(1);
        self.request_sender
            .send(EngineRequest::Images { request, response_sender })
            .await
            .map_err(|e| format!("Failed to send request to engine: {}", e))?;

        response_receiver
            .recv()
            .await
            .ok_or("Engine response channel closed".to_string())?
    }

    // Admin helpers (simple; no persistence)
    pub async fn list_models(&self) -> (Vec<String>, Vec<String>, Vec<String>, Vec<String>) {
        let llm = { self.llm_runtimes.read().await.keys().cloned().collect::<Vec<_>>() };
        let embedding = { self.embedding_runtimes.read().await.keys().cloned().collect::<Vec<_>>() };
        let multimodal = { self.multimodal_runtimes.read().await.keys().cloned().collect::<Vec<_>>() };
        let image = { self.image_runtimes.read().await.keys().cloned().collect::<Vec<_>>() };
        (llm, embedding, multimodal, image)
    }

    pub async fn load_model(&self, kind: &str, name: &str, path: Option<&str>) -> Result<(), String> {
        match kind {
            "llm" => {
                #[cfg(feature = "llama")]
                if let Some(p) = path {
                    let rt = LlamaCppRuntime::new(p).map_err(|e| format!("load llama: {}", e))?;
                    self.llm_runtimes.write().await.insert(name.to_string(), Arc::new(rt));
                    return Ok(());
                }
                // fallback: dummy
                self.llm_runtimes.write().await.insert(name.to_string(), Arc::new(DummyRuntime::new()));
                Ok(())
            }
            "embedding" => {
                #[cfg(feature = "onnx")]
                if let Some(p) = path {
                    if let Ok(rt) = OnnxEmbeddingRuntime::new(p, 384) {
                        self.embedding_runtimes.write().await.insert(name.to_string(), Arc::new(rt));
                        return Ok(());
                    }
                }
                // fallback: dummy
                self.embedding_runtimes.write().await.insert(name.to_string(), Arc::new(DummyEmbeddingRuntime::new(384)));
                Ok(())
            }
            "multimodal" => {
                #[cfg(feature = "llava")]
                {
                    // If explicit path triple provided as comma-separated, parse; else try env
                    if let Some(p) = path {
                        let parts: Vec<&str> = p.split(',').collect();
                        if parts.len() == 3 {
                            let rt = crate::runtime::llava::LlavaRuntime::new(parts[0], parts[1], parts[2])
                                .map_err(|e| format!("load llava: {}", e))?;
                            self.multimodal_runtimes.write().await.insert(name.to_string(), Arc::new(rt));
                            return Ok(());
                        }
                    }
                    if let (Ok(vision), Ok(proj), Ok(llm)) = (
                        std::env::var("LLAVA_VISION_MODEL_PATH"),
                        std::env::var("LLAVA_PROJECTION_PATH"),
                        std::env::var("LLAMA_MODEL_PATH"),
                    ) {
                        let rt = crate::runtime::llava::LlavaRuntime::new(&vision, &proj, &llm)
                            .map_err(|e| format!("load llava: {}", e))?;
                        self.multimodal_runtimes.write().await.insert(name.to_string(), Arc::new(rt));
                        return Ok(());
                    }
                }
                // fallback: dummy runtime also implements MultimodalRuntime
                self.multimodal_runtimes.write().await.insert(name.to_string(), Arc::new(DummyRuntime::new()));
                Ok(())
            }
            _ => Err("unknown kind".to_string()),
        }
    }

    pub async fn unload_model(&self, kind: &str, name: &str) -> Result<(), String> {
        match kind {
            "llm" => { self.llm_runtimes.write().await.remove(name); Ok(()) }
            "embedding" => { self.embedding_runtimes.write().await.remove(name); Ok(()) }
            "multimodal" => { self.multimodal_runtimes.write().await.remove(name); Ok(()) }
            _ => Err("unknown kind".to_string()),
        }
    }
}