use serde::{Deserialize, Serialize};

// ---- Chat API ----
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    pub stream: Option<bool>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
}

// OpenAI-compatible Chat content: either string or array of parts
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum ChatMessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")] 
    pub detail: Option<String>, // auto|low|high
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: ChatMessageContent,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ---- Embeddings API ----
#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub object: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingObject {
    pub object: String,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

// ---- Images Generation API ----
#[derive(Debug, Deserialize)]
pub struct ImagesGenerationRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_n")] 
    pub n: u32,
    #[serde(default = "default_size")] 
    pub size: String, // e.g., "512x512"
    #[serde(default = "default_response_format")] 
    pub response_format: String, // "b64_json" (default)
}

fn default_n() -> u32 { 1 }
fn default_size() -> String { "512x512".to_string() }
fn default_response_format() -> String { "b64_json".to_string() }

#[derive(Debug, Serialize)]
pub struct ImagesGenerationResponse {
    pub created: u64,
    pub data: Vec<ImageDataObject>,
}

#[derive(Debug, Serialize)]
pub struct ImageDataObject {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

// ---- Admin API (Dynamic Model Management) ----
#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub model: String,
    pub kind: String, // "llm" | "embedding"
    pub path: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UnloadModelRequest {
    pub model: String,
    pub kind: String, // "llm" | "embedding"
}

#[derive(Debug, Serialize)]
pub struct ModelsListResponse {
    pub llm: Vec<String>,
    pub embedding: Vec<String>,
    pub multimodal: Vec<String>,
    pub image: Vec<String>,
}