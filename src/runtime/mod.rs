use async_trait::async_trait;

#[cfg(feature = "llama")]
pub mod llama_cpp;
pub mod dummy;
pub mod dummy_embedding;
pub mod sampler;
#[cfg(feature = "onnx")]
pub mod onnx_embedding;

#[async_trait]
pub trait MultimodalRuntime: Send + Sync {
    async fn generate_from_vision(
        &self,
        text: &str,
        image_urls: &[String],
        options: &GenerationOptions,
    ) -> Result<String, String>;
}

#[async_trait]
pub trait LlmRuntime: Send + Sync {
    async fn generate(&self, prompt: &str, options: &GenerationOptions) -> Result<String, String>;
}

#[async_trait]
pub trait EmbeddingRuntime: Send + Sync {
    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, String>;
}

#[derive(Debug, Clone)]
pub struct GenerationOptions {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
}

impl GenerationOptions {
    pub fn from_request(max_tokens: Option<u32>, temperature: Option<f32>, top_p: Option<f32>) -> Self {
        Self {
            max_tokens: max_tokens.unwrap_or(100),
            temperature: temperature.unwrap_or(1.0),
            top_p: top_p.unwrap_or(1.0),
        }
    }
}
