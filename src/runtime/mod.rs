use async_trait::async_trait;

#[cfg(feature = "llama")]
pub mod llama_cpp;
pub mod dummy;
pub mod dummy_embedding;

#[async_trait]
pub trait LlmRuntime: Send + Sync {
    async fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String, String>;
}

#[async_trait]
pub trait EmbeddingRuntime: Send + Sync {
    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, String>;
}
