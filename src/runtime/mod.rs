use async_trait::async_trait;

#[cfg(feature = "llama")]
pub mod llama_cpp;
pub mod dummy;

#[async_trait]
pub trait LlmRuntime: Send + Sync {
    async fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String, String>;
}
