use async_trait::async_trait;

use crate::runtime::{LlmRuntime, MultimodalRuntime, GenerationOptions};

pub struct DummyRuntime;

impl DummyRuntime {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl LlmRuntime for DummyRuntime {
    async fn generate(&self, prompt: &str, options: &GenerationOptions) -> Result<String, String> {
        let truncated: String = prompt.chars().take(options.max_tokens as usize).collect();
        Ok(format!("Echo: {}", truncated))
    }
}

#[async_trait]
impl MultimodalRuntime for DummyRuntime {
    async fn generate_from_vision(
        &self,
        text: &str,
        image_urls: &[String],
        options: &GenerationOptions,
    ) -> Result<String, String> {
        let mut response = format!("Echo(Vision): {}", text);
        if !image_urls.is_empty() {
            response.push_str(&format!(" | images={}", image_urls.len()));
        }
        let truncated: String = response.chars().take(options.max_tokens as usize).collect();
        Ok(truncated)
    }
}
