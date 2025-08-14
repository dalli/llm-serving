use async_trait::async_trait;

use crate::runtime::{LlmRuntime, GenerationOptions};

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
