use async_trait::async_trait;

use crate::runtime::LlmRuntime;

pub struct DummyRuntime;

impl DummyRuntime {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl LlmRuntime for DummyRuntime {
    async fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String, String> {
        let truncated: String = prompt.chars().take(max_tokens as usize).collect();
        Ok(format!("Echo: {}", truncated))
    }
}
