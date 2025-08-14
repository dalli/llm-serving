use async_trait::async_trait;

use crate::runtime::{MultimodalRuntime, GenerationOptions};

// Placeholder LLaVA runtime skeleton: vision encode -> prompt augment -> LLM generate
pub struct LlavaRuntime {
    // TODO: store vision encoder session, projection, and llama session/model
}

impl LlavaRuntime {
    pub fn new(_vision_model_path: &str, _proj_path: &str, _llm_model_path: &str) -> Result<Self, String> {
        // TODO: initialize ONNX vision encoder + projection and llama model
        Ok(Self {})
    }
}

#[async_trait]
impl MultimodalRuntime for LlavaRuntime {
    async fn generate_from_vision(
        &self,
        text: &str,
        image_urls: &[String],
        options: &GenerationOptions,
    ) -> Result<String, String> {
        // TODO: fetch images, run vision encoder, project to tokens, and call llama
        let mut out = format!("[LLaVA] {}", text);
        if !image_urls.is_empty() {
            out.push_str(&format!(" ({} images)", image_urls.len()));
        }
        let truncated: String = out.chars().take(options.max_tokens as usize).collect();
        Ok(truncated)
    }
}
