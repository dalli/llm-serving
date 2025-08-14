use async_trait::async_trait;

use crate::runtime::{MultimodalRuntime, GenerationOptions};

#[cfg(feature = "llama")]
use crate::runtime::llama_cpp::LlamaCppRuntime;
#[cfg(feature = "onnx")]
use ort::{environment::Environment, session::{Session, builder::SessionBuilder}};
#[cfg(feature = "onnx")]
use std::path::Path;

// LLaVA runtime: loads vision encoder + projection (ONNX) and delegates text generation to llama.cpp
pub struct LlavaRuntime {
    #[cfg(feature = "onnx")]
    vision_env: Environment,
    #[cfg(feature = "onnx")]
    vision_session: Session,
    #[cfg(feature = "onnx")]
    projection_session: Session,
    #[cfg(feature = "llama")]
    llm: LlamaCppRuntime,
}

impl LlavaRuntime {
    pub fn new(vision_model_path: &str, proj_path: &str, llm_model_path: &str) -> Result<Self, String> {
        #[cfg(feature = "onnx")]
        let vision_env = Environment::builder().with_name("llava-vision").build().map_err(|e| format!("ORT env error: {}", e))?;
        #[cfg(feature = "onnx")]
        let vision_session = SessionBuilder::new(&vision_env)
            .with_model_from_file(Path::new(vision_model_path))
            .map_err(|e| format!("ORT load vision model error: {}", e))?;
        #[cfg(feature = "onnx")]
        let projection_session = SessionBuilder::new(&vision_env)
            .with_model_from_file(Path::new(proj_path))
            .map_err(|e| format!("ORT load projection model error: {}", e))?;

        #[cfg(feature = "llama")]
        let llm = LlamaCppRuntime::new(llm_model_path)?;

        Ok(Self {
            #[cfg(feature = "onnx")]
            vision_env,
            #[cfg(feature = "onnx")]
            vision_session,
            #[cfg(feature = "onnx")]
            projection_session,
            #[cfg(feature = "llama")]
            llm,
        })
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
        // NOTE: For now, we do not execute the vision encoder path to keep
        // default builds fast and stable. We augment the prompt with image count
        // and delegate to the LLM runtime. A future change will run vision -> projection
        // to obtain visual tokens and condition generation.

        let mut augmented_prompt = String::new();
        if !image_urls.is_empty() {
            augmented_prompt.push_str(&format!("[images:{}] ", image_urls.len()));
        }
        augmented_prompt.push_str(text);

        #[cfg(feature = "llama")]
        {
            return self.llm.generate(&augmented_prompt, &options).await;
        }

        #[allow(unreachable_code)]
        {
            // Fallback (should not happen with llava feature), mimic truncation behavior
            let truncated: String = augmented_prompt.chars().take(options.max_tokens as usize).collect();
            Ok(truncated)
        }
    }
}
