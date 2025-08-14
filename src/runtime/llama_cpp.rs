use async_trait::async_trait;
use llama_cpp::{LlamaModel, LlamaParams, LlamaSession, SessionParams, Token};
use std::path::PathBuf;

use crate::runtime::LlmRuntime;

pub struct LlamaCppRuntime {
    model: LlamaModel,
}

impl LlamaCppRuntime {
    pub fn new(model_path: &str) -> Result<Self, String> {
        let model_path = PathBuf::from(model_path);
        let model = LlamaModel::load_from_file(model_path, LlamaParams::default())
            .map_err(|e| format!("Failed to load Llama model: {}", e))?;
        Ok(Self { model })
    }

    fn create_session(&self) -> LlamaSession {
        self.model.create_session(SessionParams::default()).expect("Failed to create session")
    }
}

#[async_trait]
impl LlmRuntime for LlamaCppRuntime {
    async fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String, String> {
        let mut session = self.create_session();
        let tokens: Vec<Token> = self.model.tokenize(prompt.as_bytes(), true).map_err(|e| format!("Failed to tokenize prompt: {}", e))?; // Use self.model.tokenize
        session
            .advance_context_with_tokens(&tokens)
            .map_err(|e| format!("Failed to advance context: {}", e))?;

        let mut generated_text = String::new();
        for _ in 0..max_tokens {
            let token = session
                .decode_next_token(&self.model) // Use decode_next_token
                .map_err(|e| format!("Failed to decode next token: {}", e))?;
            generated_text.push_str(&self.model.token_to_piece(token)); // Use self.model.token_to_piece
        }
        Ok(generated_text)
    }
}