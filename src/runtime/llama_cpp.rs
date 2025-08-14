use async_trait::async_trait;
use llama_cpp::{LlamaModel, LlamaParams, LlamaSession, SessionParams, Token};
use std::{fs::File, path::PathBuf};
use memmap2::Mmap;

use crate::runtime::{LlmRuntime, GenerationOptions};

pub struct LlamaCppRuntime {
    model: LlamaModel,
}

impl LlamaCppRuntime {
    pub fn new(model_path: &str) -> Result<Self, String> {
        let model_path = PathBuf::from(model_path);
        // Basic validation and memory-map to verify GGUF/GGML file
        let file = File::open(&model_path)
            .map_err(|e| format!("Failed to open model file {:?}: {}", model_path, e))?;
        // SAFETY: File is not mutated while mapping; map is read-only
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("Failed to memory-map model file {:?}: {}", model_path, e))?;

        if let Some(ext) = model_path.extension().and_then(|s| s.to_str()) {
            match ext.to_ascii_lowercase().as_str() {
                "gguf" => {
                    if mmap.len() < 4 || &mmap[0..4] != b"GGUF" {
                        return Err("Invalid GGUF header: expected 'GGUF' magic".to_string());
                    }
                }
                "ggml" => {
                    // GGML variants have multiple magics; we don't enforce here
                }
                _ => {
                    return Err(format!("Unsupported model extension: .{} (expected .gguf or .ggml)", ext));
                }
            }
        } else {
            return Err("Model file has no extension; expected .gguf or .ggml".to_string());
        }

        // Delegate to llama.cpp loader (which may use its own mmap internally)
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
    async fn generate(&self, prompt: &str, options: &GenerationOptions) -> Result<String, String> {
        let mut session = self.create_session();
        let tokens: Vec<Token> = self.model.tokenize(prompt.as_bytes(), true).map_err(|e| format!("Failed to tokenize prompt: {}", e))?; // Use self.model.tokenize
        session
            .advance_context_with_tokens(&tokens)
            .map_err(|e| format!("Failed to advance context: {}", e))?;

        let mut generated_text = String::new();
        for _ in 0..options.max_tokens {
            let token = session
                .decode_next_token(&self.model) // Use decode_next_token
                .map_err(|e| format!("Failed to decode next token: {}", e))?;
            generated_text.push_str(&self.model.token_to_piece(token)); // Use self.model.token_to_piece
        }
        Ok(generated_text)
    }
}