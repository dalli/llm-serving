use async_trait::async_trait;
use std::path::Path;

use crate::runtime::EmbeddingRuntime;

#[cfg(feature = "onnx")]
use ort::{environment::Environment, session::{Session, builder::SessionBuilder}};

pub struct OnnxEmbeddingRuntime {
    #[cfg(feature = "onnx")]
    env: Environment,
    #[cfg(feature = "onnx")]
    session: Session,
    dim: usize,
}

impl OnnxEmbeddingRuntime {
    pub fn new(model_path: &str, dim: usize) -> Result<Self, String> {
        #[cfg(feature = "onnx")]
        {
            let env = Environment::builder().with_name("onnx-embed").build().map_err(|e| format!("ORT env error: {}", e))?;
            let session = SessionBuilder::new(&env)
                .with_model_from_file(Path::new(model_path))
                .map_err(|e| format!("ORT load model error: {}", e))?;
            Ok(Self { env, session, dim })
        }
        #[cfg(not(feature = "onnx"))]
        {
            let _ = (model_path, dim);
            Err("onnx feature not enabled".to_string())
        }
    }
}

#[async_trait]
impl EmbeddingRuntime for OnnxEmbeddingRuntime {
    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, String> {
        #[cfg(feature = "onnx")]
        {
            // This is a placeholder: actual tokenization and model IO binding depend on the specific model.
            // For now, return zero vectors with the configured dimension.
            Ok(inputs.iter().map(|_| vec![0.0f32; self.dim]).collect())
        }
        #[cfg(not(feature = "onnx"))]
        {
            let _ = inputs;
            Err("onnx feature not enabled".to_string())
        }
    }
}
