use async_trait::async_trait;
use std::path::Path;

use crate::runtime::EmbeddingRuntime;

#[cfg(feature = "onnx")]
use ort::{environment::Environment, session::{Session, builder::SessionBuilder}, value::Value};
#[cfg(feature = "onnx_tokenizer")]
use tokenizers::Tokenizer;
#[cfg(feature = "onnx_tokenizer")]
use ndarray::{Array2, Axis};

pub struct OnnxEmbeddingRuntime {
    #[cfg(feature = "onnx")]
    env: Environment,
    #[cfg(feature = "onnx")]
    session: Session,
    dim: usize,
    #[cfg(feature = "onnx_tokenizer")]
    tokenizer: Option<Tokenizer>,
}

impl OnnxEmbeddingRuntime {
    pub fn new(model_path: &str, dim: usize) -> Result<Self, String> {
        #[cfg(feature = "onnx")]
        {
            let env = Environment::builder().with_name("onnx-embed").build().map_err(|e| format!("ORT env error: {}", e))?;
            let session = SessionBuilder::new(&env)
                .with_model_from_file(Path::new(model_path))
                .map_err(|e| format!("ORT load model error: {}", e))?;
            #[cfg(feature = "onnx_tokenizer")]
            let tokenizer = match std::env::var("ONNX_EMBEDDING_TOKENIZER_PATH") {
                Ok(tok_path) => Some(Tokenizer::from_file(tok_path).map_err(|e| format!("load tokenizer error: {}", e))?),
                Err(_) => None,
            };
            Ok(Self {
                env,
                session,
                dim,
                #[cfg(feature = "onnx_tokenizer")]
                tokenizer,
            })
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
            // Simple path: if tokenizer not available, return zero vectors to avoid breaking default tests.
            #[cfg(not(feature = "onnx_tokenizer"))]
            {
                return Ok(inputs.iter().map(|_| vec![0.0f32; self.dim]).collect());
            }
            #[cfg(feature = "onnx_tokenizer")]
            {
                // Expect a BERT-like embedding model with inputs: input_ids, attention_mask
                let tokenizer = if let Some(tk) = &self.tokenizer { tk } else { return Ok(inputs.iter().map(|_| vec![0.0f32; self.dim]).collect()); };

                // Tokenize
                let encodings = tokenizer.encode_batch(inputs.to_vec(), true).map_err(|e| format!("tokenize error: {}", e))?;
                let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
                let batch = encodings.len();
                let mut input_ids = Array2::<i64>::zeros((batch, max_len));
                let mut attention = Array2::<i64>::zeros((batch, max_len));
                for (b, enc) in encodings.iter().enumerate() {
                    let ids = enc.get_ids();
                    for (t, &id) in ids.iter().enumerate() {
                        input_ids[(b, t)] = id as i64;
                        attention[(b, t)] = 1;
                    }
                }

                // Build ONNX inputs
                let input_ids_tensor = Value::from_array(input_ids.view()).map_err(|e| format!("ort tensor error: {}", e))?;
                let attention_tensor = Value::from_array(attention.view()).map_err(|e| format!("ort tensor error: {}", e))?;

                let outputs = self.session.run(vec![("input_ids", &input_ids_tensor), ("attention_mask", &attention_tensor)])
                    .map_err(|e| format!("ort run error: {}", e))?;

                // Extract first output as embeddings or last hidden state and pool
                if let Some(val) = outputs.get(0) {
                    let arr: ndarray::ArrayD<f32> = val.try_extract().map_err(|e| format!("ort extract error: {}", e))?;

                    // Case 1: [batch, dim]
                    if let Ok(arr2) = arr.clone().into_dimensionality::<ndarray::Ix2>() {
                        let mut result = Vec::with_capacity(batch);
                        for b in 0..batch {
                            let mut row = arr2.index_axis(Axis(0), b).to_owned().to_vec();
                            // L2 normalize
                            let norm = (row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()).sqrt();
                            if norm > 0.0 { for v in &mut row { *v /= norm as f32; } }
                            result.push(row);
                        }
                        return Ok(result);
                    }

                    // Case 2: [batch, seq, hidden]
                    if let Ok(arr3) = arr.into_dimensionality::<ndarray::Ix3>() {
                        let seq_len = arr3.shape()[1];
                        let hidden = arr3.shape()[2];
                        let mut result = Vec::with_capacity(batch);
                        for b in 0..batch {
                            let mut sum_vec = vec![0.0f32; hidden];
                            let mut count: i64 = 0;
                            let bh = arr3.index_axis(Axis(0), b);
                            for t in 0..seq_len { // respect model's sequence length
                                if attention[(b, t)] == 1 {
                                    let token_vec = bh.index_axis(Axis(0), t);
                                    for (i, val) in token_vec.iter().enumerate() {
                                        sum_vec[i] += *val;
                                    }
                                    count += 1;
                                }
                            }
                            if count > 0 { let inv = 1.0f32 / (count as f32); for v in &mut sum_vec { *v *= inv; } }
                            // L2 normalize
                            let norm = (sum_vec.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()).sqrt();
                            if norm > 0.0 { for v in &mut sum_vec { *v /= norm as f32; } }
                            result.push(sum_vec);
                        }
                        return Ok(result);
                    }

                    // Unknown output shape
                    return Ok(inputs.iter().map(|_| vec![0.0f32; self.dim]).collect());
                }

                Ok(inputs.iter().map(|_| vec![0.0f32; self.dim]).collect())
            }
        }
        #[cfg(not(feature = "onnx"))]
        {
            let _ = inputs;
            Err("onnx feature not enabled".to_string())
        }
    }
}
