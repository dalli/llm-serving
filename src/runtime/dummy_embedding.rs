use async_trait::async_trait;

use crate::runtime::EmbeddingRuntime;

pub struct DummyEmbeddingRuntime {
    dimension: usize,
}

impl DummyEmbeddingRuntime {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl EmbeddingRuntime for DummyEmbeddingRuntime {
    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let mut results: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
        for text in inputs {
            let mut vec = vec![0.0_f32; self.dimension];
            let mut hash: u64 = 1469598103934665603; // FNV offset basis
            for b in text.as_bytes() {
                hash ^= *b as u64;
                hash = hash.wrapping_mul(1099511628211);
            }
            // Fill vector deterministically from hash
            for i in 0..self.dimension {
                let v = ((hash.rotate_left((i % 64) as u32) % 1000) as f32) / 1000.0;
                vec[i] = v;
            }
            // L2 normalize
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut vec {
                    *v /= norm;
                }
            }
            results.push(vec);
        }
        Ok(results)
    }
}
