use async_trait::async_trait;

use crate::runtime::ImageGenRuntime;

pub struct DummyImageRuntime;

impl DummyImageRuntime {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl ImageGenRuntime for DummyImageRuntime {
    async fn generate_images(&self, _prompt: &str, n: u32, size: &str) -> Result<Vec<Vec<u8>>, String> {
        // Returns n placeholder PNG-like byte arrays tagged with size
        let mut result = Vec::new();
        let header = format!("DUMMY_PNG:{}:", size).into_bytes();
        for _ in 0..n { result.push(header.clone()); }
        Ok(result)
    }
}
