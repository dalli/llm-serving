use rand::{rngs::StdRng, Rng as RandRng};

/// Top-p (nucleus) + temperature sampling over a vector of token logits.
/// This is a generic helper intended to be used by runtimes that can expose logits.
#[allow(dead_code)]
pub fn sample_token_index_from_logits(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    rng: &mut StdRng,
) -> Option<usize> {
    if logits.is_empty() {
        return None;
    }
    // Apply temperature: logits / T, then softmax
    let t = if temperature <= 0.0 { 1e-6 } else { temperature };
    let mut max_logit = f32::NEG_INFINITY;
    for &v in logits { if v > max_logit { max_logit = v; } }
    // Stabilize with max subtraction and temperature
    let mut probs: Vec<f32> = logits.iter().map(|&z| ((z - max_logit) / t).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 { return Some(0); }
    for p in &mut probs { *p /= sum; }

    // Sort indices by probability descending
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap_or(std::cmp::Ordering::Equal));

    // Build nucleus up to top_p cumulative probability
    let mut nucleus: Vec<(usize, f32)> = Vec::new();
    let mut cumulative = 0.0f32;
    let threshold = top_p.clamp(0.0, 1.0);
    for &i in &indices {
        let p = probs[i];
        nucleus.push((i, p));
        cumulative += p;
        if cumulative >= threshold { break; }
    }

    // Sample from nucleus
    let sum_p: f32 = nucleus.iter().map(|(_, p)| *p).sum();
    let mut r = rng.r#gen::<f32>() * sum_p.max(1e-8);
    for (i, p) in nucleus {
        if r <= p { return Some(i); }
        r -= p;
    }
    Some(indices[0])
}
