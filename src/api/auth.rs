use axum::http::HeaderMap;
use governor::{Quota, RateLimiter, state::keyed::DefaultKeyedStateStore, clock::DefaultClock};
use nonzero_ext::nonzero;
use once_cell::sync::Lazy;

type Limiter = RateLimiter<String, DefaultKeyedStateStore<String>, DefaultClock>;
static RATE_LIMITER: Lazy<Limiter> = Lazy::new(|| {
    // Default: 60 req per minute per key
    let q = Quota::per_minute(nonzero!(60u32));
    RateLimiter::keyed(q)
});

pub fn authorize_request(headers: &HeaderMap) -> Result<(), String> {
    // Read API_KEYS from env. If empty, auth disabled.
    let keys_env = std::env::var("API_KEYS").ok().unwrap_or_default();
    let keys: Vec<String> = keys_env
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .collect();
    if keys.is_empty() {
        return Ok(());
    }
    let auth_header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if let Some(token) = auth_header.strip_prefix("Bearer ") {
        if keys.iter().any(|k| k == token) {
            // Rate limit per token (if present)
            if RATE_LIMITER.check_key(&token.to_string()).is_ok() {
                return Ok(());
            } else {
                return Err("Rate limit exceeded".to_string());
            }
        }
    }
    Err("Unauthorized".to_string())
}
