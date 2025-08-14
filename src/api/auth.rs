use axum::http::HeaderMap;

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
            return Ok(());
        }
    }
    Err("Unauthorized".to_string())
}
