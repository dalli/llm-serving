use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

#[derive(Debug)]
pub enum AppError {
    InternalServerError(String),
    BadRequest(String),
    NotFound(String),
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub message: String,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            AppError::InternalServerError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
        };

        let body = Json(ErrorResponse {
            message: error_message,
        });

        (status, body).into_response()
    }
}

impl From<String> for AppError {
    fn from(err: String) -> Self {
        AppError::InternalServerError(err)
    }
}