# Work History

## Phase 1: Text LLM MVP

- **Duration:** 2025-08-14 ~ 2025-08-14
- **Completed Work:**
  - Project initial setup
    - Created project structure (`main.rs`, `api`, `engine`, `runtime` modules)
    - Added core dependencies in `Cargo.toml` (axum, tokio, serde, tracing, uuid, tokio-stream, async-trait, tracing-subscriber, futures, tower; `llama_cpp` optional via feature)
  - F1: Implement Chat API
    - Defined OpenAI Chat API request/response DTOs using `serde`
    - Implemented `/v1/chat/completions` route and handler using `axum`
    - Implemented `text/event-stream` SSE streaming responses with final `[DONE]`
    - Implemented standard API error response format
  - F2: Text Model Runtime
    - Defined `LlmRuntime` trait (abstraction)
    - Implemented `DummyRuntime` for development
    - Added optional `LlamaCppRuntime` behind `llama` feature; loads model via `LLAMA_MODEL_PATH`
  - F3: Core Engine & Concurrency
    - Implemented `CoreEngine` to manage runtimes
    - Implemented request queue using `tokio::mpsc`
    - Implemented worker to process requests and integrated API handler with engine
  - Tests
    - Added integration tests for non-stream and stream chat completions

- **Issues Encountered:**
  - **Issue 1:** `llama_cpp` API method mismatches caused build failures
    - **Cause:** Crate API differences (e.g., tokenize/decode methods unavailable as initially referenced)
    - **Solution:** Made `llama_cpp` optional behind `llama` feature and provided `DummyRuntime` as default. Selected llama runtime via `LLAMA_MODEL_PATH` when the feature is enabled.
  - **Issue 2:** Integration test harness failures (unresolved crate and missing `oneshot`)
    - **Cause:** No `lib` target for tests to import; `tower::util::ServiceExt` not imported
    - **Solution:** Added `src/lib.rs`; added `tower` dependency; imported `ServiceExt`

- **Retrospective:**
  - **What went well:** Clear separation of API, engine, and runtime; SSE streaming works; integration tests provided fast correctness checks.
  - **What to improve:** Finalize `LlamaCppRuntime` against the crate’s current API; add token accounting in `usage`; make generation parameters configurable; expand error handling and validation.

## Phase 2: Multimodal Support (in progress)

- **Duration:** 2025-08-14 ~ 2025-08-14
- **Completed Work:**
  - F1: Implement Embeddings API
    - Defined Embeddings API DTOs (`EmbeddingsRequest`, `EmbeddingsResponse`, etc.)
    - Implemented `/v1/embeddings` route and handler
  - F2: Embedding Model Runtime
    - Defined `EmbeddingRuntime` trait
    - Implemented `DummyEmbeddingRuntime` (deterministic, normalized vectors) and integrated into `CoreEngine`
  - Tests
    - Added integration test to validate `/v1/embeddings` returns a list with two embeddings

- **Issues Encountered:**
  - None yet

- **Retrospective:**
  - **What went well:** Embeddings API wiring mirrored chat flow, enabling quick integration and testing.
  - **What to improve:** Add real ONNX runtime-backed implementation and token accounting for usage.