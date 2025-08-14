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
  - F3: Core Engine & Concurrency
    - Added semaphore-limited Tokio worker pool to process requests concurrently (config via `ENGINE_WORKERS`)
    - Refactored worker to spawn per request while preserving backpressure via channel + semaphore
  - F1: Generation parameters
    - Extended `ChatCompletionRequest` with `max_tokens`, `temperature`, `top_p`
    - Engine applies `max_tokens` to runtime generation; tests updated
  - F4: Response Caching
    - Added `moka` cache with TTL and capacity; keyed by request hash
    - Cached non-streaming chat responses; preserved streaming semantics
  - F5: Dynamic Model Management
    - Refactored runtimes to `Arc<RwLock<HashMap<...>>>` for dynamic updates
    - Added admin endpoints: `GET /admin/models`, `POST /admin/models/load`, `POST /admin/models/unload`
    - Implemented load/unload helpers in engine; kept dummy fallbacks

- **Issues Encountered:**
  - Concurrency orchestration within a single worker loop caused potential head-of-line blocking
  - Backward compatibility risk when adding new request fields
  - Cache get is async in `moka::future`; initial code missed `.await`
  - Sharing runtimes across workers and admin mutations required read/write synchronization
- **Solution:**
  - Switched to per-request task spawn with a shared `Semaphore` to bound concurrency, avoiding blocking the receiver loop
  - Used optional fields with serde defaulting to maintain compatibility
  - Fixed by awaiting cache get and cloning response for insertion
  - Used `RwLock` to allow concurrent read access and exclusive writes during model changes

- **Retrospective:**
  - **What went well:** Simple, bounded concurrency model improved throughput without complicating the engine interface.
  - **What to improve:** Add graceful shutdown and drain logic; expose concurrency in config; add per-model concurrency limits; wire `temperature/top_p` through runtimes; add cache invalidation controls and metrics; persist model configs.

## Process Update: Per-task workflow and helper

- **Duration:** 2025-08-14 ~ 2025-08-14
- **Completed Work:**
  - Documented per-task rule in `GEMINI.md` (4.3 Task Lifecycle Rule)
  - Added `scripts/finish_task.sh` to append standardized entries to `history.md`
- **Issues Encountered:**
  - CI-based enforcement was considered but deemed unnecessary
- **Solution:**
  - Dropped CI enforcement; kept lightweight local script and documentation
- **Retrospective:**
  - **What went well:** Lightweight process aligns with local dev flow
  - **What to improve:** Consider adding a pre-push hook template later if team grows