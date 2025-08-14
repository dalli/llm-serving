# syntax=docker/dockerfile:1

# ---------- Builder Stage ----------
FROM rust:1.83-slim-bookworm AS builder
WORKDIR /app

# Cache deps
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY tests ./tests
COPY README.md GEMINI.md history.md ./

# Build in release mode (default features)
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release

# ---------- Runtime Stage ----------
FROM debian:bookworm-slim AS runtime

# Create a non-root user
RUN useradd -m -u 10001 appuser

# Minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/llm-serving /usr/local/bin/llm-serving

# Default environment (override as needed)
ENV RUST_LOG=info
EXPOSE 3000

USER appuser
ENTRYPOINT ["/usr/local/bin/llm-serving"]
