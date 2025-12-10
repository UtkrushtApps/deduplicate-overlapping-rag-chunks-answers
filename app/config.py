import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    """Application configuration loaded from environment variables.

    Using a simple dataclass keeps dependencies light while still
    supporting 12-factor style configuration.
    """

    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "ML RAG Service"))

    # ChromaDB connection
    chroma_host: str = field(default_factory=lambda: os.getenv("CHROMA_HOST", "localhost"))
    chroma_port: int = field(default_factory=lambda: int(os.getenv("CHROMA_PORT", "8000")))
    chroma_collection: str = field(default_factory=lambda: os.getenv("COLLECTION_NAME", "ml-concepts"))

    # Retrieval / RAG behavior
    max_results: int = field(default_factory=lambda: int(os.getenv("MAX_RESULTS", "12")))
    max_context_chunks: int = field(default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHUNKS", "4")))
    dedup_similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.8"))
    )

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


settings = Settings()
