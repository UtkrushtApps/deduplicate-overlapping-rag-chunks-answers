# Solution Steps

1. Review the existing architecture to understand the three containers involved: (1) a ChromaDB server, (2) a one-time initialization service that loads data into ChromaDB, and (3) a FastAPI app that exposes the RAG API.

2. Define a clear Docker Compose setup (`docker-compose.yml`) that wires these three services together: configure ChromaDB with a health check, add an `init-chroma` service that depends on Chroma health, and then a `rag-app` service that waits for both Chroma and the init service to complete.

3. Create a Dockerfile for the FastAPI RAG application (`app/Dockerfile`): use a slim Python base image, install dependencies from `requirements.txt`, copy the application code, expose port 8000, and add a container-level health check that pings the `/health` endpoint.

4. Create a Dockerfile for the init service (`init/Dockerfile`): use a slim Python base image, install only `chromadb`, copy the init script, and configure the container to run `python init_chroma.py` once and then exit.

5. Implement minimal but explicit dependency lists in `app/requirements.txt` and `init/requirements.txt` so the containers can install FastAPI, Uvicorn, ChromaDB, and related libraries without pulling in unnecessary packages.

6. Implement a simple configuration module (`app/config.py`) using a dataclass `Settings` that reads all important knobs (Chroma host/port, collection name, max results, max context chunks, dedup similarity threshold, log level) from environment variables, with sensible defaults for local development.

7. Implement structured logging (`app/logging_config.py`) using `logging.config.dictConfig`: configure a console handler with a consistent log format and expose a helper `get_logger` to retrieve named loggers throughout the codebase.

8. Design Pydantic models in `app/models.py` for strong typing and validation: `QueryRequest` for incoming questions, `ContextChunk` for the contextual snippets returned to clients, `QueryResponse` for the final RAG answer, and `HealthStatus` for the `/health` endpoint.

9. Implement the ChromaDB initialization script in `init/init_chroma.py`: wait for Chroma to become ready with a retry loop, obtain or create the target collection, check whether it already has documents (idempotency), and if not, insert a small but representative corpus of machine learning concept chunks with IDs and basic metadata.

10. In `init/init_chroma.py`, embed overlapping and related chunks about ML concepts (e.g., machine learning definition, supervised vs unsupervised learning, overfitting, gradient descent) so that the RAG service has enough content to demonstrate deduplication of overlapping context windows.

11. Create the core RAG service in `app/rag_service.py` by implementing a `RAGService` class that: (a) connects to ChromaDB via `chromadb.HttpClient` using configured host/port, and (b) obtains the configured collection, raising a `RetrievalError` if connection or collection access fails.

12. Inside `RAGService`, implement a lightweight `healthy()` method that calls `collection.count()` and returns `True` on success, `False` on failure, so the FastAPI health endpoint can accurately reflect Chroma connectivity without being expensive.

13. Implement `_retrieve_relevant_chunks()` in `RAGService`: issue a `collection.query()` with `query_texts=[question]` and `n_results=n_results`, then convert the response dict into a list of `RetrievedChunk` dataclass instances (id, text, metadata, distance) while handling missing fields and logging how many chunks were retrieved.

14. Design a text normalization helper `_normalize_text()` that lowercases strings and collapses whitespace to create stable representations used in deduplication, ensuring that superficial differences (case, spacing) do not prevent matching.

15. Implement `_jaccard_similarity()` to compute a simple token-level Jaccard similarity between two normalized strings; this provides a lightweight estimate of how much two chunks overlap in terms of vocabulary.

16. Implement `_is_near_duplicate()` to check whether a candidate chunk is a near-duplicate of any existing kept chunk by: (1) checking for exact normalized equality, (2) checking for substring containment for sufficiently long chunks, and (3) computing Jaccard similarity and comparing against a configurable threshold (e.g., 0.8).

17. Implement `_deduplicate_chunks()` in `RAGService`: sort all retrieved chunks by distance (ascending) so more relevant chunks are considered first, then iterate and keep at most `max_chunks`, skipping any chunk for which `_is_near_duplicate()` returns `True`. Log the input and output counts for observability.

18. Implement `_generate_answer()` in `RAGService` as a deterministic, LLM-free summarizer: take the deduplicated chunks, split them into sentences, drop repeated sentences using a case-insensitive set, and join the unique sentences into a concise answer string. This ensures that even within selected chunks, obvious sentence-level repetition is removed.

19. Implement `answer_question()` as the main RAG pipeline: compute how many results to request (`n_results = min(max_results, max_chunks * 3)`), call `_retrieve_relevant_chunks()`, run `_deduplicate_chunks()` with the configured similarity threshold, then call `_generate_answer()` and return both the answer and the deduplicated chunk list.

20. Wire up the FastAPI application in `app/main.py`: use a lifespan context manager to configure logging once at startup, instantiate `RAGService` with the global `settings`, and attach both to `app.state` so they can be used by dependency-injected endpoint functions.

21. Add CORS middleware in `app/main.py` (allowing all origins for simplicity) to support local tools and browser-based frontends without additional configuration issues.

22. Implement the `/health` endpoint in `app/main.py` to call `rag_service.healthy()`, return a `HealthStatus` object with `status` set to `ok` or `degraded` and `chroma` set to `ok` or `unreachable`, and log a warning when Chroma is down while still returning HTTP 200 to keep health probes functioning.

23. Implement the `/query` POST endpoint in `app/main.py`: validate the incoming `QueryRequest`, call `rag_service.answer_question()` with `top_k` falling back to configured `max_context_chunks`, map internal `RetrievedChunk` objects to the public `ContextChunk` schema, and return a `QueryResponse`. Catch `RetrievalError` and surface it as an HTTP 502 with logging so operational issues are visible.

24. Ensure that Docker Compose environment variables (e.g., `CHROMA_HOST`, `CHROMA_PORT`, `COLLECTION_NAME`, `MAX_CONTEXT_CHUNKS`, `DEDUP_SIMILARITY_THRESHOLD`) are correctly consumed by the `Settings` dataclass, making it easy to tune deduplication behavior and retrieval sizes without changing code.

25. Bring the system up with `docker-compose up --build`, watch the logs to confirm: (1) ChromaDB becomes healthy, (2) the `init-chroma` service connects and either populates or skips the collection, and (3) the `rag-app` passes its own health check and logs successful connection to Chroma.

26. Test retrieval and deduplication manually by sending a POST to `/query` with a question like "What is machine learning?" and verifying that the `chunks` returned in `QueryResponse` are semantically similar but not near-identical (i.e., overlapping windows are removed), and the `answer` is concise without repeated sentences.

