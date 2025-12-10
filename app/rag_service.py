from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import chromadb

from .config import Settings

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Raised when we cannot retrieve context from ChromaDB."""


@dataclass
class RetrievedChunk:
    """Internal representation of a retrieved text chunk."""

    id: str
    text: str
    metadata: Dict[str, Any]
    distance: float | None


class RAGService:
    """Encapsulates interaction with ChromaDB and the RAG pipeline.

    Responsibilities:
    * Connect to the ChromaDB HTTP server
    * Retrieve relevant chunks for a question
    * Deduplicate overlapping/near-duplicate chunks
    * Construct a concise answer from the deduplicated context
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = self._create_client(settings)
        self._collection = self._get_collection(settings)

    @staticmethod
    def _create_client(settings: Settings) -> chromadb.HttpClient:
        try:
            logger.info(
                "Initializing ChromaDB HTTP client", extra={"host": settings.chroma_host, "port": settings.chroma_port}
            )
            client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
            return client
        except Exception as exc:  # pragma: no cover - protected by integration tests
            logger.exception("Failed to initialize ChromaDB client")
            raise RetrievalError("Failed to initialize ChromaDB client") from exc

    @staticmethod
    def _get_collection(settings: Settings):
        try:
            client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
            collection = client.get_collection(settings.chroma_collection)
            logger.info("Connected to ChromaDB collection", extra={"collection": settings.chroma_collection})
            return collection
        except Exception as exc:  # pragma: no cover - protected by integration tests
            logger.exception("Failed to get ChromaDB collection", extra={"collection": settings.chroma_collection})
            raise RetrievalError("Failed to get ChromaDB collection") from exc

    def healthy(self) -> bool:
        """Check if the RAG service can talk to ChromaDB.

        We keep this check deliberately lightweight: just ensure the
        collection is reachable by invoking a trivial method.
        """

        try:
            # "count" is inexpensive and does not pull documents
            _ = self._collection.count()
            return True
        except Exception as exc:  # pragma: no cover - runtime integration path
            logger.warning("ChromaDB health check failed", exc_info=exc)
            return False

    # ------------------------------------------------------------------
    # Core RAG operations
    # ------------------------------------------------------------------

    def answer_question(self, question: str, top_k: int | None = None) -> Tuple[str, List[RetrievedChunk]]:
        """Run a full RAG pipeline for a question.

        Parameters
        ----------
        question: str
            Natural language user question.
        top_k: Optional[int]
            Maximum number of unique context chunks to use in the answer.

        Returns
        -------
        Tuple[str, List[RetrievedChunk]]
            The generated answer and the underlying context chunks used.
        """

        max_chunks = top_k or self._settings.max_context_chunks
        # Retrieve more than we finally use so deduplication has room to work.
        n_results = min(self._settings.max_results, max_chunks * 3)

        logger.info(
            "Retrieving context from ChromaDB",
            extra={"question": question, "n_results": n_results, "max_chunks": max_chunks},
        )

        retrieved = self._retrieve_relevant_chunks(question, n_results=n_results)
        deduped = self._deduplicate_chunks(
            retrieved,
            similarity_threshold=self._settings.dedup_similarity_threshold,
            max_chunks=max_chunks,
        )

        answer = self._generate_answer(question, deduped)
        return answer, deduped

    def _retrieve_relevant_chunks(self, question: str, n_results: int) -> List[RetrievedChunk]:
        """Query ChromaDB for relevant chunks.

        This relies on the embedding function configured for the
        ChromaDB server. We pass only the raw text query.
        """

        try:
            raw = self._collection.query(query_texts=[question], n_results=n_results)
        except Exception as exc:  # pragma: no cover - integration path
            logger.exception("Error querying ChromaDB")
            raise RetrievalError("Error querying ChromaDB") from exc

        documents_lists = raw.get("documents") or []
        ids_lists = raw.get("ids") or []
        distances_lists = raw.get("distances") or []
        metadatas_lists = raw.get("metadatas") or []

        if not documents_lists:
            logger.info("No documents returned from ChromaDB", extra={"question": question})
            return []

        documents = documents_lists[0]
        ids = ids_lists[0] if ids_lists else [str(i) for i in range(len(documents))]
        distances = distances_lists[0] if distances_lists else [None] * len(documents)
        metadatas = metadatas_lists[0] if metadatas_lists else [{} for _ in documents]

        results: List[RetrievedChunk] = []
        for doc_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            results.append(
                RetrievedChunk(
                    id=str(doc_id),
                    text=text or "",
                    metadata=metadata or {},
                    distance=float(distance) if distance is not None else None,
                )
            )

        logger.info("Retrieved raw chunks from ChromaDB", extra={"count": len(results)})
        return results

    # ------------------------------------------------------------------
    # Deduplication logic
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison: lowercase and collapse whitespace."""

        return " ".join(text.lower().strip().split())

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        """Token-level Jaccard similarity between two strings.

        This is a simple but effective way to detect overlapping
        windows produced by common text chunkers.
        """

        tokens_a = set(a.split())
        tokens_b = set(b.split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    def _is_near_duplicate(self, text: str, others: List[str], threshold: float) -> bool:
        """Check if `text` is a near-duplicate of any string in `others`.

        We combine substring checks and Jaccard similarity. This keeps
        obviously overlapping chunks out of the final context passed to
        the downstream LLM.
        """

        normalized = self._normalize_text(text)
        for other in others:
            other_norm = self._normalize_text(other)

            # Exact match
            if normalized == other_norm:
                return True

            # One chunk almost fully contained in another
            if len(normalized) > 40 and len(other_norm) > 40:
                if normalized in other_norm or other_norm in normalized:
                    return True

            # Token-level overlap
            sim = self._jaccard_similarity(normalized, other_norm)
            if sim >= threshold:
                return True

        return False

    def _deduplicate_chunks(
        self,
        chunks: List[RetrievedChunk],
        similarity_threshold: float,
        max_chunks: int,
    ) -> List[RetrievedChunk]:
        """Remove near-duplicate chunks while preserving the most relevant ones.

        Chunks are first sorted by distance (ascending) so that the
        most relevant content is kept when duplicates are removed.
        """

        if not chunks:
            return []

        # Sort by distance (ascending). If distance is None, treat as worst.
        sorted_chunks = sorted(
            chunks,
            key=lambda c: float("inf") if c.distance is None else c.distance,
        )

        kept: List[RetrievedChunk] = []
        kept_texts: List[str] = []

        for chunk in sorted_chunks:
            if len(kept) >= max_chunks:
                break

            if self._is_near_duplicate(chunk.text, kept_texts, threshold=similarity_threshold):
                logger.debug("Dropping near-duplicate chunk", extra={"chunk_id": chunk.id})
                continue

            kept.append(chunk)
            kept_texts.append(chunk.text)

        logger.info(
            "Deduplicated chunks",
            extra={"input_count": len(chunks), "output_count": len(kept)},
        )
        return kept

    # ------------------------------------------------------------------
    # Answer generation (simple, LLM-free for this exercise)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_answer(question: str, chunks: List[RetrievedChunk]) -> str:
        """Generate a concise answer from the context chunks.

        In a production RAG system this is where an LLM would be
        invoked. For the purposes of this exercise, we create a
        deterministic, concise answer that clearly shows the effect of
        context deduplication.
        """

        if not chunks:
            return (
                "I could not find relevant context in the knowledge base "
                "for this question. Please try rephrasing it or ask about "
                "another machine learning concept."
            )

        # Keep each chunk short and avoid re-introducing duplication
        unique_sentences: List[str] = []
        seen_sentences: set[str] = set()

        for chunk in chunks:
            for sentence in chunk.text.split("."):
                sentence = sentence.strip()
                if not sentence:
                    continue
                normalized = sentence.lower()
                if normalized in seen_sentences:
                    continue
                seen_sentences.add(normalized)
                unique_sentences.append(sentence)

        summary = ". ".join(unique_sentences)
        if summary and not summary.endswith("."):
            summary += "."

        return summary
