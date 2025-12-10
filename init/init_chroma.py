"""One-time initialization service to populate ChromaDB with ML content.

This container is designed to run exactly once on deployment. It connects
to the ChromaDB HTTP server and ensures that a collection with basic
machine learning concepts exists and is populated.

In real deployments this script would ingest data from files or object
storage. For this exercise we embed a small, representative corpus
inline so that the RAG service can answer basic conceptual questions.
"""

from __future__ import annotations

import os
import time
from typing import List

import chromadb


CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ml-concepts")


def get_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def wait_for_chroma(max_attempts: int = 30, delay_seconds: int = 2) -> None:
    """Wait until ChromaDB is reachable.

    Although Docker Compose uses health checks, the server may still be
    coming up when this container starts. This retry loop makes the
    initialization more robust in face of transient failures.
    """

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            client = get_client()
            client.heartbeat()
            return
        except Exception:
            time.sleep(delay_seconds)

    raise RuntimeError("ChromaDB did not become ready in time")


def get_or_create_collection():
    client = get_client()
    # get_or_create_collection is idempotent
    return client.get_or_create_collection(name=COLLECTION_NAME)


def collection_is_populated(collection) -> bool:
    try:
        return collection.count() > 0
    except Exception:
        return False


def seed_documents() -> List[str]:
    """Return a small corpus of ML-related text chunks.

    These intentionally have some conceptual overlap so that the
    deduplication logic in the RAG service has something to work with.
    """

    docs = [
        (
            "Machine learning is a subfield of artificial intelligence "
            "that focuses on algorithms which learn patterns from data "
            "rather than being explicitly programmed. A machine learning "
            "model is trained on historical examples and then used to "
            "make predictions or decisions on new, unseen data."
        ),
        (
            "Supervised learning is a type of machine learning where the "
            "algorithm is trained on labeled data. Each training example "
            "includes both the input features and the correct output, so "
            "the model learns to map inputs to outputs. Common supervised "
            "tasks include classification and regression."
        ),
        (
            "Unsupervised learning deals with unlabeled data. The goal is "
            "to discover hidden structure, such as clusters or lower-"\
            "dimensional representations, without explicit target labels. "
            "Clustering and dimensionality reduction are common "
            "unsupervised learning techniques."
        ),
        (
            "Overfitting occurs when a machine learning model learns "
            "noise or idiosyncrasies in the training data instead of "
            "general patterns. An overfitted model performs well on the "
            "training set but poorly on new data. Regularization, "
            "cross-validation, and gathering more data are common "
            "strategies to combat overfitting."
        ),
        (
            "A training set is the portion of available data used to fit "
            "the parameters of a machine learning model. A validation set "
            "is used for model selection and hyperparameter tuning, while "
            "a test set is held out until the end to estimate how well the "
            "chosen model is likely to perform on truly unseen data."
        ),
        (
            "Gradient descent is an iterative optimization algorithm used "
            "to minimize a loss function. At each step it updates model "
            "parameters in the direction of the negative gradient, which "
            "points toward the steepest decrease of the loss. Variants "
            "like stochastic gradient descent and Adam are widely used in "
            "deep learning."
        ),
        (
            "In classification problems the goal is to assign each input "
            "example to one of several discrete categories or classes. "
            "Common evaluation metrics include accuracy, precision, "
            "recall, and F1 score. Logistic regression, decision trees, "
            "and neural networks can all be used for classification."
        ),
        (
            "Feature engineering is the process of transforming raw data "
            "into features that better capture the underlying problem "
            "structure for a machine learning model. This can involve "
            "cleaning, encoding categorical variables, scaling numeric "
            "values, and creating domain-specific features."
        ),
    ]

    return docs


def populate_collection(collection) -> None:
    docs = seed_documents()
    ids = [f"ml_doc_{i}" for i in range(len(docs))]
    metadatas = [{"source": "seed_corpus", "index": i} for i in range(len(docs))]

    # Add in a single batch for simplicity. Chroma will compute
    # embeddings server-side if configured with an embedding function.
    collection.add(ids=ids, documents=docs, metadatas=metadatas)


def main() -> None:
    wait_for_chroma()
    collection = get_or_create_collection()

    if collection_is_populated(collection):
        # Idempotent behavior: do nothing if the collection already has data.
        return

    populate_collection(collection)


if __name__ == "__main__":  # pragma: no cover - executed in container
    main()
