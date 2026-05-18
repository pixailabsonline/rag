from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.store import document_store


@dataclass
class ScoredChunk:
    chunk_id: str
    text: str
    chunk_hash: str
    index: int
    score: float


def retrieve(
    document_id: str,
    query: str,
    top_k: int,
    min_score: float,
    max_context_chars: int,
) -> list[ScoredChunk]:
    chunks = document_store.get_document(document_id)
    if chunks is None:
        return []

    if not chunks:
        return []

    texts = [c.text for c in chunks]
    corpus = texts + [query]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    query_vec = tfidf_matrix[-1]
    doc_vecs = tfidf_matrix[:-1]

    similarities = cosine_similarity(query_vec, doc_vecs).flatten()

    scored = []
    for i, score in enumerate(similarities):
        if score >= min_score:
            scored.append(
                ScoredChunk(
                    chunk_id=chunks[i].chunk_id,
                    text=chunks[i].text,
                    chunk_hash=chunks[i].chunk_hash,
                    index=chunks[i].index,
                    score=float(score),
                )
            )

    scored.sort(key=lambda x: x.score, reverse=True)
    scored = scored[:top_k]

    result = []
    total_chars = 0
    for chunk in scored:
        if total_chars + len(chunk.text) > max_context_chars:
            break
        result.append(chunk)
        total_chars += len(chunk.text)

    return result
