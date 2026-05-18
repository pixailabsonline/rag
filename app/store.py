from dataclasses import dataclass
from threading import Lock


@dataclass
class StoredChunk:
    chunk_id: str
    text: str
    chunk_hash: str
    index: int


class DocumentStore:
    def __init__(self):
        self._documents: dict[str, list[StoredChunk]] = {}
        self._lock = Lock()

    def store_document(self, document_id: str, chunks: list[StoredChunk]) -> None:
        with self._lock:
            self._documents[document_id] = chunks

    def get_document(self, document_id: str) -> list[StoredChunk] | None:
        return self._documents.get(document_id)

    def clear(self):
        with self._lock:
            self._documents.clear()


document_store = DocumentStore()
