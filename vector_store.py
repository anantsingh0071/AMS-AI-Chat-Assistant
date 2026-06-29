"""
vector_store.py
─────────────────────────────────────────────────────────────────────────
Pure Python FAISS vector store + document loader. No LangChain.
"""

from __future__ import annotations

import csv
import io
import re
from typing import NamedTuple

import numpy as np

from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K


class SearchResult(NamedTuple):
    chunk: str
    source: str
    score: float


# ══════════════════════════════════════════════════════════════════════════
# Document loading
# ══════════════════════════════════════════════════════════════════════════
def load_uploaded_file(uploaded_file) -> str:
    """Extract plain text from a Streamlit UploadedFile."""
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    raw = uploaded_file.read()
    uploaded_file.seek(0)

    if ext == "pdf":
        return _load_pdf(raw)
    if ext in ("txt", "md"):
        return raw.decode("utf-8", errors="replace")
    if ext == "csv":
        return _load_csv(raw)
    raise ValueError(f"Unsupported file type: .{ext}")

def _load_pdf(data: bytes) -> str:
    import io
    import PyPDF2

    reader = PyPDF2.PdfReader(io.BytesIO(data))

    # Check if the PDF is encrypted
    if reader.is_encrypted:
        try:
            # Try to decrypt with an empty password
            reader.decrypt("")
        except Exception:
            raise RuntimeError(
                "This PDF is encrypted. Please upload an unencrypted PDF."
            )

    text = []

    for page in reader.pages:
        text.append(page.extract_text() or "")

    return "\n\n".join(text)



def _load_csv(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    return "\n".join(
        ", ".join(f"{k}: {v}" for k, v in row.items()) for row in reader
    )


# ══════════════════════════════════════════════════════════════════════════
# Chunking
# ══════════════════════════════════════════════════════════════════════════
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= chunk_size:
        return [text]

    paragraphs = text.split("\n\n")
    raw_chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).lstrip()
        else:
            if current:
                raw_chunks.append(current)
            if len(para) > chunk_size:
                raw_chunks.extend(_sentence_split(para, chunk_size))
                current = ""
            else:
                current = para
    if current:
        raw_chunks.append(current)

    if overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        overlapped.append(raw_chunks[i - 1][-overlap:] + " " + raw_chunks[i])
    return overlapped


def _sentence_split(text: str, max_len: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_len:
            current = (current + " " + s).lstrip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks


# ══════════════════════════════════════════════════════════════════════════
# FAISS Vector Store
# ══════════════════════════════════════════════════════════════════════════
class VectorStore:
    """In-memory FAISS store with SentenceTransformer embeddings."""

    def __init__(self):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(f"Missing dependency: {e}. Run: pip install faiss-cpu sentence-transformers")

        import faiss as _faiss
        from sentence_transformers import SentenceTransformer as _ST

        self._faiss  = _faiss
        self._encoder = _ST(EMBEDDING_MODEL)
        self._dim     = self._encoder.get_sentence_embedding_dimension()
        self._index   = _faiss.IndexFlatIP(self._dim)
        self._meta: list[dict] = []

    def add(self, chunks: list[str], source: str) -> int:
        if not chunks:
            return 0
        vecs = self._encode(chunks)
        self._index.add(vecs)
        self._meta.extend({"chunk": c, "source": source} for c in chunks)
        return len(chunks)

    def search(self, query: str, k: int = RETRIEVER_K) -> list[SearchResult]:
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        vec = self._encode([query])
        distances, indices = self._index.search(vec, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            m = self._meta[idx]
            results.append(SearchResult(m["chunk"], m["source"], float(dist)))
        return results

    @property
    def total(self) -> int:
        return self._index.ntotal

    @property
    def sources(self) -> set[str]:
        return {m["source"] for m in self._meta}

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")