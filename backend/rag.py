"""
rag.py — Real retrieval-augmented generation layer for CareRights AI.

Loads the source corpus (medical guidelines + insurance rights), splits it into
chunks, and retrieves the most relevant chunks for a given query using TF-IDF +
cosine similarity. This is a genuine, deterministic, fully offline retrieval
method — no external embedding API needed, which keeps this reproducible and
free to run. For production, swap in a proper embedding model
(e.g. Voyage AI, OpenAI embeddings, or a local sentence-transformers model) —
the retrieve() interface below stays the same either way.
"""

import re
from pathlib import Path
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class Chunk:
    chunk_id: str
    topic: str
    content: str
    source_file: str


def _parse_corpus_file(path: Path) -> list[Chunk]:
    """Parse our simple CHUNK_ID / TOPIC / CONTENT markdown format into Chunk objects."""
    text = path.read_text(encoding="utf-8")
    blocks = text.split("---\n")
    chunks = []
    for block in blocks:
        block = block.strip()
        if not block.startswith("CHUNK_ID:"):
            continue
        chunk_id_match = re.search(r"CHUNK_ID:\s*(.+)", block)
        topic_match = re.search(r"TOPIC:\s*(.+)", block)
        content_match = re.search(r"CONTENT:\s*(.+)", block, re.DOTALL)
        if not (chunk_id_match and topic_match and content_match):
            continue
        chunks.append(Chunk(
            chunk_id=chunk_id_match.group(1).strip(),
            topic=topic_match.group(1).strip(),
            content=" ".join(content_match.group(1).strip().split()),
            source_file=path.name,
        ))
    return chunks


class RAGIndex:
    """Loads and indexes a named corpus (e.g. 'medical' or 'insurance') for retrieval."""

    def __init__(self, filename: str):
        self.chunks = _parse_corpus_file(DATA_DIR / filename)
        if not self.chunks:
            raise ValueError(f"No chunks parsed from {filename} — check corpus formatting.")
        corpus_texts = [f"{c.topic}. {c.content}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus_texts)

    def retrieve(self, query: str, top_k: int = 2, min_score: float = 0.05) -> list[dict]:
        """Return the top_k most relevant chunks for a query, each with a relevance score.
        Chunks scoring below min_score are dropped — this is what lets an agent honestly
        say 'not covered by available sources' instead of forcing a weak match."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        results = []
        for chunk, score in ranked[:top_k]:
            if score < min_score:
                continue
            results.append({
                "chunk_id": chunk.chunk_id,
                "topic": chunk.topic,
                "content": chunk.content,
                "source_file": chunk.source_file,
                "relevance_score": round(float(score), 3),
            })
        return results


# Load both indexes once at import time (real startup cost, done once per server run)
medical_index = RAGIndex("medical_guidelines.md")
insurance_index = RAGIndex("insurance_rights.md")


def retrieve_medical(query: str, top_k: int = 2) -> list[dict]:
    return medical_index.retrieve(query, top_k=top_k)


def retrieve_insurance(query: str, top_k: int = 2) -> list[dict]:
    return insurance_index.retrieve(query, top_k=top_k)


if __name__ == "__main__":
    # Quick manual test — run: python rag.py
    test_query = "My insurer denied coverage for a knee replacement after physiotherapy failed"
    print("=== Medical retrieval ===")
    for r in retrieve_medical(test_query):
        print(f"[{r['chunk_id']}] score={r['relevance_score']}  {r['topic']}")
    print("\n=== Insurance retrieval ===")
    for r in retrieve_insurance(test_query):
        print(f"[{r['chunk_id']}] score={r['relevance_score']}  {r['topic']}")
