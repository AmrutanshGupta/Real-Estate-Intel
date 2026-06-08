import re
import numpy as np


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, also breaking on bare newlines."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = []
    for s in sentences:
        result.extend([p.strip() for p in s.split('\n') if p.strip()])
    return [s for s in result if len(s) > 3]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(
    pages: list[dict],
    similarity_threshold: float = 0.45,
    encode_fn=None,         
) -> list[dict]:
    """
    Semantically chunk pages into variable-length chunks.

    Parameters
    ----------
    pages               : list of page dicts with keys: text, source, page_num, …
    similarity_threshold: cosine similarity below which a new chunk starts (0–1)
    encode_fn           : a callable that encodes a list of strings to a numpy
                          embedding matrix.  Injected by SearchEngine so we don't
                          load a second copy of MiniLM.  Falls back to a bundled
                          lightweight model only if None is passed (testing only).
    """
    if encode_fn is None:
        from sentence_transformers import SentenceTransformer
        _fallback = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        encode_fn = lambda sentences: _fallback.encode(sentences, convert_to_numpy=True)

    chunks: list[dict] = []
    chunk_index = 0

    for page in pages:
        raw_text: str = page.get("text", "").strip()
        if not raw_text:
            continue

        sentences = _split_sentences(raw_text)
        if not sentences:
            continue

        base = {
            "source":    page["source"],
            "file_path": page.get("file_path", ""),
            "page_num":  page["page_num"],
            "org_id":    page.get("org_id", "default"),
            "doc_id":    page.get("doc_id", ""),
        }

        if len(sentences) == 1:
            chunks.append({**base, "text": sentences[0], "chunk_index": chunk_index})
            chunk_index += 1
            continue

        embeddings = encode_fn(sentences)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)          # avoid div-by-zero
        embeddings = embeddings / norms

        similarities = (embeddings[:-1] * embeddings[1:]).sum(axis=1)  # vectorised

        current: list[str] = [sentences[0]]

        for i, sim in enumerate(similarities):
            next_sentence = sentences[i + 1]
            if sim < similarity_threshold:
                chunks.append({
                    **base,
                    "text":        " ".join(current),
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
                current = []

            current.append(next_sentence)

        if current:
            chunks.append({
                **base,
                "text":        " ".join(current),
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    return chunks