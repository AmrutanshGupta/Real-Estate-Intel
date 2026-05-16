import re
from transformers import AutoTokenizer

from app.config import Config


# Force the fast Rust-based tokenizer for significant CPU speedups
_tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=True)


# ── Sentence splitter ──────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = []
    for s in sentences:
        result.extend([p.strip() for p in s.split('\n') if p.strip()])
    return [s for s in result if len(s) > 10]


# ── Chunk builder ──────────────────────────────────────────────────────────────

def _make_chunk(
    sentences: list[str],
    page: dict,
    index: int,
    token_count: int,
) -> dict:
    return {
        "text":        " ".join(sentences),
        "source":      page["source"],
        "file_path":   page.get("file_path", ""),
        "page_num":    page["page_num"],
        "chunk_index": index,
        "token_count": token_count,
        # ── Multi-tenant fields propagated from pdf_loader ──
        "org_id":      page.get("org_id", "default"),
        "doc_id":      page.get("doc_id", ""),
    }


# ── Main chunking function ─────────────────────────────────────────────────────

def chunk_text(pages: list[dict]) -> list[dict]:
    chunks = []

    for page in pages:
        sentences = _split_sentences(page["text"])
        if not sentences:
            continue

        # Pre-tokenize all sentences once to avoid redundant CPU cycles
        encoded_sents = [
            (sent, _tokenizer.encode(sent, add_special_tokens=False))
            for sent in sentences
        ]

        current_sents  = []
        current_tokens = 0
        chunk_index    = 0

        for sent, tokens in encoded_sents:
            sent_tokens = len(tokens)

            # Handle sentences that exceed CHUNK_SIZE on their own
            if sent_tokens > Config.CHUNK_SIZE:
                if current_sents:
                    chunks.append(
                        _make_chunk(
                            [s[0] for s in current_sents],
                            page,
                            chunk_index,
                            current_tokens,
                        )
                    )
                    chunk_index   += 1
                    current_sents  = []
                    current_tokens = 0

                # Slide a window over the oversized sentence
                for start in range(0, sent_tokens, Config.CHUNK_SIZE - Config.CHUNK_OVERLAP):
                    end  = min(start + Config.CHUNK_SIZE, sent_tokens)
                    text = _tokenizer.decode(tokens[start:end])
                    chunks.append({
                        "text":        text,
                        "source":      page["source"],
                        "file_path":   page.get("file_path", ""),
                        "page_num":    page["page_num"],
                        "chunk_index": chunk_index,
                        "token_count": end - start,
                        "org_id":      page.get("org_id", "default"),
                        "doc_id":      page.get("doc_id", ""),
                    })
                    chunk_index += 1
                continue

            # Flush chunk when adding this sentence would overflow
            if current_tokens + sent_tokens > Config.CHUNK_SIZE and current_sents:
                chunks.append(
                    _make_chunk(
                        [s[0] for s in current_sents],
                        page,
                        chunk_index,
                        current_tokens,
                    )
                )
                chunk_index += 1

                # Carry the last 2 sentences as overlap context (no re-tokenizing)
                current_sents  = current_sents[-2:]
                current_tokens = sum(s[1] for s in current_sents)

            current_sents.append((sent, sent_tokens))
            current_tokens += sent_tokens

        # Flush any remaining sentences
        if current_sents:
            chunks.append(
                _make_chunk(
                    [s[0] for s in current_sents],
                    page,
                    chunk_index,
                    current_tokens,
                )
            )

    return chunks