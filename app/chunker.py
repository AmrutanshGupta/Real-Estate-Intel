import re
from transformers import AutoTokenizer
from app.config import Config

_tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)


def _split_sentences(text):
    """Split on sentence boundaries, keeping sentences intact."""
    # Split after . ! ? followed by whitespace or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Also split on newlines that look like section breaks
    result = []
    for s in sentences:
        parts = [p.strip() for p in s.split('\n') if p.strip()]
        result.extend(parts)
    return [s for s in result if len(s) > 10]


def chunk_text(pages):
    """
    Sentence-aware chunking — never cuts mid-sentence.
    
    Accumulates sentences until the token budget is hit, then starts
    a new chunk with a sentence-level overlap (last 2 sentences carried over).
    This gives the embedding model complete, coherent units to work with.
    """
    chunks = []

    for page in pages:
        sentences   = _split_sentences(page["text"])
        if not sentences:
            continue

        current_sents  = []
        current_tokens = 0
        chunk_index    = 0

        for sent in sentences:
            sent_tokens = len(_tokenizer.encode(sent, add_special_tokens=False))

            # If single sentence exceeds budget, hard-split it (rare but happens)
            if sent_tokens > Config.CHUNK_SIZE:
                # Flush what we have first
                if current_sents:
                    chunks.append(_make_chunk(current_sents, page, chunk_index))
                    chunk_index += 1
                    current_sents  = []
                    current_tokens = 0

                # Hard-split the long sentence by tokens
                tokens = _tokenizer.encode(sent, add_special_tokens=False)
                for start in range(0, len(tokens), Config.CHUNK_SIZE - Config.CHUNK_OVERLAP):
                    end   = min(start + Config.CHUNK_SIZE, len(tokens))
                    text  = _tokenizer.decode(tokens[start:end])
                    chunks.append({
                        "text":        text,
                        "source":      page["source"],
                        "file_path":   page.get("file_path", ""),
                        "page_num":    page["page_num"],
                        "chunk_index": chunk_index,
                        "token_count": end - start,
                    })
                    chunk_index += 1
                    if end >= len(tokens):
                        break
                continue

            # Would this sentence push us over budget?
            if current_tokens + sent_tokens > Config.CHUNK_SIZE and current_sents:
                chunks.append(_make_chunk(current_sents, page, chunk_index))
                chunk_index += 1
                # Carry last 2 sentences as overlap context
                current_sents  = current_sents[-2:]
                current_tokens = sum(
                    len(_tokenizer.encode(s, add_special_tokens=False))
                    for s in current_sents
                )

            current_sents.append(sent)
            current_tokens += sent_tokens

        # Flush remainder
        if current_sents:
            chunks.append(_make_chunk(current_sents, page, chunk_index))

    return chunks


def _make_chunk(sentences, page, index):
    text = " ".join(sentences)
    return {
        "text":        text,
        "source":      page["source"],
        "file_path":   page.get("file_path", ""),
        "page_num":    page["page_num"],
        "chunk_index": index,
        "token_count": len(_tokenizer.encode(text, add_special_tokens=False)),
    }