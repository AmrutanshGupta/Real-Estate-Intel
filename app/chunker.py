import re
from transformers import AutoTokenizer
from app.config import Config

# Force the fast Rust-based tokenizer for significant CPU speedups
_tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=True)

def _split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = []
    for s in sentences:
        result.extend([p.strip() for p in s.split('\n') if p.strip()])
    return [s for s in result if len(s) > 10]

def chunk_text(pages):
    chunks = []

    for page in pages:
        sentences = _split_sentences(page["text"])
        if not sentences:
            continue

        # Pre-tokenize to prevent redundant CPU cycles during overlap and chunk creation
        encoded_sents = [
            (sent, _tokenizer.encode(sent, add_special_tokens=False)) 
            for sent in sentences
        ]

        current_sents = []
        current_tokens = 0
        chunk_index = 0

        for sent, tokens in encoded_sents:
            sent_tokens = len(tokens)

            if sent_tokens > Config.CHUNK_SIZE:
                if current_sents:
                    chunks.append(_make_chunk([s[0] for s in current_sents], page, chunk_index, current_tokens))
                    chunk_index += 1
                    current_sents = []
                    current_tokens = 0

                for start in range(0, sent_tokens, Config.CHUNK_SIZE - Config.CHUNK_OVERLAP):
                    end = min(start + Config.CHUNK_SIZE, sent_tokens)
                    text = _tokenizer.decode(tokens[start:end])
                    chunks.append({
                        "text": text,
                        "source": page["source"],
                        "file_path": page.get("file_path", ""),
                        "page_num": page["page_num"],
                        "chunk_index": chunk_index,
                        "token_count": end - start,
                    })
                    chunk_index += 1
                continue

            if current_tokens + sent_tokens > Config.CHUNK_SIZE and current_sents:
                chunks.append(_make_chunk([s[0] for s in current_sents], page, chunk_index, current_tokens))
                chunk_index += 1
                
                # Carry overlap context without re-tokenizing
                current_sents = current_sents[-2:]
                current_tokens = sum(s[1] for s in current_sents)

            current_sents.append((sent, sent_tokens))
            current_tokens += sent_tokens

        if current_sents:
            chunks.append(_make_chunk([s[0] for s in current_sents], page, chunk_index, current_tokens))

    return chunks

def _make_chunk(sentences, page, index, token_count):
    text = " ".join(sentences)
    return {
        "text": text,
        "source": page["source"],
        "file_path": page.get("file_path", ""),
        "page_num": page["page_num"],
        "chunk_index": index,
        "token_count": token_count,
    }