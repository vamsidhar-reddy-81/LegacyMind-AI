from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*")


@dataclass
class Chunk:
    source: str
    text: str
    index: int


MAX_CHUNKS = 140

QUERY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "explain",
    "for",
    "give",
    "how",
    "in",
    "is",
    "me",
    "of",
    "on",
    "tell",
    "the",
    "to",
    "what",
    "whats",
    "with",
}


def chunk_documents(documents: list[tuple[str, str]], chunk_size: int = 1600, overlap: int = 80) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = max(1, chunk_size - overlap)
    for source, text in documents:
        cleaned = " ".join(text.split())
        if not cleaned:
            continue
        for start in range(0, len(cleaned), step):
            piece = cleaned[start : start + chunk_size].strip()
            if piece:
                chunks.append(Chunk(source=source, text=piece, index=len(chunks)))
            if len(chunks) >= MAX_CHUNKS:
                return chunks
            if start + chunk_size >= len(cleaned):
                break
    return chunks


def retrieve(query: str, chunks: list[Chunk], k: int = 5) -> list[Chunk]:
    if not chunks:
        return []

    query_terms = [token for token in _tokens(query) if token not in QUERY_STOP_WORDS]
    query_vector = Counter(query_terms or _tokens(query))
    if not query_vector:
        return chunks[:k]

    scored = []
    for chunk in chunks:
        chunk_tokens = _tokens(chunk.text)
        chunk_vector = Counter(chunk_tokens)
        score = _cosine(query_vector, chunk_vector)
        score += _exact_match_boost(query_terms, chunk.text)
        score -= _noise_penalty(chunk.text)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:k]] or chunks[:k]


def knowledge_map(chunks: list[Chunk], limit: int = 18) -> list[tuple[str, int]]:
    stop_words = {
        "about",
        "after",
        "also",
        "and",
        "are",
        "because",
        "but",
        "can",
        "for",
        "from",
        "have",
        "into",
        "not",
        "that",
        "the",
        "their",
        "this",
        "with",
        "would",
    }
    counts: Counter[str] = Counter()
    for chunk in chunks:
        counts.update(token for token in _tokens(chunk.text) if token not in stop_words and len(token) > 3)
    return counts.most_common(limit)


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _exact_match_boost(query_terms: list[str], text: str) -> float:
    if not query_terms:
        return 0.0
    lower = text.lower()
    boost = 0.0
    for term in query_terms:
        if term in lower:
            boost += 0.08
        if term.endswith("s") and term[:-1] in lower:
            boost += 0.05
    return boost


def _noise_penalty(text: str) -> float:
    lower = text.lower()
    penalty = 0.0
    if "contents" in lower or "disclaimer" in lower:
        penalty += 0.25
    if lower.count("chapter") > 8 or lower.count("section") > 10:
        penalty += 0.25
    if text.count(".") / max(len(text), 1) > 0.12:
        penalty += 0.2
    return penalty
