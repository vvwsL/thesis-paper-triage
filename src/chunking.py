"""Нарезка текста на чанки с overlap. Адаптировано из оригинала (rag_api/chunking.py)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .pdf_extract import Paper


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_text(text: str, max_chars: int, overlap: int) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(n, start + max_chars)
        if end < n:
            last_space = text.rfind(" ", start, end)
            if last_space > start + int(max_chars * 0.6):
                end = last_space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        nxt = end - overlap if overlap > 0 else end
        start = nxt if nxt > start else end
    return chunks


@dataclass
class Chunk:
    paper_name: str
    paper_path: str
    page: int
    text: str


def chunk_paper(paper: Paper, max_chars: int, overlap: int, max_chunks: int) -> List[Chunk]:
    """Режет каждую страницу отдельно, чтобы сохранить номер страницы для цитат.
    max_chunks<=0 -> без лимита (индексируем статью целиком)."""
    out: List[Chunk] = []
    for page_no, page_text in paper.pages:
        for piece in split_text(page_text, max_chars, overlap):
            out.append(Chunk(paper.name, paper.path, page_no, piece))
            if 0 < max_chunks <= len(out):   # лимит только если задан (>0)
                return out
    return out
