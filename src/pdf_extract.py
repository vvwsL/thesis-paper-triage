"""Извлечение текста из входных файлов.

PDF -> текст по страницам через PyMuPDF (текстовый слой, без OCR).
.txt/.md -> читаем как одну "страницу" (удобно для тестов без реальных PDF).

Каждый источник возвращается как список страниц [(page_no, text)]. Скан/пустой PDF
(нет текстового слоя) даёт пустой список -> вызывающий код помечает его и пропускает.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

SUPPORTED = {".pdf", ".txt", ".md"}


@dataclass
class Paper:
    path: str
    name: str
    pages: List[Tuple[int, str]]   # [(page_no, text), ...]
    content_hash: str
    error: str = ""

    @property
    def full_text(self) -> str:
        return "\n".join(t for _, t in self.pages)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _extract_pdf(path: Path) -> List[Tuple[int, str]]:
    import fitz  # PyMuPDF; импорт внутри, чтобы .txt-режим работал без установленного pymupdf

    pages: List[Tuple[int, str]] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = (page.get_text() or "").strip()
            if text:
                pages.append((i, text))
    return pages


def _extract_text_file(path: Path) -> List[Tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [(1, text)] if text else []


def extract_paper(path: str | Path) -> Paper:
    p = Path(path)
    try:
        if p.suffix.lower() == ".pdf":
            pages = _extract_pdf(p)
        else:
            pages = _extract_text_file(p)
    except Exception as exc:  # битый файл / ошибка парсинга
        return Paper(str(p), p.name, [], "", error=f"extract_failed: {exc}")

    if not pages:
        return Paper(str(p), p.name, [], "", error="no_text_layer (возможно скан или пустой файл)")

    return Paper(str(p), p.name, pages, _hash("\n".join(t for _, t in pages)))


def load_papers(papers_dir: str | Path) -> Tuple[List[Paper], List[Paper], List[str]]:
    """Возвращает (валидные_уникальные, проблемные, сообщения_о_дублях)."""
    d = Path(papers_dir)
    files = sorted(f for f in d.glob("*") if f.suffix.lower() in SUPPORTED) if d.exists() else []

    valid: List[Paper] = []
    failed: List[Paper] = []
    dup_notes: List[str] = []
    seen_hashes: dict[str, str] = {}

    for f in files:
        paper = extract_paper(f)
        if paper.error:
            failed.append(paper)
            continue
        if paper.content_hash in seen_hashes:
            dup_notes.append(f"{paper.name}: дубль (контент совпадает с {seen_hashes[paper.content_hash]})")
            continue
        seen_hashes[paper.content_hash] = paper.name
        valid.append(paper)

    return valid, failed, dup_notes
