"""Сборка итоговых артефактов: report.md и trace.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .config import settings
from .agent import PaperResult, Trace
from .pdf_extract import Paper

_EMOJI = {"подходит": "✅", "спорно": "🟡", "не подходит": "❌"}


def write_report(results: List[PaperResult], failed: List[Paper], dup_notes: List[str],
                 thesis_text: str, trace: Trace) -> str:
    out_dir = Path(settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    providers = ", ".join(p.name for p in settings.llm_providers)
    mode = "DRY-RUN (LLM не вызывался)" if not settings.has_llm else f"LLM (фолбэк: {providers})"
    embed_mode = "Jina" if settings.has_embeddings else "hashed-fallback (без API)"

    lines: List[str] = []
    lines.append("# Отчёт триажа статей под тему диплома\n")
    lines.append(f"- Режим: **{mode}**, эмбеддинги: **{embed_mode}**, "
                 f"гибрид(BM25): **{'вкл' if settings.hybrid else 'выкл'}**")
    lines.append(f"- Статей оценено: **{len(results)}**, "
                 f"проблемных файлов: **{len(failed)}**, дублей: **{len(dup_notes)}**\n")

    lines.append("## Результаты\n")
    lines.append("| Статья | Вердикт | Близость | Уверенность | Почему |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        why = (r.reason or "").replace("|", "\\|")[:160]
        lines.append(f"| {r.name} | {_EMOJI.get(r.verdict,'')} {r.verdict} | "
                     f"{r.triage_score} | {r.confidence} | {why} |")
    lines.append("")

    lines.append("## Детали по статьям\n")
    for r in results:
        lines.append(f"### {_EMOJI.get(r.verdict,'')} {r.name} — {r.verdict}")
        lines.append(f"- Близость (триаж): {r.triage_score}; уверенность: {r.confidence}; "
                     f"LLM: {'да' if r.llm_used else 'нет'}; итераций агента: {r.iters}")
        if r.reason:
            lines.append(f"- Почему: {r.reason}")
        if r.missing:
            lines.append(f"- Что смущает / чего не хватает: {r.missing}")
        if r.self_check:
            lines.append(f"- Самопроверка: {r.self_check}")
        if r.evidence:
            lines.append("- Источники (цитаты):")
            for e in r.evidence:
                lines.append(f"  - *{r.name}, стр. {e.page}* (score {round(e.score,4)}): "
                             f"{e.snippet[:160].strip()}…")
        lines.append("")

    if failed:
        lines.append("## Пропущенные файлы\n")
        for p in failed:
            lines.append(f"- {p.name}: {p.error}")
        lines.append("")
    if dup_notes:
        lines.append("## Дубли (проиндексированы один раз)\n")
        for d in dup_notes:
            lines.append(f"- {d}")
        lines.append("")

    lines.append("## Limitations\n")
    lines.append("- Извлекается только текстовый слой PDF; сканы без текста пропускаются (нужен OCR).")
    lines.append("- Решение зависит от качества эмбеддингов и формулировки `thesis.md`.")
    lines.append("- В dry-run вердикты эвристические (относительная близость), не семантические.")
    lines.append("- LLM может ошибаться; самопроверка снижает, но не исключает галлюцинации.")
    lines.append("- Корпус на 5–10 статей; для тысяч документов нужен серверный Qdrant и батчинг.\n")

    lines.append("## Next steps\n")
    lines.append("- Разбивка вердикта по каждому исследовательскому вопросу диплома.")
    lines.append("- Гибрид (BM25) по умолчанию + reranker; OCR для сканов.")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    (out_dir / "trace.json").write_text(
        json.dumps({"events": trace.events}, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(report_path)
