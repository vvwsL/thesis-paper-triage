"""Точка входа CLI.

Примеры:
  python -m src.cli                       # триаж по thesis.md
  python -m src.cli --dry-run             # без вызовов LLM/эмбеддингов
  python -m src.cli --rebuild             # пересобрать индекс принудительно
  python -m src.cli --thesis my.md
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


# --dry-run должен выставить DRY_RUN ДО построения настроек, чтобы все модули увидели режим.
# Делаем это здесь, до импорта .config (его settings строится при первом импорте).
if "--dry-run" in sys.argv[1:]:
    os.environ["DRY_RUN"] = "true"

from .config import settings           # noqa: E402
from .pdf_extract import load_papers   # noqa: E402
from .chunking import chunk_paper      # noqa: E402
from .store import Store               # noqa: E402
from .agent import run_agent, Trace    # noqa: E402
from .report import write_report       # noqa: E402


def _setup_logger() -> logging.Logger:
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("triage")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(Path(settings.output_dir) / "run.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main() -> int:
    ap = argparse.ArgumentParser(description="Агент-триаж научных PDF под тему диплома")
    ap.add_argument("--thesis", default="thesis.md", help="файл с темой диплома")
    ap.add_argument("--dry-run", action="store_true", help="без вызовов LLM/эмбеддингов")
    ap.add_argument("--rebuild", action="store_true", help="пересобрать индекс принудительно")
    ap.add_argument("--check-index", action="store_true",
                    help="только проверить статус индекса (FRESH/STALE/EMPTY) и выйти")
    args = ap.parse_args()

    # быстрый режим проверки индекса (для меню): без логов и без LLM.
    # Печатает per-file JSON: что проиндексировано, что новое/изменённое, что пропущено.
    if args.check_index:
        import json as _json
        valid, failed, dups = load_papers(settings.papers_dir)
        store = Store()
        d = store.diff(valid)
        files = [{"name": s.name, "action": s.action, "chunks": s.chunks} for s in d["statuses"]]
        for p in failed:
            files.append({"name": p.name, "action": "skipped", "chunks": 0, "reason": p.error})
        pending = any(s.action in ("new", "changed") for s in d["statuses"]) or d["rebuild_all"]
        summary = "STALE" if (pending or not valid) else "FRESH"
        print("INDEX_JSON:" + _json.dumps(
            {"summary": summary, "rebuild_all": d["rebuild_all"],
             "removed": d["removed"], "files": files}, ensure_ascii=False))
        print("INDEX:EMPTY" if not valid else ("INDEX:FRESH" if summary == "FRESH" else "INDEX:STALE"))
        return 0 if summary == "FRESH" and valid else (4 if not valid else 3)

    logger = _setup_logger()
    trace = Trace()
    providers = ",".join(p.name for p in settings.llm_providers) or "—"
    logger.info("=== Запуск ===  dry_run=%s hybrid=%s embeddings=%s llm_providers=[%s]",
                settings.dry_run, settings.hybrid, settings.has_embeddings, providers)

    # 1. тема
    thesis_path = Path(args.thesis)
    if not thesis_path.exists():
        logger.error("Не найден файл темы: %s", thesis_path)
        return 2
    thesis_text = thesis_path.read_text(encoding="utf-8")

    # 2. статьи (+дедуп, +проблемные)
    valid, failed, dup_notes = load_papers(settings.papers_dir)
    logger.info("Статей: валидных=%d, проблемных=%d, дублей=%d",
                len(valid), len(failed), len(dup_notes))
    trace.add("load", valid=[p.name for p in valid],
              failed=[(p.name, p.error) for p in failed], dups=dup_notes)

    if not valid:
        logger.error("Нет валидных статей в '%s'. Положи .pdf/.txt/.md и повтори.", settings.papers_dir)
        write_report([], failed, dup_notes, thesis_text, trace)
        return 1

    # 3. инкрементальная индексация: только новые/изменённые статьи
    store = Store()
    chunks_by_name = {p.name: chunk_paper(p, settings.chunk_max_chars,
                                          settings.chunk_overlap_chars,
                                          settings.max_chunks_per_paper) for p in valid}
    report = store.sync(valid, chunks_by_name, force=args.rebuild)
    for s in report:
        logger.info("  [index] %-10s %s (%d чанков)", s.action, s.name, s.chunks)
    n_new = sum(1 for s in report if s.action in ("new", "changed", "indexed"))
    logger.info("Индексация: новых/обновлённых=%d, без изменений=%d",
                n_new, sum(1 for s in report if s.action == "unchanged"))
    trace.add("index", actions=[(s.name, s.action, s.chunks) for s in report])

    # 4. агент
    results = run_agent(store, thesis_text, [p.name for p in valid], logger, trace)

    # 5. отчёт
    path = write_report(results, failed, dup_notes, thesis_text, trace)
    logger.info("=== Готово ===  отчёт: %s", path)

    print("\n--- ИТОГ ---")
    for r in results:
        print(f"  {r.verdict:>12} | {r.triage_score:>7} | {r.name}")
    print(f"\nОтчёт:  {path}\nЛог:    {Path(settings.output_dir) / 'run.log'}\n"
          f"Trace:  {Path(settings.output_dir) / 'trace.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
