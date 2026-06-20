"""Интерактивное меню на questionary (через тонкую обёртку src.ui).

Меню — тонкая обёртка: собирает параметры и зовёт пайплайн как подпроцесс (src.cli).
Состояние (тема, гибрид, быстрый режим, провайдеры) — в .menu_state.json (gitignore).

Запуск:  python -m src.menu   (или двойной клик по run.bat)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import ui
from .ui import Choice

ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / ".menu_state.json"

# провайдеры: внутренний код -> человекочитаемая подпись (label<->value)
PROVIDERS = [("cerebras", "Cerebras (быстрый)"), ("openrouter", "OpenRouter"),
             ("mistral", "Mistral"), ("gemini", "Gemini"), ("cohere", "Cohere")]
KEY_BY_PROV = {"cerebras": "CEREBRAS_API_KEY", "openrouter": "OPENROUTER_API_KEY",
               "mistral": "MISTRAL_API_KEY", "gemini": "GEMINI_API_KEY", "cohere": "COHERE_API_KEY"}


def log(msg: str) -> None:
    """Единый формат вывода прогона: 'HH:MM:SS | сообщение'."""
    print(f"{datetime.now():%H:%M:%S} | {msg}")


def _enable_ansi() -> None:
    """Включить обработку ANSI в консоли Windows (для очистки экрана на месте)."""
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            pass


def clear() -> None:
    """Очистка «окно на месте» (без накопления прокрутки)."""
    try:
        if sys.stdout.isatty():
            sys.stdout.write("\x1b[H\x1b[J")
            sys.stdout.flush()
            return
    except Exception:
        pass
    os.system("cls" if os.name == "nt" else "clear")


# ---------- состояние ----------

class State:
    def __init__(self) -> None:
        d = {}
        if STATE_FILE.exists():
            try:
                d = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            except Exception:
                d = {}
        self.thesis = d.get("thesis", "thesis.md")
        self.hybrid = d.get("hybrid", False)
        self.full_agency = d.get("full_agency", False)   # True = поштучный цикл; False = батч
        self.providers = d.get("providers", [])   # коды; пусто = автопорядок из .env

    def save(self) -> None:
        STATE_FILE.write_text(json.dumps(
            {"thesis": self.thesis, "hybrid": self.hybrid, "full_agency": self.full_agency,
             "providers": self.providers}, ensure_ascii=False, indent=2), encoding="utf-8")


def _env_file_get(key: str, default: str = "") -> str:
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith(f"{key}=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip()
    return os.environ.get(key, default)


def _has_key(key: str) -> bool:
    return bool(_env_file_get(key))


def _available_providers() -> list[str]:
    return [name for name, _ in PROVIDERS if _has_key(KEY_BY_PROV[name])]


# ---------- проверка индекса ----------

def _run_env(state: State) -> dict:
    env = dict(os.environ)
    env["HYBRID"] = "true" if state.hybrid else "false"
    # режим агента задаётся прямо из меню (в .env лезть не нужно)
    env["AGENT_LOOP"] = "true" if state.full_agency else "false"
    env["SELF_VERIFY"] = "true" if state.full_agency else "false"
    if state.providers:
        env["LLM_PROVIDERS"] = ",".join(state.providers)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _index_check(state: State) -> dict:
    result = {"summary": "unknown", "files": [], "removed": [], "rebuild_all": False}
    try:
        r = subprocess.run([sys.executable, "-m", "src.cli", "--check-index"],
                           cwd=ROOT, env=_run_env(state), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=120)
        out = (r.stdout or "") + (r.stderr or "")
        for line in out.splitlines():
            if line.startswith("INDEX_JSON:"):
                result = json.loads(line[len("INDEX_JSON:"):])
                break
        if not result.get("files") and "INDEX:EMPTY" in out:
            result["summary"] = "EMPTY"
    except Exception:
        pass
    return result


def _index_status(state: State) -> str:
    s = _index_check(state).get("summary", "unknown")
    return {"FRESH": "fresh", "STALE": "stale", "EMPTY": "empty"}.get(s, "unknown")


_ACTION_LABEL = {
    "unchanged": "[готов]",
    "new": "[новый, будет проиндексирован]",
    "changed": "[изменён, будет переиндексирован]",
    "indexed": "[будет проиндексирован]",
    "skipped": "[пропуск: скан/без текста]",
}


# ---------- запуск ----------

def _run_cli(state: State, *, dry_run: bool, rebuild: bool) -> None:
    args = [sys.executable, "-m", "src.cli", "--thesis", state.thesis]
    if dry_run:
        args.append("--dry-run")
    if rebuild:
        args.append("--rebuild")
    clear()
    agent_mode = "полная агентность" if state.full_agency else "батч"
    log(f"=== Запуск === {' '.join(args[2:])} (hybrid={state.hybrid}, агент={agent_mode})")
    subprocess.run(args, cwd=ROOT, env=_run_env(state), encoding="utf-8")


def _pause() -> None:
    try:
        input("\n  [Enter] назад ")
    except (EOFError, KeyboardInterrupt):
        pass


# ---------- экраны ----------

def _screen_status(state: State) -> None:
    clear()
    papers = [p for p in (ROOT / "papers").glob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"}]
    avail = state.providers or _available_providers()
    log("=== Статус ===")
    print(f"  Тема (thesis):     {state.thesis}")
    print(f"  Статей в papers/:  {len(papers)}")
    print(f"  JINA_API_KEY:      {'есть' if _has_key('JINA_API_KEY') else 'нет -> dry-режим эмбеддингов'}")
    print(f"  LLM-провайдеры:    {', '.join(avail) if avail else 'нет ключей -> dry-режим'}")
    print(f"  Гибрид (BM25):     {'вкл' if state.hybrid else 'выкл'}")
    print(f"  Режим агента:      {'полная агентность (медленно)' if state.full_agency else 'батч (быстро)'}")
    print(f"  Индекс:            {_index_status(state)}")
    _pause()


def _screen_index_files(state: State) -> None:
    clear()
    data = _index_check(state)
    files = data.get("files", [])
    log("=== Файлы в индексе ===")
    if not files:
        print("  В papers/ нет статей для индексации.")
    for f in files:
        label = _ACTION_LABEL.get(f["action"], f["action"])
        extra = f"  ({f['chunks']} чанков)" if f.get("chunks") else ""
        name = f["name"] if len(f["name"]) <= 50 else f["name"][:47] + "..."
        print(f"  {name:<52} {label}{extra}")
    if data.get("removed"):
        print("  Удалены из папки (будут вычищены):", ", ".join(data["removed"]))
    if data.get("rebuild_all"):
        print("  ВНИМАНИЕ: сменился конфиг -> нужен полный пересбор (пункт «Полный пересбор»).")
    _pause()


def _action_run(state: State) -> None:
    st = _index_status(state)
    if st == "empty":
        log("Нет статей в papers/."); _pause(); return
    if st != "fresh":
        if not ui.confirm("Есть непроиндексированные/изменённые статьи. Доиндексировать и запустить?", True):
            return
    _run_cli(state, dry_run=False, rebuild=False)   # sync доиндексирует только нужное
    _pause()


def _choose_providers(state: State) -> None:
    avail = _available_providers()
    if not avail:
        log("Нет провайдеров с ключами (заполни .env)."); _pause(); return
    label_by = dict(PROVIDERS)
    choices = [Choice(title=label_by[name], value=name) for name in avail]
    cur = [p for p in (state.providers or avail) if p in avail] or [avail[0]]
    state.providers = ui.multiselect("Выбор LLM-провайдеров (фолбэк по порядку)",
                                     choices, default=cur, min_select=1)
    state.save()


def _open(path: Path) -> None:
    if not path.exists():
        log(f"Файл ещё не создан: {path.name} (сначала запусти триаж)."); _pause(); return
    try:
        os.startfile(path)  # type: ignore[attr-defined]
    except AttributeError:
        subprocess.run(["xdg-open" if sys.platform.startswith("linux") else "open", str(path)])
    log(f"Открыл: {path}")


def _run_tests() -> None:
    clear()
    log("=== Тесты ===")
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    _pause()


# ---------- главный цикл ----------

def main() -> int:
    _enable_ansi()
    state = State()
    out = ROOT / "output"
    cur = None
    while True:
        clear()
        avail = state.providers or _available_providers()
        choices = [
            Choice("Статус", "status"),
            Choice("Список файлов индекса", "index"),
            Choice("Запустить триаж", "run"),
            Choice("Полный пересбор + запуск", "rebuild"),
            Choice("Dry-run (без ключей)", "dry"),
            Choice("Открыть отчёт (report.md)", "report"),
            Choice("Открыть лог (run.log)", "log"),
            Choice(f"Тема: {state.thesis}", "thesis"),
            Choice(f"Гибрид (BM25): {'вкл' if state.hybrid else 'выкл'}", "hybrid"),
            Choice(f"Режим агента: {'полная агентность' if state.full_agency else 'батч (быстро)'}", "agent_mode"),
            Choice(f"LLM-провайдеры: {', '.join(avail) if avail else 'нет'}", "providers"),
            Choice("Тесты", "tests"),
            Choice("Выход", "exit"),
        ]
        try:
            cur = ui.select("Триаж научных PDF под тему диплома", choices, default=cur) or "exit"
        except KeyboardInterrupt:
            print(); break

        if cur == "exit":
            break
        try:
            if cur == "status":
                _screen_status(state)
            elif cur == "index":
                _screen_index_files(state)
            elif cur == "run":
                _action_run(state)
            elif cur == "rebuild":
                _run_cli(state, dry_run=False, rebuild=True); _pause()
            elif cur == "dry":
                _run_cli(state, dry_run=True, rebuild=False); _pause()
            elif cur == "report":
                _open(out / "report.md")
            elif cur == "log":
                _open(out / "run.log")
            elif cur == "thesis":
                state.thesis = ui.ask_text("Файл темы", state.thesis); state.save()
            elif cur == "hybrid":
                state.hybrid = not state.hybrid; state.save()      # тумблер на месте
            elif cur == "agent_mode":
                state.full_agency = not state.full_agency; state.save()   # тумблер на месте
            elif cur == "providers":
                _choose_providers(state)
            elif cur == "tests":
                _run_tests()
        except KeyboardInterrupt:
            continue   # отмена действия -> назад в меню

    print("Пока!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
