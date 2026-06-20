"""Тонкая обёртка над questionary — единый вид/поведение меню для разных проектов.

Без своей рамки и raw-режима: всё рисует questionary. Есть fallback без TTY
(пайп/тесты/CI), чтобы не падать и не виснуть. Семантика value (label<->value).
"""
from __future__ import annotations

import sys
from typing import List, Optional, Sequence, Tuple

import questionary
from questionary import Choice, Style

# единый стиль (один на оба проекта)
STYLE = Style([
    ("qmark",       "fg:#00afff bold"),
    ("question",    "bold"),
    ("pointer",     "fg:#00afff bold"),
    ("highlighted", "fg:#00afff bold"),
    ("selected",    "fg:#5fd700"),
])
POINTER = "►"

# подсказки по управлению (единые во всех проектах)
INSTRUCTION = "(↑/↓ — выбрать · Enter — ок · Ctrl+C — выход)"
INSTRUCTION_MULTI = "(↑/↓ · Пробел — отметить · Enter — ок · Ctrl+C — выход)"

# кириллица в Windows-консоли
for _s in (sys.stdout, sys.stdin):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def _interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def _normalize(choices: Sequence) -> List[Tuple[str, object]]:
    """Приводит choices к списку (label, value). Принимает Choice или строку."""
    out: List[Tuple[str, object]] = []
    for c in choices:
        if isinstance(c, Choice):
            out.append((c.title if isinstance(c.title, str) else str(c.title), c.value))
        else:
            out.append((str(c), c))
    return out


# ---------- select ----------

def select(title: str, choices: Sequence, default=None) -> Optional[object]:
    """Одиночный выбор стрелками. Возвращает value выбранного или None.
    default — value текущего пункта (курсор откроется на нём)."""
    if not _interactive():
        return _fallback_select(title, choices, default)
    try:
        return questionary.select(title, choices=list(choices), default=default,
                                  pointer=POINTER, style=STYLE, qmark="?",
                                  instruction=INSTRUCTION).ask()
    except ValueError:   # default не совпал с choices — без него
        return questionary.select(title, choices=list(choices),
                                  pointer=POINTER, style=STYLE, qmark="?",
                                  instruction=INSTRUCTION).ask()


def _fallback_select(title, choices, default) -> Optional[object]:
    norm = _normalize(choices)
    print(title)
    for i, (label, _v) in enumerate(norm):
        print(f"  {i + 1}. {label}")
    try:
        raw = input("Выбор (номер, пусто=назад): ").strip()
    except EOFError:
        return None
    if not raw:
        return None
    try:
        idx = int(raw) - 1
        return norm[idx][1] if 0 <= idx < len(norm) else None
    except ValueError:
        return None


# ---------- multiselect ----------

def multiselect(title: str, choices: Sequence, default: Optional[List] = None,
                min_select: int = 1) -> List:
    """Мультивыбор галочками. Гарантирует «выбрано >= min_select»."""
    default = list(default or [])
    norm = _normalize(choices)
    if not _interactive():
        return _fallback_multiselect(title, norm, default, min_select)
    while True:
        qchoices = [Choice(title=lbl, value=val, checked=(val in default)) for lbl, val in norm]
        ans = questionary.checkbox(title, choices=qchoices, pointer=POINTER, style=STYLE,
                                   instruction=INSTRUCTION_MULTI).ask()
        if ans is None:               # Ctrl+C / отмена — оставляем как было
            return default
        if len(ans) >= min_select:
            return ans
        questionary.print(f"  Нужно выбрать минимум {min_select}.", style="fg:#ff5f5f")


def _fallback_multiselect(title, norm, default, min_select) -> List:
    print(title)
    for i, (label, val) in enumerate(norm):
        print(f"  {i + 1}. [{'x' if val in default else ' '}] {label}")
    try:
        raw = input("Номера через запятую (пусто=как есть): ").strip()
    except EOFError:
        return default
    if not raw:
        return default
    try:
        picked = {int(x) - 1 for x in raw.split(",")}
        vals = [norm[i][1] for i in picked if 0 <= i < len(norm)]
        return vals if len(vals) >= min_select else default
    except ValueError:
        return default


# ---------- ввод / подтверждение ----------

def ask_text(title: str, default: str = "") -> str:
    """Ввод строки. questionary сам управляет строкой ввода (хвоста не остаётся)."""
    if not _interactive():
        try:
            v = input(f"{title} [{default}]: ").strip()
        except EOFError:
            return default
        return v or default
    v = questionary.text(title, default=default, style=STYLE, qmark="?").ask()
    return (v.strip() or default) if isinstance(v, str) else default


def confirm(title: str, default: bool = False) -> bool:
    if not _interactive():
        try:
            v = input(f"{title} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        except EOFError:
            return default
        return default if not v else v in {"y", "yes", "д", "да"}
    a = questionary.confirm(title, default=default, style=STYLE, qmark="?").ask()
    return bool(a) if a is not None else default
