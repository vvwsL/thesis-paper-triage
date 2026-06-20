"""LLM через OpenRouter (OpenAI-совместимый chat).

Структура вызова адаптирована из оригинала (rag_api/clients/llm.py): requests + парсинг
choices[0].message.content.

Контроль нагрузки против 429 (rate limiting):
  - throttle: глобальный минимальный интервал между ЛЮБЫМИ LLM-вызовами (сериализация burst-а);
  - backoff: экспоненциальный с jitter, с уважением заголовка Retry-After на 429.
Вызовы в проекте последовательные, поэтому throttle = простая пауза от времени прошлого запроса.
"""
from __future__ import annotations

import json
import random
import threading
import time
from typing import Any, Dict, List

import requests

from .config import settings


class LLMError(RuntimeError):
    pass


# ---------- throttle (минимальный интервал между запросами) ----------

_last_call_ts = 0.0
_throttle_lock = threading.Lock()


def _throttle() -> None:
    global _last_call_ts
    with _throttle_lock:
        wait = settings.llm_min_interval - (time.time() - _last_call_ts)
        if wait > 0:
            time.sleep(wait)
        _last_call_ts = time.time()


def _retry_after(resp: requests.Response, attempt: int) -> float:
    """Сколько ждать перед повтором: Retry-After если есть, иначе экспонента + jitter."""
    ra = resp.headers.get("Retry-After") if resp is not None else None
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    return settings.llm_backoff_base ** attempt + random.uniform(0, 0.75)


# имя провайдера, ответившего на последний успешный вызов (для trace)
last_used_provider = ""


def _headers(provider) -> Dict[str, str]:
    h = {"Content-Type": "application/json",
         "Authorization": f"Bearer {provider.api_key}"}
    if provider.name == "openrouter":  # необязательная идентификация приложения
        h["HTTP-Referer"] = "https://localhost/thesis-paper-triage"
        h["X-Title"] = "thesis-paper-triage"
    return h


def _chat_one(provider, messages: List[Dict[str, str]], temperature: float) -> str:
    """Вызов одного провайдера с throttle + backoff. Бросает LLMError при исчерпании."""
    payload: Dict[str, Any] = {
        "model": provider.model, "messages": messages, "temperature": temperature,
    }
    retries = settings.llm_max_retries
    last = ""
    for attempt in range(retries + 1):
        _throttle()
        resp = None
        try:
            resp = requests.post(provider.url, json=payload,
                                 headers=_headers(provider), timeout=90)
            if resp.status_code == 429 or resp.status_code >= 500:
                last = f"HTTP {resp.status_code}"
                if attempt < retries:
                    time.sleep(_retry_after(resp, attempt))
                    continue
                raise LLMError(last)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                if isinstance(content, str) and content.strip():
                    return content
            last = f"пустой ответ: {json.dumps(data)[:150]}"
        except requests.RequestException as exc:
            last = f"сеть/HTTP: {exc}"
        if attempt < retries:
            time.sleep(_retry_after(resp, attempt))
    raise LLMError(last or "нет ответа")


def chat(messages: List[Dict[str, str]], *, temperature: float = 0.1) -> str:
    """Перебирает провайдеров по порядку; при 429/ошибке переходит к следующему."""
    global last_used_provider
    if not settings.has_llm:
        raise LLMError("LLM не настроен (нет ключей провайдеров или включён DRY_RUN)")

    errors: List[str] = []
    for provider in settings.llm_providers:
        try:
            result = _chat_one(provider, messages, temperature)
            last_used_provider = provider.name
            return result
        except LLMError as exc:
            errors.append(f"{provider.name}({exc})")
            continue
    raise LLMError("все провайдеры не ответили: " + " | ".join(errors))


def chat_json(messages: List[Dict[str, str]], **kw) -> Any:
    """Просит модель вернуть JSON; устойчиво парсит (вырезает ```json блоки)."""
    raw = chat(messages, **kw)
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)
