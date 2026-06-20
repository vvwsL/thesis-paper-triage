"""Конфигурация из .env. Паттерн _env/_int/_bool адаптирован из оригинала (rag_api/config.py)."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None or v == "" else v


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    v = _env(name, "")
    if not v:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Provider:
    """Один OpenAI-совместимый LLM-провайдер для фолбэка."""
    name: str
    url: str
    api_key: str
    model: str


# Реестр провайдеров: имя -> (url, env-ключа, env-модели, модель по умолчанию).
# Все эндпоинты OpenAI-совместимые (у Gemini и Cohere — compat-режим).
_PROVIDER_REGISTRY = {
    "cerebras":   ("https://api.cerebras.ai/v1/chat/completions",
                   "CEREBRAS_API_KEY", "CEREBRAS_MODEL", "llama-3.3-70b"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions",
                   "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
    "mistral":    ("https://api.mistral.ai/v1/chat/completions",
                   "MISTRAL_API_KEY", "MISTRAL_MODEL", "mistral-small-latest"),
    "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                   "GEMINI_API_KEY", "GEMINI_MODEL", "gemini-2.0-flash"),
    "cohere":     ("https://api.cohere.ai/compatibility/v1/chat/completions",
                   "COHERE_API_KEY", "COHERE_MODEL", "command-r-08-2024"),
}


def build_providers() -> tuple[Provider, ...]:
    """Строит упорядоченный список доступных провайдеров (у кого есть ключ).
    Порядок задаётся LLM_PROVIDERS (через запятую)."""
    order = [p.strip().lower() for p in
             _env("LLM_PROVIDERS", "cerebras,openrouter,mistral,gemini,cohere").split(",") if p.strip()]
    out = []
    for name in order:
        reg = _PROVIDER_REGISTRY.get(name)
        if not reg:
            continue
        url, key_env, model_env, default_model = reg
        key = _env(key_env)
        if key:
            out.append(Provider(name=name, url=url, api_key=key,
                                 model=_env(model_env, default_model)))
    return tuple(out)


@dataclass(frozen=True)
class Settings:
    jina_api_key: str

    embed_model: str
    embed_dim: int
    embed_url: str

    llm_providers: tuple

    chunk_max_chars: int
    chunk_overlap_chars: int
    max_chunks_per_paper: int
    top_k_papers: int
    search_limit: int
    agent_max_iters: int

    # ── контроль нагрузки на LLM (против 429) ──
    llm_min_interval: float   # мин. секунд между LLM-вызовами (throttle)
    llm_max_retries: int      # попыток при 429/ошибке
    llm_backoff_base: float   # база экспоненциального backoff
    agent_loop: bool          # включать ли агентный цикл "хватает ли данных?" (доп. вызовы)
    self_verify: bool         # включать ли шаг самопроверки (доп. вызов)

    hybrid: bool
    dry_run: bool

    papers_dir: str
    qdrant_path: str
    qdrant_collection: str
    output_dir: str

    @property
    def has_embeddings(self) -> bool:
        return bool(self.jina_api_key) and not self.dry_run

    @property
    def has_llm(self) -> bool:
        return bool(self.llm_providers) and not self.dry_run


def load_settings() -> Settings:
    return Settings(
        jina_api_key=_env("JINA_API_KEY"),
        embed_model=_env("EMBED_MODEL", "jina-embeddings-v3"),
        embed_dim=_int("EMBED_DIM", 1024),
        embed_url=_env("EMBED_URL", "https://api.jina.ai/v1/embeddings"),
        llm_providers=build_providers(),
        chunk_max_chars=_int("CHUNK_MAX_CHARS", 900),
        chunk_overlap_chars=_int("CHUNK_OVERLAP_CHARS", 120),
        max_chunks_per_paper=_int("MAX_CHUNKS_PER_PAPER", 0),   # 0 = без лимита (вся статья)
        top_k_papers=_int("TOP_K_PAPERS", 6),
        search_limit=_int("SEARCH_LIMIT", 8),
        agent_max_iters=_int("AGENT_MAX_ITERS", 2),
        llm_min_interval=_float("LLM_MIN_INTERVAL", 1.0),
        llm_max_retries=_int("LLM_MAX_RETRIES", 2),
        llm_backoff_base=_float("LLM_BACKOFF_BASE", 1.6),
        agent_loop=_bool("AGENT_LOOP", False),   # False = быстрый батч-вердикт; True = поштучный цикл
        self_verify=_bool("SELF_VERIFY", True),
        hybrid=_bool("HYBRID", False),
        dry_run=_bool("DRY_RUN", False),
        papers_dir=_env("PAPERS_DIR", "papers"),
        qdrant_path=_env("QDRANT_PATH", "qdrant_db"),
        qdrant_collection=_env("QDRANT_COLLECTION", "thesis_papers"),
        output_dir=_env("OUTPUT_DIR", "output"),
    )


settings = load_settings()
