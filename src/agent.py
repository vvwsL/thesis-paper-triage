"""Агент-триаж научных статей под тему диплома.

Поток (см. план):
  plan   — LLM строит под-запросы/ключевые слова из thesis.md            (агентность #1)
  triage — код: близость статья↔тема по под-запросам, агрегируем score    (дёшево, без LLM)
  judge  — по топ-K статьям агентный цикл "хватает ли доказательств?":     (агентность #2)
           [нет -> дозапрос] -> вердикт -> самопроверка против цитат        (агентность #3)
  refusal— статьи ниже топа -> "не подходит" без траты LLM

В dry-run (нет ключей) LLM-шаги заменяются детерминированной логикой,
а промпты-решения фиксируются в trace как "что бы сделал агент".
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List

from .config import settings
from .store import Store, Hit
from . import llm

VERDICTS = ("подходит", "спорно", "не подходит")


# ---------- trace ----------

class Trace:
    def __init__(self) -> None:
        self.events: List[dict] = []
        self.t0 = time.time()

    def add(self, step: str, **data) -> None:
        self.events.append({"t": round(time.time() - self.t0, 3), "step": step, **data})

    def log(self, logger, step: str, msg: str) -> None:
        logger.info("[%s] %s", step, msg)
        self.add(step, msg=msg)


@dataclass
class Evidence:
    page: int
    snippet: str
    score: float


@dataclass
class PaperResult:
    name: str
    path: str
    triage_score: float
    verdict: str = "не подходит"
    confidence: str = "low"
    reason: str = ""
    missing: str = ""
    evidence: List[Evidence] = field(default_factory=list)
    llm_used: bool = False
    iters: int = 0
    self_check: str = ""


# ---------- helpers ----------

def parse_thesis(thesis_text: str) -> Dict[str, object]:
    """Эвристический разбор thesis.md для dry-run-плана (без LLM)."""
    topic = ""
    for line in thesis_text.splitlines():
        s = line.strip()
        if s and not s.startswith("#") and not s.startswith(">"):
            topic = s
            break
    questions = [re.sub(r"^\d+\.\s*", "", l.strip())
                 for l in thesis_text.splitlines() if l.strip().endswith("?")]
    kw_match = re.search(r"(?is)ключевые слова\s*\n(.+)", thesis_text)
    keywords: List[str] = []
    if kw_match:
        keywords = [w.strip() for w in re.split(r"[,\n]", kw_match.group(1)) if w.strip()]
    return {"topic": topic or thesis_text[:200], "questions": questions, "keywords": keywords}


def _clean_query(q: str) -> str:
    """Убирает гугл-операторы из refine-запроса (site:, OR/AND, кавычки, URL, годы),
    оставляя только осмысленные термины для семантического поиска."""
    q = re.sub(r"\bsite:\S+", " ", q or "")
    q = re.sub(r"https?://\S+", " ", q)
    q = re.sub(r"\b(OR|AND)\b", " ", q)
    q = re.sub(r"\b\d{4}(\.\.\d{4})?\b", " ", q)   # годы / диапазоны
    q = q.replace('"', " ").replace("'", " ")
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _aggregate(hits: List[Hit]) -> Dict[str, dict]:
    """Группирует хиты по статье: лучший score + накопленные доказательства."""
    by_paper: Dict[str, dict] = {}
    for h in hits:
        rec = by_paper.setdefault(h.paper_name, {"path": h.paper_path, "score": 0.0, "ev": {}})
        rec["score"] = max(rec["score"], h.score)
        # дедуп доказательств по странице, держим лучший фрагмент
        prev = rec["ev"].get(h.page)
        if prev is None or h.score > prev.score:
            rec["ev"][h.page] = Evidence(page=h.page, snippet=h.text[:300], score=h.score)
    return by_paper


# ---------- step 1: plan ----------

def plan_search(thesis_text: str, logger, trace: Trace) -> Dict[str, object]:
    parsed = parse_thesis(thesis_text)
    if not settings.has_llm:
        queries = [parsed["topic"]] + list(parsed["questions"])[:3]
        trace.add("plan", mode="dry-run", queries=queries, keywords=parsed["keywords"],
                  note="LLM не вызывался; план собран эвристикой из thesis.md")
        logger.info("[plan] dry-run: %d под-запросов из thesis.md", len(queries))
        return {"queries": [q for q in queries if q], "keywords": parsed["keywords"]}

    prompt = [
        {"role": "system", "content": "Ты помогаешь искать научные статьи под тему диплома. "
         "Верни строго JSON."},
        {"role": "user", "content":
            "Тема диплома и вопросы:\n" + thesis_text +
            "\n\nСформулируй 3-5 коротких поисковых под-запросов (на английском, т.к. статьи "
            "англоязычные) и список ключевых терминов. Ответ строго JSON: "
            '{"queries": ["..."], "keywords": ["..."]}'},
    ]
    try:
        data = llm.chat_json(prompt)
        queries = [q for q in data.get("queries", []) if q][:5] or [parsed["topic"]]
        keywords = data.get("keywords", []) or parsed["keywords"]
        trace.add("plan", mode="llm", provider=llm.last_used_provider,
                  queries=queries, keywords=keywords)
        logger.info("[plan] LLM построил %d под-запросов", len(queries))
        return {"queries": queries, "keywords": keywords}
    except Exception as exc:
        logger.warning("[plan] LLM-план не удался (%s), fallback на эвристику", exc)
        trace.add("plan", mode="fallback", error=str(exc))
        return {"queries": [parsed["topic"]] + list(parsed["questions"])[:3],
                "keywords": parsed["keywords"]}


# ---------- step 2: triage ----------

def triage(store: Store, queries: List[str], logger, trace: Trace) -> Dict[str, dict]:
    all_hits: List[Hit] = []
    for q in queries:
        hits = store.search(q, limit=settings.search_limit)
        all_hits.extend(hits)
        trace.add("triage_search", query=q, hits=len(hits),
                  top_score=round(hits[0].score, 4) if hits else None)
    agg = _aggregate(all_hits)
    ranked = sorted(agg.items(), key=lambda kv: kv[1]["score"], reverse=True)
    logger.info("[triage] статей с попаданиями: %d", len(ranked))
    trace.add("triage", ranked=[(n, round(r["score"], 4)) for n, r in ranked])
    return agg


# ---------- step 3: judge (agentic loop) ----------

def _gather_evidence(rec: dict) -> List[Evidence]:
    return sorted(rec["ev"].values(), key=lambda e: e.score, reverse=True)


def _dryrun_verdict(score: float, max_score: float) -> tuple[str, str]:
    """Относительные пороги: адаптируются к шкале score (cosine или RRF)."""
    if max_score <= 0:
        return "не подходит", "low"
    ratio = score / max_score
    if ratio >= 0.6:
        return "подходит", "medium"
    if ratio >= 0.3:
        return "спорно", "low"
    return "не подходит", "low"


def judge_paper(store: Store, name: str, rec: dict, thesis_text: str,
                max_score: float, logger, trace: Trace) -> PaperResult:
    res = PaperResult(name=name, path=rec["path"], triage_score=round(rec["score"], 4))
    evidence = _gather_evidence(rec)

    if not settings.has_llm:
        verdict, conf = _dryrun_verdict(rec["score"], max_score)
        res.verdict, res.confidence = verdict, conf
        res.evidence = evidence[:3]
        res.reason = (f"dry-run: относительная близость {res.triage_score} "
                      f"(к лучшему {round(max_score,4)}). LLM не вызывался.")
        trace.add("judge", paper=name, mode="dry-run", verdict=verdict,
                  triage_score=res.triage_score,
                  prompt_template="Оцени релевантность статьи теме диплома по фрагментам …")
        logger.info("[judge] %s -> %s (dry-run)", name, verdict)
        return res

    # --- агентный цикл: хватает ли доказательств? (можно выключить AGENT_LOOP=false ради квоты) ---
    for it in range(settings.agent_max_iters if settings.agent_loop else 0):
        res.iters = it + 1
        ev_text = "\n".join(f"(стр.{e.page}) {e.snippet}" for e in evidence[:5])
        decide = [
            {"role": "system", "content": "Ты агент-ресёрчер. Верни строго JSON."},
            {"role": "user", "content":
                f"Тема диплома:\n{thesis_text}\n\nФрагменты статьи '{name}':\n{ev_text}\n\n"
                "Достаточно ли этих фрагментов, чтобы уверенно решить, релевантна ли статья?\n"
                "Если нет — дай refine_query для СЕМАНТИЧЕСКОГО поиска ВНУТРИ этой же статьи "
                "(локальный векторный индекс, не интернет). Это должна быть короткая фраза или "
                "5-8 ключевых терминов на английском, ОПИСЫВАЮЩИХ искомый аспект. "
                "Запрещено: операторы site:, OR/AND, кавычки, URL, годы.\n"
                'JSON: {"enough": true|false, "refine_query": "..."}'},
        ]
        try:
            d = llm.chat_json(decide)
        except llm.LLMError as exc:
            logger.warning("[judge] %s: ошибка решения (%s), выношу вердикт по текущему", name, exc)
            trace.add("agent_loop", paper=name, iter=it + 1, error=str(exc))
            break
        trace.add("agent_loop", paper=name, iter=it + 1, enough=d.get("enough"),
                  refine_query=d.get("refine_query", ""))
        if d.get("enough") or it == settings.agent_max_iters - 1:
            break
        # дозапрос доказательств именно по этой статье (запрос очищаем от гугл-операторов)
        refined = _clean_query(d.get("refine_query") or "") or name
        more = store.search(refined, limit=settings.search_limit, paper_name=name)
        for h in more:
            if h.page not in rec["ev"] or h.score > rec["ev"][h.page].score:
                rec["ev"][h.page] = Evidence(page=h.page, snippet=h.text[:300], score=h.score)
        evidence = _gather_evidence(rec)
        logger.info("[judge] %s: дозапрос '%s' -> +%d фрагментов",
                    name, d.get("refine_query"), len(more))

    # --- вердикт ---
    ev_text = "\n".join(f"(стр.{e.page}) {e.snippet}" for e in evidence[:5])
    verdict_prompt = [
        {"role": "system", "content": "Ты оцениваешь релевантность научной статьи теме диплома. "
         "Отвечай только на основании фрагментов, не выдумывай. Верни строго JSON."},
        {"role": "user", "content":
            f"Тема диплома:\n{thesis_text}\n\nФрагменты статьи '{name}':\n{ev_text}\n\n"
            f"Вердикт строго одно из: {VERDICTS}. JSON: "
            '{"verdict": "...", "confidence": "low|medium|high", '
            '"reason": "почему (1-2 предложения, ссылайся на содержание)", '
            '"missing": "чего не хватает / что смущает"}'},
    ]
    try:
        v = llm.chat_json(verdict_prompt)
        res.verdict = v.get("verdict", "спорно") if v.get("verdict") in VERDICTS else "спорно"
        res.confidence = v.get("confidence", "low")
        res.reason = v.get("reason", "")
        res.missing = v.get("missing", "")
        res.llm_used = True
        res.evidence = evidence[:3]
        trace.add("judge", paper=name, mode="llm", provider=llm.last_used_provider,
                  verdict=res.verdict, confidence=res.confidence, iters=res.iters)
        logger.info("[judge] %s -> %s (%s, %d итер.)", name, res.verdict, res.confidence, res.iters)
        if settings.self_verify:
            _self_verify(res, ev_text, logger, trace)
    except llm.LLMError as exc:
        logger.warning("[judge] %s: вердикт не получен (%s), деградация до триажа", name, exc)
        verdict, conf = _dryrun_verdict(rec["score"], max_score)
        res.verdict, res.confidence = verdict, conf
        res.reason = f"LLM-вердикт не удался ({exc}); вердикт по близости."
        res.evidence = evidence[:3]
        trace.add("judge", paper=name, mode="degraded", error=str(exc), verdict=verdict)
    return res


def batch_judge(top, thesis_text: str, max_score: float, logger, trace: Trace) -> List[PaperResult]:
    """Один LLM-вызов на ВСЕ топ-статьи (быстро, экономит квоту). Агентность сохранена:
    модель обосновывает вердикт и сама проверяет, подтверждён ли он фрагментами (grounded)."""
    results: List[PaperResult] = []
    items = list(top)

    # dry-run: без LLM — эвристические вердикты по относительной близости
    if not settings.has_llm:
        for name, rec in items:
            verdict, conf = _dryrun_verdict(rec["score"], max_score)
            r = PaperResult(name=name, path=rec["path"], triage_score=round(rec["score"], 4),
                            verdict=verdict, confidence=conf, evidence=_gather_evidence(rec)[:3],
                            reason=f"dry-run: относительная близость {round(rec['score'],4)}. LLM не вызывался.")
            results.append(r)
        trace.add("judge_batch", mode="dry-run", papers=[n for n, _ in items])
        return results

    # собираем контекст: по каждой статье — топ-фрагменты
    blocks = []
    for i, (name, rec) in enumerate(items):
        ev = _gather_evidence(rec)[:4]
        ev_text = "\n".join(f"(стр.{e.page}) {e.snippet}" for e in ev)
        blocks.append(f"[{i}] Статья: {name}\nФрагменты:\n{ev_text}")
    prompt = [
        {"role": "system", "content": "Ты оцениваешь релевантность научных статей теме диплома. "
         "Отвечай только по фрагментам, не выдумывай. Верни строго JSON."},
        {"role": "user", "content":
            f"Тема диплома:\n{thesis_text}\n\nСтатьи и фрагменты:\n" + "\n\n".join(blocks) +
            f"\n\nДля каждой статьи дай вердикт (одно из {VERDICTS}). Верни JSON: "
            '{"results":[{"id":0,"verdict":"...","confidence":"low|medium|high",'
            '"reason":"почему (1-2 предложения по содержанию)","missing":"что смущает",'
            '"grounded":true|false}]}'},
    ]
    try:
        data = llm.chat_json(prompt)
        by_id = {int(r.get("id", -1)): r for r in data.get("results", [])}
        logger.info("[judge] батч-вердикт по %d статьям (provider=%s)", len(items), llm.last_used_provider)
        trace.add("judge_batch", mode="llm", provider=llm.last_used_provider, n=len(items))
        for i, (name, rec) in enumerate(items):
            v = by_id.get(i, {})
            verdict = v.get("verdict") if v.get("verdict") in VERDICTS else "спорно"
            conf = v.get("confidence", "low")
            if v.get("grounded") is False and conf == "high":
                conf = "medium"   # неподтверждённое обоснование -> снижаем уверенность
            results.append(PaperResult(
                name=name, path=rec["path"], triage_score=round(rec["score"], 4),
                verdict=verdict, confidence=conf, reason=v.get("reason", ""),
                missing=v.get("missing", ""), llm_used=True, iters=1,
                evidence=_gather_evidence(rec)[:3],
                self_check=("подтверждено" if v.get("grounded") else "не подтверждено")))
    except llm.LLMError as exc:
        logger.warning("[judge] батч не удался (%s) -> деградация до триажа", exc)
        trace.add("judge_batch", mode="degraded", error=str(exc))
        for name, rec in items:
            verdict, conf = _dryrun_verdict(rec["score"], max_score)
            results.append(PaperResult(
                name=name, path=rec["path"], triage_score=round(rec["score"], 4),
                verdict=verdict, confidence=conf, evidence=_gather_evidence(rec)[:3],
                reason=f"LLM-вердикт не удался ({exc}); вердикт по близости."))
    return results


def _self_verify(res: PaperResult, ev_text: str, logger, trace: Trace) -> None:
    """Самопроверка: подтверждается ли обоснование цитатами (анти-галлюцинация)."""
    check = [
        {"role": "system", "content": "Проверь, подтверждается ли вывод приведёнными фрагментами. "
         "Верни строго JSON."},
        {"role": "user", "content":
            f"Фрагменты:\n{ev_text}\n\nВывод: вердикт='{res.verdict}', обоснование='{res.reason}'.\n"
            'Подтверждается ли обоснование фрагментами? JSON: {"grounded": true|false, "note": "..."}'},
    ]
    try:
        c = llm.chat_json(check)
        res.self_check = ("подтверждено" if c.get("grounded") else "НЕ подтверждено") + \
                         (f": {c.get('note')}" if c.get("note") else "")
        if not c.get("grounded") and res.confidence == "high":
            res.confidence = "medium"   # понижаем уверенность при неподтверждённом обосновании
        trace.add("self_verify", paper=res.name, grounded=c.get("grounded"), note=c.get("note", ""))
    except llm.LLMError as exc:
        res.self_check = f"проверка не выполнена ({exc})"
        trace.add("self_verify", paper=res.name, error=str(exc))


# ---------- orchestration ----------

def run_agent(store: Store, thesis_text: str, all_paper_names: List[str],
              logger, trace: Trace) -> List[PaperResult]:
    plan = plan_search(thesis_text, logger, trace)
    agg = triage(store, plan["queries"], logger, trace)

    max_score = max((r["score"] for r in agg.values()), default=0.0)
    ranked = sorted(agg.items(), key=lambda kv: kv[1]["score"], reverse=True)
    top = ranked[: settings.top_k_papers]
    top_names = {n for n, _ in top}

    results: List[PaperResult] = []
    if settings.agent_loop:
        # полная агентность: поштучный цикл "хватает ли доказательств" (медленнее, для демо)
        for name, rec in top:
            results.append(judge_paper(store, name, rec, thesis_text, max_score, logger, trace))
    else:
        # по умолчанию: батч-вердикт одним вызовом (быстро, экономит квоту)
        results.extend(batch_judge(top, thesis_text, max_score, logger, trace))

    # статьи без попаданий / ниже топа -> честный отказ без LLM
    for name in all_paper_names:
        if name in agg and name in top_names:
            continue
        rec = agg.get(name, {"path": "", "score": 0.0, "ev": {}})
        results.append(PaperResult(name=name, path=rec.get("path", ""),
                                   triage_score=round(rec["score"], 4),
                                   verdict="не подходит", confidence="low",
                                   reason="вне топ-кандидатов по близости; LLM не вызывался"))
        trace.add("refusal", paper=name, triage_score=round(rec["score"], 4))

    order = {"подходит": 0, "спорно": 1, "не подходит": 2}
    results.sort(key=lambda r: (order[r.verdict], -r.triage_score))
    return results
