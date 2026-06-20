"""End-to-end dry-run: пайплайн отрабатывает без ключей и даёт осмысленный порядок."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(env_extra: dict, thesis: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update({"DRY_RUN": "true", "HYBRID": "false", **env_extra})
    return subprocess.run(
        [sys.executable, "-m", "src.cli", "--dry-run", "--rebuild", "--thesis", thesis],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=180,
    )


def test_dryrun_pipeline(tmp_path: Path):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "relevant.txt").write_text(
        "anomaly detection fraud transactions time series LSTM autoencoder imbalanced "
        "precision recall F1 streaming concept drift", encoding="utf-8")
    (papers / "irrelevant.txt").write_text(
        "convolutional neural network image segmentation MRI brain dice coefficient U-Net",
        encoding="utf-8")
    thesis = tmp_path / "thesis.md"
    thesis.write_text(
        "# Тема\nAnomaly detection in financial transaction time series for fraud detection.\n"
        "## Ключевые слова\nanomaly detection, fraud, transactions, time series, imbalanced, LSTM",
        encoding="utf-8")
    out = tmp_path / "out"

    proc = _run({
        "PAPERS_DIR": str(papers),
        "QDRANT_PATH": str(tmp_path / "qdb"),
        "OUTPUT_DIR": str(out),
    }, thesis=str(thesis))
    assert proc.returncode == 0, proc.stderr

    report = out / "report.md"
    trace = out / "trace.json"
    runlog = out / "run.log"
    assert report.exists() and trace.exists() and runlog.exists()
    assert "DRY-RUN" in report.read_text(encoding="utf-8")

    # релевантная статья должна получить близость не ниже нерелевантной
    events = json.loads(trace.read_text(encoding="utf-8"))["events"]
    ranked = next(e for e in events if e["step"] == "triage")["ranked"]
    scores = {name: score for name, score in ranked}
    assert scores["relevant.txt"] >= scores.get("irrelevant.txt", 0.0)
