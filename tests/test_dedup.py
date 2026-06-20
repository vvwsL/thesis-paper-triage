from pathlib import Path

from src.pdf_extract import load_papers, extract_paper


def test_dedup_identical_content(tmp_path: Path):
    (tmp_path / "a.txt").write_text("одинаковый научный текст про anomaly detection", encoding="utf-8")
    (tmp_path / "b.txt").write_text("одинаковый научный текст про anomaly detection", encoding="utf-8")
    (tmp_path / "c.txt").write_text("другой текст про image segmentation", encoding="utf-8")

    valid, failed, dups = load_papers(tmp_path)
    assert len(valid) == 2          # дубль схлопнут
    assert len(dups) == 1
    assert not failed


def test_empty_file_is_failed(tmp_path: Path):
    (tmp_path / "scan.txt").write_text("   ", encoding="utf-8")
    paper = extract_paper(tmp_path / "scan.txt")
    assert paper.error and "no_text_layer" in paper.error


def test_empty_dir(tmp_path: Path):
    valid, failed, dups = load_papers(tmp_path)
    assert valid == [] and failed == [] and dups == []
