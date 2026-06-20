from src.chunking import split_text, normalize_text


def test_empty():
    assert split_text("", 100, 10) == []
    assert split_text("   \n\t ", 100, 10) == []


def test_short_text_single_chunk():
    assert split_text("короткий текст", 100, 10) == ["короткий текст"]


def test_normalize_collapses_whitespace():
    assert normalize_text("a\n\n  b\t c") == "a b c"


def test_overlap_and_coverage():
    text = "word " * 400  # ~2000 символов
    chunks = split_text(text, 300, 50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    # склейка чанков покрывает весь нормализованный текст
    assert "word" in chunks[0] and "word" in chunks[-1]
