import json

from src import ingest
from src.ingest import clean_text, count_pua


def test_bullet_glyphs_replaced():
    assert clean_text(" item one\n item two") == "- item one\n- item two"


def test_unmapped_private_use_glyphs_dropped():
    out = clean_text("abc")
    assert out == "abc" and count_pua(out) == 0


def test_ligatures_folded():
    assert clean_text("ﬁrst ﬂow") == "first flow"


def test_blank_lines_and_spaces_collapsed():
    assert clean_text("a   b  \n\n\n\n\nc") == "a b\n\nc"


def test_count_pua():
    assert count_pua("xy") == 2 and count_pua("plain") == 0


def test_save_chunks_roundtrip(tmp_path, monkeypatch):
    class Doc:
        def __init__(self, text, meta):
            self.page_content, self.metadata = text, meta

    monkeypatch.setattr(ingest, "VECTORSTORE_DIR", tmp_path)
    ingest.save_chunks([Doc("hello", {"source": "a.pdf", "page": 2})])
    saved = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
    assert saved == [{"text": "hello", "metadata": {"source": "a.pdf", "page": 2}}]
