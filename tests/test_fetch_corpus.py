from src import fetch_corpus as fc
from src import ingest


def test_slugify_keeps_plus_apart_from_dash():
    assert fc.slugify("B+ tree") == "bplus_tree" and fc.slugify("B-tree") == "b_tree"
    assert fc.slugify("Dijkstra's algorithm") == "dijkstra_s_algorithm"


def test_strip_sections_drops_references_but_keeps_body():
    text = "Intro.\n\n== History ==\nOld.\n\n== References ==\nSomebody 1999.\n\n== Usage ==\nNew.\n\n== External links ==\nhttp://x"
    out = fc.strip_sections(text)
    assert "Intro." in out and "Old." in out and "New." in out
    assert "Somebody" not in out and "http://x" not in out


def test_saved_article_is_read_back_by_ingest(tmp_path):
    fc.save_article({"title": "Quicksort", "url": "https://en.wikipedia.org/wiki/Quicksort",
                     "text": "Quicksort is a sorting algorithm."}, tmp_path)
    docs = ingest.load_corpus(tmp_path)
    assert len(docs) == 1
    assert docs[0].metadata == {"source": "Wikipedia: Quicksort", "url": "https://en.wikipedia.org/wiki/Quicksort"}
    assert docs[0].page_content == "Quicksort is a sorting algorithm."
