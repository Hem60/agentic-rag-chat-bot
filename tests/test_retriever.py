from src.retriever import HybridRetriever, rrf_fuse, tokenize

CHUNKS = [
    {"text": "A stack follows the LIFO principle. Push and pop work at the top.",
     "metadata": {"source": "ds.pdf", "page": 1}},
    {"text": "A queue follows FIFO. Insertion at rear, deletion at front.",
     "metadata": {"source": "ds.pdf", "page": 2}},
    {"text": "AVL trees are self balancing binary search trees using rotations.",
     "metadata": {"source": "ds.pdf", "page": 3}},
    {"text": "Linked lists store nodes with pointers to the next node.",
     "metadata": {"source": "ds.pdf", "page": 4}},
]


class _FakeReranker:
    """Scores by how many query words appear in the chunk, then favours page 4."""
    def predict(self, pairs):
        return [len(set(tokenize(q)) & set(tokenize(t))) + (10 if "pointers" in t else 0)
                for q, t in pairs]


def test_tokenize_keeps_terms():
    assert tokenize("AVL O(n log n)") == ["avl", "o", "n", "log", "n"]


def test_rrf_item_in_both_lists_wins():
    fused = rrf_fuse([[1, 2, 3], [3, 1, 4]], [0.5, 0.5])
    assert fused[0] == 1  # ranks 1 and 2 beats ranks 3 and 1
    assert set(fused) == {1, 2, 3, 4}


def test_rrf_weights_shift_order():
    assert rrf_fuse([[1], [2]], [0.9, 0.1])[0] == 1
    assert rrf_fuse([[1], [2]], [0.1, 0.9])[0] == 2


def test_bm25_finds_exact_term_vector_misses():
    # vector search returns nothing useful; BM25 must still surface the AVL chunk
    r = HybridRetriever(CHUNKS, vector_search=lambda q, k: [1, 3], reranker=None, top_n=4)
    assert r.bm25_search("AVL rotations", 4)[0] == 2
    assert 2 in r.candidates("AVL rotations")


def test_known_question_returns_expected_page_in_top_n():
    r = HybridRetriever(CHUNKS, vector_search=lambda q, k: [0, 3], reranker=_FakeReranker(), top_n=2)
    pages = [h["page"] for h in r.search("what is the LIFO stack principle")]
    assert 1 in pages


def test_reranker_order_differs_from_fusion_order():
    r = HybridRetriever(CHUNKS, vector_search=lambda q, k: [0, 3], reranker=_FakeReranker(), top_n=4)
    q = "stack"
    fusion = r.candidates(q)
    hits = r.search(q)
    assert hits[0]["page"] == 4            # reranker promotes the pointers chunk
    assert fusion[0] == 0                  # fusion alone had the stack chunk first
    assert [h["rank"] for h in hits] == list(range(1, len(hits) + 1))
    assert hits == sorted(hits, key=lambda h: h["score"], reverse=True)


def test_hit_fields_and_top_n_limit():
    r = HybridRetriever(CHUNKS, vector_search=lambda q, k: [0, 1, 2, 3], reranker=_FakeReranker(), top_n=2)
    hits = r.search("stack")
    assert len(hits) == 2
    assert set(hits[0]) == {"text", "source", "page", "score", "rank"}
    assert hits[0]["source"] == "ds.pdf"


def test_empty_query_and_no_candidates():
    r = HybridRetriever(CHUNKS, vector_search=lambda q, k: [], reranker=None)
    assert r.search("???") == []
