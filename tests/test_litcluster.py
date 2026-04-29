"""Tests for litcluster — covers tokenisation, TF-IDF, cosine similarity,
k-means, Paper/Cluster data structures, LitCluster API, and I/O parsers."""

import csv
import json
import math
import pathlib
import sys
import tempfile

import pytest

# Allow running tests directly from the project root without installation.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import litcluster as lc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paper(i=0, title="", abstract="", keywords=""):
    return lc.Paper(paper_id=str(i), title=title, abstract=abstract, keywords=keywords)


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------

def test_import():
    assert hasattr(lc, "LitCluster")


def test_paper_class_exported():
    assert hasattr(lc, "Paper")


def test_cluster_class_exported():
    assert hasattr(lc, "Cluster")


def test_all_list():
    assert "LitCluster" in lc.__all__
    assert "Paper" in lc.__all__
    assert "Cluster" in lc.__all__


# ---------------------------------------------------------------------------
# _tokenise
# ---------------------------------------------------------------------------

def test_tokenise_basic():
    tokens = lc._tokenise("hello world test foo bar")
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_tokenise_removes_stopwords():
    tokens = lc._tokenise("the and or a an is was this that")
    assert tokens == []


def test_tokenise_lowercases():
    tokens = lc._tokenise("Machine Learning")
    assert all(t == t.lower() for t in tokens)


def test_tokenise_min_length():
    # Tokens shorter than 3 chars must be excluded.
    tokens = lc._tokenise("hi go do be")
    assert "hi" not in tokens
    assert "go" not in tokens


def test_tokenise_empty():
    assert lc._tokenise("") == []


def test_tokenise_numbers_excluded():
    # Digits are not alphabetic so they should not appear.
    tokens = lc._tokenise("2024 results")
    assert "2024" not in tokens


# ---------------------------------------------------------------------------
# _tfidf
# ---------------------------------------------------------------------------

def test_tfidf_empty():
    vectors, vocab = lc._tfidf([])
    assert vectors == []
    assert vocab == []


def test_tfidf_single_doc():
    vectors, vocab = lc._tfidf([["neural", "network"]])
    assert len(vectors) == 1
    assert "neural" in vectors[0]
    assert "network" in vectors[0]
    assert all(v > 0 for v in vectors[0].values())


def test_tfidf_multiple_docs():
    docs = [
        ["deep", "learning", "neural"],
        ["machine", "learning", "algorithm"],
        ["neural", "network", "deep"],
    ]
    vectors, vocab = lc._tfidf(docs)
    assert len(vectors) == 3
    # "learning" appears in 2/3 docs; "deep" in 2/3; "neural" in 2/3
    # common terms should have lower IDF than rare ones
    assert "learning" in vocab


def test_tfidf_idf_rare_term_higher():
    docs = [
        ["common", "rare_xyz"],
        ["common", "ordinary"],
        ["common", "ordinary"],
    ]
    vectors, vocab = lc._tfidf(docs)
    # "rare_xyz" only in doc 0; "common" in all — rare_xyz should score higher in doc 0
    assert vectors[0].get("rare_xyz", 0) > vectors[0].get("common", 0)


def test_tfidf_vocab_sorted():
    vectors, vocab = lc._tfidf([["zebra", "apple", "mango"]])
    assert vocab == sorted(vocab)


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------

def test_cosine_identical():
    v = {"alpha": 1.0, "beta": 0.5}
    assert abs(lc._cosine(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal():
    a = {"alpha": 1.0}
    b = {"beta": 1.0}
    assert lc._cosine(a, b) == 0.0


def test_cosine_zero_vector():
    assert lc._cosine({}, {"alpha": 1.0}) == 0.0
    assert lc._cosine({"alpha": 1.0}, {}) == 0.0


def test_cosine_symmetry():
    a = {"x": 1.0, "y": 2.0}
    b = {"x": 0.5, "z": 3.0}
    assert abs(lc._cosine(a, b) - lc._cosine(b, a)) < 1e-9


def test_cosine_range():
    a = {"x": 1.0, "y": 1.0}
    b = {"x": 0.5, "y": 2.0}
    sim = lc._cosine(a, b)
    assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# _kmeans
# ---------------------------------------------------------------------------

def test_kmeans_empty():
    assert lc._kmeans([], k=3) == []


def test_kmeans_k_capped():
    # Only 2 documents; k=5 should be silently capped to 2.
    vecs = [{"a": 1.0}, {"b": 1.0}]
    labels = lc._kmeans(vecs, k=5)
    assert len(labels) == 2
    assert all(l in (0, 1) for l in labels)


def test_kmeans_returns_correct_length():
    vecs = [{"a": float(i)} for i in range(1, 7)]
    labels = lc._kmeans(vecs, k=3)
    assert len(labels) == 6


def test_kmeans_reproducible():
    vecs = [{"a": float(i)} for i in range(1, 11)]
    labels1 = lc._kmeans(vecs, k=3, seed=7)
    labels2 = lc._kmeans(vecs, k=3, seed=7)
    assert labels1 == labels2


def test_kmeans_different_seeds_may_differ():
    vecs = [{"a": float(i), "b": float(10 - i)} for i in range(10)]
    labels1 = lc._kmeans(vecs, k=3, seed=1)
    labels2 = lc._kmeans(vecs, k=3, seed=999)
    # Not guaranteed to differ, but with very different seeds this is likely.
    # We just assert both are valid label lists.
    assert len(labels1) == len(labels2) == 10


# ---------------------------------------------------------------------------
# Paper dataclass
# ---------------------------------------------------------------------------

def test_paper_text_property():
    p = lc.Paper(paper_id="p1", title="Deep Learning", abstract="neural nets", keywords="AI")
    assert "Deep Learning" in p.text
    assert "neural nets" in p.text
    assert "AI" in p.text


def test_paper_to_dict_keys():
    p = lc.Paper(paper_id="p1", title="Test")
    d = p.to_dict()
    for key in ("paper_id", "title", "abstract", "authors", "year", "venue", "doi", "keywords"):
        assert key in d


def test_paper_defaults():
    p = lc.Paper(paper_id="x", title="T")
    assert p.abstract == ""
    assert p.year == ""
    assert p.doi == ""


# ---------------------------------------------------------------------------
# Cluster dataclass
# ---------------------------------------------------------------------------

def test_cluster_label_format():
    c = lc.Cluster(cluster_id=0, top_terms=["neural", "network", "deep", "learning"])
    assert c.label.startswith("Cluster 0:")
    assert "neural" in c.label


def test_cluster_to_dict_keys():
    c = lc.Cluster(cluster_id=1, papers=[_make_paper()], top_terms=["foo"])
    d = c.to_dict()
    for key in ("cluster_id", "size", "top_terms", "label", "papers"):
        assert key in d
    assert d["size"] == 1


def test_cluster_empty_top_terms():
    c = lc.Cluster(cluster_id=0)
    assert c.label == "Cluster 0: "


# ---------------------------------------------------------------------------
# LitCluster — end-to-end fit
# ---------------------------------------------------------------------------

_PAPERS = [
    lc.Paper("p1", "Deep learning for image recognition", "Convolutional neural networks"),
    lc.Paper("p2", "Recurrent neural networks for NLP", "Sequence modelling with LSTM"),
    lc.Paper("p3", "Random forests and gradient boosting", "Ensemble tree methods"),
    lc.Paper("p4", "Support vector machines classification", "Kernel methods for classification"),
    lc.Paper("p5", "Topic modelling with LDA", "Latent Dirichlet Allocation for text"),
    lc.Paper("p6", "Transformer architecture attention mechanism", "Self-attention BERT GPT"),
]


def test_fit_runs():
    obj = lc.LitCluster(k=2, min_term_freq=1)
    obj.papers = list(_PAPERS)
    obj.fit()
    assert len(obj.clusters) > 0


def test_fit_returns_self():
    obj = lc.LitCluster(k=2, min_term_freq=1)
    obj.papers = list(_PAPERS)
    result = obj.fit()
    assert result is obj


def test_fit_all_papers_assigned():
    obj = lc.LitCluster(k=2, min_term_freq=1)
    obj.papers = list(_PAPERS)
    obj.fit()
    assigned = sum(len(c.papers) for c in obj.clusters)
    assert assigned == len(_PAPERS)


def test_fit_empty_papers():
    obj = lc.LitCluster(k=3)
    obj.fit()
    assert obj.clusters == []


def test_fit_k_capped_to_n():
    obj = lc.LitCluster(k=100, min_term_freq=1)
    obj.papers = list(_PAPERS[:2])
    obj.fit()
    assert len(obj.clusters) <= 2


def test_fit_top_terms_populated():
    obj = lc.LitCluster(k=2, min_term_freq=1)
    obj.papers = list(_PAPERS)
    obj.fit()
    for c in obj.clusters:
        assert isinstance(c.top_terms, list)
        assert len(c.top_terms) > 0


def test_summary_string():
    obj = lc.LitCluster(k=2, min_term_freq=1)
    obj.papers = list(_PAPERS)
    obj.fit()
    s = obj.summary()
    assert isinstance(s, str)
    assert "papers" in s.lower()


def test_reproducible_with_same_seed():
    def _run():
        obj = lc.LitCluster(k=3, seed=42, min_term_freq=1)
        obj.papers = list(_PAPERS)
        obj.fit()
        return [c.cluster_id for c in obj.clusters]

    assert _run() == _run()


# ---------------------------------------------------------------------------
# LitCluster — export
# ---------------------------------------------------------------------------

def test_export_csv(tmp_path):
    obj = lc.LitCluster(k=2, min_term_freq=1)
    obj.papers = list(_PAPERS)
    obj.fit()
    out = tmp_path / "out.csv"
    obj.export_csv(out)
    assert out.exists()
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(_PAPERS)
    assert "cluster_id" in rows[0]
    assert "title" in rows[0]


def test_export_json(tmp_path):
    obj = lc.LitCluster(k=2, min_term_freq=1)
    obj.papers = list(_PAPERS)
    obj.fit()
    out = tmp_path / "out.json"
    obj.export_json(out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert all("cluster_id" in c for c in data)
    assert all("papers" in c for c in data)


# ---------------------------------------------------------------------------
# LitCluster — I/O parsers
# ---------------------------------------------------------------------------

def test_from_csv(tmp_path):
    p = tmp_path / "papers.csv"
    p.write_text(
        "paper_id,title,abstract,authors,year,venue,doi,keywords\n"
        "1,Deep Learning,Neural nets,LeCun,2015,Nature,10.1/foo,AI\n"
        "2,SVMs,Kernel methods,Vapnik,1995,NIPS,,classification\n"
    )
    obj = lc.LitCluster.from_csv(p, k=2, min_term_freq=1)
    assert len(obj.papers) == 2
    assert obj.papers[0].title == "Deep Learning"
    assert obj.papers[1].year == "1995"


def test_from_csv_missing_columns(tmp_path):
    p = tmp_path / "minimal.csv"
    p.write_text("title\nNeural Networks\nRandom Forests\n")
    obj = lc.LitCluster.from_csv(p)
    assert len(obj.papers) == 2
    assert obj.papers[0].doi == ""


def test_from_jsonl(tmp_path):
    p = tmp_path / "papers.jsonl"
    p.write_text(
        '{"paper_id":"a","title":"Attention Is All You Need","abstract":"Transformer"}\n'
        '{"paper_id":"b","title":"BERT","abstract":"Bidirectional transformer"}\n'
        "\n"  # blank line should be skipped
    )
    obj = lc.LitCluster.from_jsonl(p)
    assert len(obj.papers) == 2
    assert obj.papers[0].paper_id == "a"


def test_from_bibtex(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text(
        '@article{lecun2015,\n'
        '  title = {Deep Learning},\n'
        '  author = {LeCun, Y.},\n'
        '  year = {2015},\n'
        '  abstract = {Neural networks for image recognition},\n'
        '  journal = {Nature},\n'
        '}\n'
        '@inproceedings{vaswani2017,\n'
        '  title = {Attention Is All You Need},\n'
        '  author = {Vaswani, A.},\n'
        '  year = {2017},\n'
        '  abstract = {Transformer self-attention mechanism},\n'
        '  booktitle = {NeurIPS},\n'
        '}\n'
    )
    obj = lc.LitCluster.from_bibtex(bib)
    assert len(obj.papers) == 2
    titles = {p.title for p in obj.papers}
    assert "Deep Learning" in titles
    assert "Attention Is All You Need" in titles


def test_from_bibtex_key_extracted(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text(
        '@article{smith2020,\n'
        '  title = {Test Paper},\n'
        '  year = {2020},\n'
        '}\n'
    )
    obj = lc.LitCluster.from_bibtex(bib)
    assert obj.papers[0].paper_id == "smith2020"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_missing_file(capsys):
    ret = lc.main(["nonexistent_file.csv"])
    assert ret == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err.lower() or "error" in captured.err.lower()


def test_cli_summary(tmp_path, capsys):
    p = tmp_path / "papers.csv"
    p.write_text(
        "title,abstract\n"
        + "\n".join(
            f"Paper {i},Abstract about topic {i % 3}"
            for i in range(10)
        )
    )
    ret = lc.main([str(p), "-k", "2", "--format", "summary"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "Cluster" in out


def test_cli_csv_output(tmp_path):
    p = tmp_path / "papers.csv"
    p.write_text(
        "title,abstract\n"
        + "\n".join(f"Paper {i},Abstract {i}" for i in range(6))
    )
    out = tmp_path / "out.csv"
    ret = lc.main([str(p), "-k", "2", "--format", "csv", "-o", str(out)])
    assert ret == 0
    assert out.exists()


def test_cli_json_output(tmp_path):
    p = tmp_path / "papers.jsonl"
    lines = [json.dumps({"title": f"Paper {i}", "abstract": f"Abstract {i}"}) for i in range(6)]
    p.write_text("\n".join(lines))
    out = tmp_path / "out.json"
    ret = lc.main([str(p), "-k", "2", "--format", "json", "-o", str(out)])
    assert ret == 0
    data = json.loads(out.read_text())
    assert isinstance(data, list)
