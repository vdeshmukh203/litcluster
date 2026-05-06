"""Tests for litcluster.py.

Run with:  pytest tests/ -v
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import litcluster as lc


# ---------------------------------------------------------------------------
# Helper: minimal synthetic corpus
# ---------------------------------------------------------------------------

_CSV_CONTENT = """\
paper_id,title,abstract,authors,year,venue,doi,keywords
1,Deep Neural Networks,Using neural networks for image classification,A. Smith,2020,NeurIPS,,deep learning
2,Convolutional Architectures,CNN architectures for visual recognition tasks,B. Jones,2021,CVPR,,CNN vision
3,Linear Regression Analysis,Statistical regression methods for prediction,C. Lee,2019,JMLR,,statistics
4,Logistic Regression Methods,Binary classification using logistic models,D. Kim,2020,ICML,,regression
5,Bayesian Inference,Probabilistic models and Bayesian theorem,E. Patel,2021,NeurIPS,,bayesian
6,Random Forest Ensemble,Tree ensemble methods for classification tasks,F. Wang,2020,ICML,,ensemble trees
"""

_JSONL_CONTENT = "\n".join(
    json.dumps({"paper_id": str(i), "title": f"Paper {i}",
                "abstract": f"Research abstract discussing topic {i}"})
    for i in range(5)
)

_BIB_CONTENT = r"""
@article{smith2020,
  author  = {Smith, A.},
  title   = {Neural Network Methods},
  journal = {NeurIPS},
  year    = {2020},
  abstract = {Deep learning with neural networks},
}
@inproceedings{lee2019,
  author    = {Lee, C.},
  title     = {Statistical Regression},
  booktitle = {JMLR},
  year      = {2019},
  abstract  = {Linear and logistic regression methods},
}
"""


def _write_tmp(content: str, suffix: str) -> pathlib.Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8", newline=""
    )
    tmp.write(content)
    tmp.close()
    return pathlib.Path(tmp.name)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

def test_module_exposes_litcluster():
    assert hasattr(lc, "LitCluster")


def test_module_exposes_paper():
    assert hasattr(lc, "Paper")


def test_module_exposes_cluster():
    assert hasattr(lc, "Cluster")


# ---------------------------------------------------------------------------
# _tokenise
# ---------------------------------------------------------------------------

def test_tokenise_returns_list():
    assert isinstance(lc._tokenise("hello world"), list)


def test_tokenise_min_length():
    tokens = lc._tokenise("a ab abc abcd")
    assert "a" not in tokens
    assert "ab" not in tokens
    assert "abc" in tokens
    assert "abcd" in tokens


def test_tokenise_removes_stopwords():
    tokens = lc._tokenise("the quick brown fox")
    assert "the" not in tokens
    assert "fox" in tokens


def test_tokenise_lowercases():
    tokens = lc._tokenise("Neural Network")
    assert "neural" in tokens
    assert "network" in tokens


def test_tokenise_empty_string():
    assert lc._tokenise("") == []


# ---------------------------------------------------------------------------
# _tfidf
# ---------------------------------------------------------------------------

def test_tfidf_empty():
    vectors, vocab = lc._tfidf([])
    assert vectors == []
    assert vocab == []


def test_tfidf_single_doc():
    vectors, vocab = lc._tfidf([["machine", "learning"]])
    assert len(vectors) == 1
    assert "machine" in vectors[0]
    assert "learning" in vectors[0]


def test_tfidf_weights_positive():
    docs = [["neural", "network"], ["linear", "regression"]]
    vectors, _ = lc._tfidf(docs)
    for vec in vectors:
        assert all(v > 0 for v in vec.values())


def test_tfidf_vocab_sorted():
    docs = [["zebra", "apple"], ["mango"]]
    _, vocab = lc._tfidf(docs)
    assert vocab == sorted(vocab)


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------

def test_cosine_identical():
    v = {"a": 1.0, "b": 2.0}
    assert abs(lc._cosine(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal():
    assert lc._cosine({"x": 1.0}, {"y": 1.0}) == 0.0


def test_cosine_empty_vectors():
    assert lc._cosine({}, {"a": 1.0}) == 0.0
    assert lc._cosine({"a": 1.0}, {}) == 0.0


def test_cosine_symmetry():
    a = {"x": 1.0, "y": 2.0}
    b = {"y": 1.0, "z": 3.0}
    assert abs(lc._cosine(a, b) - lc._cosine(b, a)) < 1e-9


# ---------------------------------------------------------------------------
# _kmeans
# ---------------------------------------------------------------------------

def test_kmeans_empty():
    assert lc._kmeans([], k=3) == []


def test_kmeans_single_point():
    assert lc._kmeans([{"a": 1.0}], k=1) == [0]


def test_kmeans_label_count():
    vecs = [{"a": float(i)} for i in range(6)]
    labels = lc._kmeans(vecs, k=3)
    assert len(labels) == 6


def test_kmeans_k_capped_at_n():
    vecs = [{"a": 1.0}, {"b": 1.0}]
    labels = lc._kmeans(vecs, k=100)
    assert len(set(labels)) <= 2


def test_kmeans_reproducible():
    vecs = [{"a": float(i), "b": float(i % 3)} for i in range(10)]
    assert lc._kmeans(vecs, k=3, seed=0) == lc._kmeans(vecs, k=3, seed=0)


# ---------------------------------------------------------------------------
# Paper dataclass
# ---------------------------------------------------------------------------

def test_paper_text_combines_fields():
    p = lc.Paper(paper_id="1", title="Deep Learning",
                 abstract="Neural networks", keywords="DL")
    assert "Deep Learning" in p.text
    assert "Neural networks" in p.text
    assert "DL" in p.text


def test_paper_to_dict_keys():
    p = lc.Paper(paper_id="p1", title="Test")
    d = p.to_dict()
    for key in ("paper_id", "title", "abstract", "authors", "year", "venue", "doi", "keywords"):
        assert key in d


def test_paper_defaults():
    p = lc.Paper(paper_id="x", title="T")
    assert p.abstract == ""
    assert p.year == ""


# ---------------------------------------------------------------------------
# Cluster dataclass
# ---------------------------------------------------------------------------

def test_cluster_label_contains_id():
    c = lc.Cluster(cluster_id=3, top_terms=["alpha", "beta"])
    assert "3" in c.label


def test_cluster_label_shows_top_terms():
    c = lc.Cluster(cluster_id=0, top_terms=["neural", "network", "deep"])
    assert "neural" in c.label


def test_cluster_to_dict():
    c = lc.Cluster(cluster_id=1, papers=[], top_terms=["foo"])
    d = c.to_dict()
    assert d["cluster_id"] == 1
    assert d["size"] == 0
    assert "papers" in d


# ---------------------------------------------------------------------------
# LitCluster validation
# ---------------------------------------------------------------------------

def test_litcluster_k_zero_raises():
    with pytest.raises(ValueError):
        lc.LitCluster(k=0)


def test_litcluster_k_negative_raises():
    with pytest.raises(ValueError):
        lc.LitCluster(k=-1)


def test_litcluster_max_iter_zero_raises():
    with pytest.raises(ValueError):
        lc.LitCluster(k=1, max_iter=0)


# ---------------------------------------------------------------------------
# LitCluster.from_csv
# ---------------------------------------------------------------------------

def test_from_csv_paper_count():
    path = _write_tmp(_CSV_CONTENT, ".csv")
    try:
        obj = lc.LitCluster.from_csv(path, k=2, min_term_freq=1)
        assert len(obj.papers) == 6
    finally:
        path.unlink()


def test_from_csv_paper_fields():
    path = _write_tmp(_CSV_CONTENT, ".csv")
    try:
        obj = lc.LitCluster.from_csv(path, k=2, min_term_freq=1)
        p = obj.papers[0]
        assert p.paper_id == "1"
        assert "Neural" in p.title
        assert p.year == "2020"
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# LitCluster.from_jsonl
# ---------------------------------------------------------------------------

def test_from_jsonl_paper_count():
    path = _write_tmp(_JSONL_CONTENT, ".jsonl")
    try:
        obj = lc.LitCluster.from_jsonl(path, k=2, min_term_freq=1)
        assert len(obj.papers) == 5
    finally:
        path.unlink()


def test_from_jsonl_skips_blank_lines():
    content = _JSONL_CONTENT + "\n\n"
    path = _write_tmp(content, ".jsonl")
    try:
        obj = lc.LitCluster.from_jsonl(path, k=2, min_term_freq=1)
        assert len(obj.papers) == 5
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# LitCluster.from_bibtex
# ---------------------------------------------------------------------------

def test_from_bibtex_paper_count():
    path = _write_tmp(_BIB_CONTENT, ".bib")
    try:
        obj = lc.LitCluster.from_bibtex(path, k=2, min_term_freq=1)
        assert len(obj.papers) == 2
    finally:
        path.unlink()


def test_from_bibtex_field_extraction():
    path = _write_tmp(_BIB_CONTENT, ".bib")
    try:
        obj = lc.LitCluster.from_bibtex(path, k=2, min_term_freq=1)
        ids = {p.paper_id for p in obj.papers}
        assert "smith2020" in ids
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# LitCluster.fit
# ---------------------------------------------------------------------------

def test_fit_empty_corpus():
    obj = lc.LitCluster(k=3)
    obj.fit()
    assert obj.clusters == []


def test_fit_accounts_for_all_papers():
    path = _write_tmp(_CSV_CONTENT, ".csv")
    try:
        obj = lc.LitCluster.from_csv(path, k=2, min_term_freq=1)
        obj.fit()
        total = sum(len(c.papers) for c in obj.clusters)
        assert total == 6
    finally:
        path.unlink()


def test_fit_cluster_count():
    path = _write_tmp(_CSV_CONTENT, ".csv")
    try:
        obj = lc.LitCluster.from_csv(path, k=3, min_term_freq=1)
        obj.fit()
        assert 1 <= len(obj.clusters) <= 3
    finally:
        path.unlink()


def test_fit_top_terms_populated():
    path = _write_tmp(_CSV_CONTENT, ".csv")
    try:
        obj = lc.LitCluster.from_csv(path, k=2, min_term_freq=1)
        obj.fit()
        for c in obj.clusters:
            assert isinstance(c.top_terms, list)
            assert len(c.top_terms) > 0
    finally:
        path.unlink()


def test_fit_reproducible():
    path = _write_tmp(_CSV_CONTENT, ".csv")
    try:
        obj1 = lc.LitCluster.from_csv(path, k=2, min_term_freq=1, seed=7)
        obj1.fit()
        obj2 = lc.LitCluster.from_csv(path, k=2, min_term_freq=1, seed=7)
        obj2.fit()
        labels1 = [p.paper_id for c in obj1.clusters for p in c.papers]
        labels2 = [p.paper_id for c in obj2.clusters for p in c.papers]
        assert labels1 == labels2
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# LitCluster.summary
# ---------------------------------------------------------------------------

def test_summary_contains_paper_count():
    path = _write_tmp(_CSV_CONTENT, ".csv")
    try:
        obj = lc.LitCluster.from_csv(path, k=2, min_term_freq=1)
        obj.fit()
        s = obj.summary()
        assert "6 papers" in s
    finally:
        path.unlink()


def test_summary_empty_corpus():
    obj = lc.LitCluster(k=2)
    obj.fit()
    s = obj.summary()
    assert "0 papers" in s


# ---------------------------------------------------------------------------
# LitCluster.export_csv
# ---------------------------------------------------------------------------

def test_export_csv_creates_file():
    inp = _write_tmp(_CSV_CONTENT, ".csv")
    out = pathlib.Path(tempfile.mktemp(suffix=".csv"))
    try:
        obj = lc.LitCluster.from_csv(inp, k=2, min_term_freq=1)
        obj.fit()
        obj.export_csv(out)
        assert out.exists()
    finally:
        inp.unlink()
        out.unlink(missing_ok=True)


def test_export_csv_header():
    inp = _write_tmp(_CSV_CONTENT, ".csv")
    out = pathlib.Path(tempfile.mktemp(suffix=".csv"))
    try:
        obj = lc.LitCluster.from_csv(inp, k=2, min_term_freq=1)
        obj.fit()
        obj.export_csv(out)
        rows = list(csv.reader(out.open(encoding="utf-8")))
        assert rows[0][0] == "cluster_id"
        assert len(rows) == 7  # 1 header + 6 papers
    finally:
        inp.unlink()
        out.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# LitCluster.export_json
# ---------------------------------------------------------------------------

def test_export_json_creates_file():
    inp = _write_tmp(_CSV_CONTENT, ".csv")
    out = pathlib.Path(tempfile.mktemp(suffix=".json"))
    try:
        obj = lc.LitCluster.from_csv(inp, k=2, min_term_freq=1)
        obj.fit()
        obj.export_json(out)
        assert out.exists()
    finally:
        inp.unlink()
        out.unlink(missing_ok=True)


def test_export_json_structure():
    inp = _write_tmp(_CSV_CONTENT, ".csv")
    out = pathlib.Path(tempfile.mktemp(suffix=".json"))
    try:
        obj = lc.LitCluster.from_csv(inp, k=2, min_term_freq=1)
        obj.fit()
        obj.export_json(out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert "cluster_id" in data[0]
        assert "papers" in data[0]
        assert "top_terms" in data[0]
    finally:
        inp.unlink()
        out.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------

def test_cli_summary(capsys):
    path = _write_tmp(_CSV_CONTENT, ".csv")
    try:
        rc = lc.main([str(path), "-k", "2", "--min-freq", "1"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "papers" in out
    finally:
        path.unlink()


def test_cli_missing_file(capsys):
    rc = lc.main(["nonexistent_file.csv"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err.lower() or "error" in err.lower()


def test_cli_export_csv(tmp_path):
    inp = _write_tmp(_CSV_CONTENT, ".csv")
    out = tmp_path / "out.csv"
    try:
        rc = lc.main([str(inp), "-k", "2", "--min-freq", "1",
                      "--format", "csv", "--output", str(out)])
        assert rc == 0
        assert out.exists()
    finally:
        inp.unlink()


def test_cli_export_json(tmp_path):
    inp = _write_tmp(_CSV_CONTENT, ".csv")
    out = tmp_path / "out.json"
    try:
        rc = lc.main([str(inp), "-k", "2", "--min-freq", "1",
                      "--format", "json", "--output", str(out)])
        assert rc == 0
        assert out.exists()
    finally:
        inp.unlink()
