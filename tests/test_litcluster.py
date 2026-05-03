"""
Tests for litcluster.

Covers: tokenisation, TF-IDF, cosine similarity, k-means, Paper/Cluster
dataclasses, LitCluster.fit(), all three loaders, and both export formats.
"""

import csv
import json
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import litcluster as lc
from litcluster import (
    LitCluster, Paper, Cluster,
    _tokenise, _tfidf, _cosine, _kmeans,
)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

def test_module_exports_litcluster():
    assert hasattr(lc, "LitCluster")

def test_module_exports_paper():
    assert hasattr(lc, "Paper")

def test_module_exports_cluster():
    assert hasattr(lc, "Cluster")

def test_module_exports_helpers():
    for name in ("_tokenise", "_tfidf", "_cosine", "_kmeans"):
        assert hasattr(lc, name), f"missing: {name}"


# ---------------------------------------------------------------------------
# _tokenise
# ---------------------------------------------------------------------------

def test_tokenise_basic():
    tokens = _tokenise("deep learning neural network")
    assert "deep" in tokens
    assert "learning" in tokens
    assert "neural" in tokens
    assert "network" in tokens

def test_tokenise_removes_stopwords():
    tokens = _tokenise("the quick brown fox")
    assert "the" not in tokens

def test_tokenise_min_length_three():
    # 1-2 char words are dropped
    tokens = _tokenise("a an in go")
    assert tokens == []

def test_tokenise_lowercases():
    tokens = _tokenise("NEURAL Networks")
    assert "neural" in tokens
    assert "networks" in tokens
    assert "NEURAL" not in tokens

def test_tokenise_empty_string():
    assert _tokenise("") == []

def test_tokenise_numbers_excluded():
    tokens = _tokenise("2024 version three four")
    assert "2024" not in tokens
    assert "three" in tokens

def test_tokenise_punctuation_stripped():
    tokens = _tokenise("deep-learning, NLP! text.")
    assert "deep" in tokens
    assert "learning" in tokens
    assert "nlp" in tokens
    assert "text" in tokens


# ---------------------------------------------------------------------------
# _tfidf
# ---------------------------------------------------------------------------

def test_tfidf_shape():
    docs = [["neural", "network"], ["climate", "change"]]
    vecs, vocab = _tfidf(docs)
    assert len(vecs) == 2
    assert len(vocab) > 0

def test_tfidf_empty_input():
    vecs, vocab = _tfidf([])
    assert vecs == []
    assert vocab == []

def test_tfidf_single_doc():
    vecs, vocab = _tfidf([["neural", "network"]])
    assert len(vecs) == 1
    assert "neural" in vecs[0]

def test_tfidf_rare_term_higher_idf():
    # "rare" appears in 1 doc, "common" appears in all 3
    docs = [
        ["rare", "common"],
        ["common", "topic"],
        ["common", "word"],
    ]
    vecs, _ = _tfidf(docs)
    assert vecs[0]["rare"] > vecs[0]["common"]

def test_tfidf_vocab_sorted():
    docs = [["zebra", "apple", "mango"]]
    _, vocab = _tfidf(docs)
    assert vocab == sorted(vocab)

def test_tfidf_empty_document_in_list():
    # An empty document should yield an empty vector without crashing
    docs = [["neural"], [], ["gene"]]
    vecs, vocab = _tfidf(docs)
    assert len(vecs) == 3
    assert vecs[1] == {}


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------

def test_cosine_identical_vectors():
    v = {"a": 1.0, "b": 2.0}
    assert abs(_cosine(v, v) - 1.0) < 1e-9

def test_cosine_orthogonal_vectors():
    assert _cosine({"a": 1.0}, {"b": 1.0}) == 0.0

def test_cosine_empty_vector():
    assert _cosine({}, {"a": 1.0}) == 0.0
    assert _cosine({"a": 1.0}, {}) == 0.0
    assert _cosine({}, {}) == 0.0

def test_cosine_symmetry():
    a = {"x": 1.0, "y": 2.0}
    b = {"x": 3.0, "z": 1.0}
    assert abs(_cosine(a, b) - _cosine(b, a)) < 1e-9

def test_cosine_range():
    a = {"x": 1.0, "y": 0.5}
    b = {"x": 0.3, "y": 1.0, "z": 0.2}
    sim = _cosine(a, b)
    assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# _kmeans
# ---------------------------------------------------------------------------

def test_kmeans_returns_correct_length():
    vecs = [{"a": 1.0}, {"b": 1.0}, {"c": 1.0}]
    labels = _kmeans(vecs, k=2)
    assert len(labels) == 3

def test_kmeans_labels_are_integers():
    vecs = [{"a": 1.0}, {"b": 1.0}]
    labels = _kmeans(vecs, k=2)
    assert all(isinstance(l, int) for l in labels)

def test_kmeans_empty_input():
    assert _kmeans([], k=3) == []

def test_kmeans_k_capped_at_n():
    vecs = [{"a": 1.0}, {"b": 1.0}]
    labels = _kmeans(vecs, k=100)
    assert len(labels) == 2

def test_kmeans_single_cluster():
    vecs = [{"a": 1.0}, {"a": 0.9, "b": 0.1}]
    labels = _kmeans(vecs, k=1)
    assert all(l == 0 for l in labels)

def test_kmeans_reproducible_with_seed():
    vecs = [{"a": float(i), "b": float(i % 3)} for i in range(10)]
    l1 = _kmeans(vecs, k=3, seed=7)
    l2 = _kmeans(vecs, k=3, seed=7)
    assert l1 == l2

def test_kmeans_different_seeds_may_differ():
    vecs = [{"a": float(i)} for i in range(20)]
    l1 = _kmeans(vecs, k=4, seed=1)
    l2 = _kmeans(vecs, k=4, seed=99)
    # Not guaranteed to differ, but with very different seeds they often do.
    # Just check both are valid.
    assert len(l1) == len(l2) == 20


# ---------------------------------------------------------------------------
# Paper
# ---------------------------------------------------------------------------

def test_paper_text_combines_fields():
    p = Paper("1", "Deep Learning", abstract="Neural networks", keywords="AI ML")
    assert "Deep Learning" in p.text
    assert "Neural networks" in p.text
    assert "AI ML" in p.text

def test_paper_to_dict_keys():
    p = Paper("42", "Title", abstract="Abs", authors="Alice", year="2024")
    d = p.to_dict()
    for key in ("paper_id", "title", "abstract", "authors", "year", "venue", "doi", "keywords"):
        assert key in d

def test_paper_to_dict_values():
    p = Paper("42", "Title", abstract="Abs", year="2024")
    d = p.to_dict()
    assert d["paper_id"] == "42"
    assert d["title"] == "Title"
    assert d["abstract"] == "Abs"
    assert d["year"] == "2024"

def test_paper_defaults():
    p = Paper("1", "Title")
    assert p.abstract == ""
    assert p.doi == ""


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------

def test_cluster_label_includes_terms():
    c = Cluster(0, papers=[], top_terms=["neural", "network", "deep"])
    assert "neural" in c.label
    assert "network" in c.label

def test_cluster_label_empty_terms():
    c = Cluster(1, papers=[], top_terms=[])
    assert "Cluster 1" in c.label

def test_cluster_to_dict_structure():
    p = Paper("1", "T1", abstract="abstract one")
    c = Cluster(0, papers=[p], top_terms=["neural"])
    d = c.to_dict()
    assert d["cluster_id"] == 0
    assert d["size"] == 1
    assert len(d["papers"]) == 1
    assert d["top_terms"] == ["neural"]

def test_cluster_size_reflects_papers():
    papers = [Paper(str(i), f"Paper {i}") for i in range(5)]
    c = Cluster(0, papers=papers)
    assert c.to_dict()["size"] == 5


# ---------------------------------------------------------------------------
# LitCluster.fit()
# ---------------------------------------------------------------------------

def _make_lc(n: int = 12, k: int = 3) -> LitCluster:
    """Helper: create a fitted LitCluster with synthetic multi-topic data."""
    topics = [
        ("deep learning neural network training",
         "deep neural network training epochs gradient"),
        ("climate change temperature greenhouse emissions",
         "carbon emissions global warming temperature rise"),
        ("gene expression protein sequence genome",
         "genomics sequencing DNA protein analysis biology"),
    ]
    obj = LitCluster(k=k, min_term_freq=1, seed=0)
    for i in range(n):
        title, abstract = topics[i % len(topics)]
        obj.papers.append(Paper(str(i), f"Paper {i}", abstract=abstract))
    obj.fit()
    return obj


def test_fit_returns_self():
    obj = LitCluster(k=2, min_term_freq=1)
    obj.papers.append(Paper("1", "ML", abstract="neural network"))
    obj.papers.append(Paper("2", "Bio", abstract="gene protein"))
    assert obj.fit() is obj

def test_fit_creates_clusters():
    obj = _make_lc()
    assert len(obj.clusters) > 0

def test_fit_all_papers_assigned():
    n = 12
    obj = _make_lc(n=n)
    total = sum(len(c.papers) for c in obj.clusters)
    assert total == n

def test_fit_no_papers():
    obj = LitCluster().fit()
    assert obj.clusters == []

def test_fit_single_paper():
    obj = LitCluster(k=3, min_term_freq=1)
    obj.papers.append(Paper("1", "Only paper", abstract="deep learning"))
    obj.fit()
    assert len(obj.clusters) == 1

def test_fit_top_terms_nonempty():
    obj = _make_lc()
    for c in obj.clusters:
        assert len(c.top_terms) > 0

def test_summary_contains_paper_count():
    obj = _make_lc()
    s = obj.summary()
    assert "12" in s

def test_summary_contains_cluster_count():
    obj = _make_lc(k=3)
    s = obj.summary()
    assert "cluster" in s.lower()


# ---------------------------------------------------------------------------
# LitCluster.from_csv
# ---------------------------------------------------------------------------

def test_from_csv_basic(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text(
        "paper_id,title,abstract\n"
        "1,Deep Learning,neural network training\n"
        "2,Climate Science,global warming temperature\n",
        encoding="utf-8",
    )
    obj = LitCluster.from_csv(f)
    assert len(obj.papers) == 2
    assert obj.papers[0].title == "Deep Learning"
    assert obj.papers[1].abstract == "global warming temperature"

def test_from_csv_missing_optional_columns(tmp_path):
    f = tmp_path / "minimal.csv"
    f.write_text("title\nOnly Title\n", encoding="utf-8")
    obj = LitCluster.from_csv(f)
    assert len(obj.papers) == 1
    assert obj.papers[0].doi == ""

def test_from_csv_kwargs_passed(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("title\nPaper A\nPaper B\n", encoding="utf-8")
    obj = LitCluster.from_csv(f, k=7, seed=123)
    assert obj.k == 7
    assert obj.seed == 123


# ---------------------------------------------------------------------------
# LitCluster.from_jsonl
# ---------------------------------------------------------------------------

def test_from_jsonl_basic(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text(
        '{"paper_id":"1","title":"DL","abstract":"neural"}\n'
        '{"paper_id":"2","title":"Bio","abstract":"gene protein"}\n',
        encoding="utf-8",
    )
    obj = LitCluster.from_jsonl(f)
    assert len(obj.papers) == 2
    assert obj.papers[0].title == "DL"

def test_from_jsonl_skips_blank_lines(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text(
        '{"title":"A"}\n\n{"title":"B"}\n',
        encoding="utf-8",
    )
    obj = LitCluster.from_jsonl(f)
    assert len(obj.papers) == 2


# ---------------------------------------------------------------------------
# LitCluster.from_bibtex
# ---------------------------------------------------------------------------

_BIBTEX_SAMPLE = """\
@article{smith2020,
  title     = {Deep Learning Advances},
  abstract  = {We study neural networks and training methods.},
  author    = {Smith, Alice and Jones, Bob},
  year      = {2020},
  journal   = {Nature},
  doi       = {10.1000/xyz123},
  keywords  = {deep learning, neural networks},
}

@inproceedings{doe2021,
  title    = {Climate Modelling},
  abstract = {Temperature change and greenhouse gas emissions.},
  author   = {Doe, John},
  year     = {2021},
  booktitle = {Proceedings of ICCS},
}
"""

def test_from_bibtex_basic(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text(_BIBTEX_SAMPLE, encoding="utf-8")
    obj = LitCluster.from_bibtex(bib)
    assert len(obj.papers) >= 2

def test_from_bibtex_title_parsed(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text(_BIBTEX_SAMPLE, encoding="utf-8")
    obj = LitCluster.from_bibtex(bib)
    titles = [p.title for p in obj.papers]
    assert "Deep Learning Advances" in titles

def test_from_bibtex_abstract_parsed(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text(_BIBTEX_SAMPLE, encoding="utf-8")
    obj = LitCluster.from_bibtex(bib)
    abstracts = [p.abstract for p in obj.papers]
    assert any("neural networks" in a for a in abstracts)

def test_from_bibtex_year_parsed(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text(_BIBTEX_SAMPLE, encoding="utf-8")
    obj = LitCluster.from_bibtex(bib)
    years = {p.year for p in obj.papers}
    assert "2020" in years or "2021" in years

def test_from_bibtex_venue_journal(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text(_BIBTEX_SAMPLE, encoding="utf-8")
    obj = LitCluster.from_bibtex(bib)
    venues = [p.venue for p in obj.papers]
    assert any("Nature" in v for v in venues)

def test_from_bibtex_nested_braces(tmp_path):
    bib = tmp_path / "nested.bib"
    bib.write_text(
        "@article{test1,\n"
        "  title = {A {Nested} Title},\n"
        "  abstract = {Contains {nested} and {more} braces.},\n"
        "  year = {2022},\n"
        "}\n",
        encoding="utf-8",
    )
    obj = LitCluster.from_bibtex(bib)
    assert len(obj.papers) >= 1
    assert "Nested" in obj.papers[0].title

def test_from_bibtex_empty_file(tmp_path):
    bib = tmp_path / "empty.bib"
    bib.write_text("", encoding="utf-8")
    obj = LitCluster.from_bibtex(bib)
    assert obj.papers == []


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_export_csv_header(tmp_path):
    obj = _make_lc()
    out = tmp_path / "out.csv"
    obj.export_csv(out)
    with out.open(encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert "cluster_id" in header
    assert "title" in header

def test_export_csv_row_count(tmp_path):
    obj = _make_lc(n=12)
    out = tmp_path / "out.csv"
    obj.export_csv(out)
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    # header + 12 data rows
    assert len(rows) == 13

def test_export_json_is_list(tmp_path):
    obj = _make_lc()
    out = tmp_path / "out.json"
    obj.export_json(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) > 0

def test_export_json_cluster_structure(tmp_path):
    obj = _make_lc()
    out = tmp_path / "out.json"
    obj.export_json(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    for cluster in data:
        assert "cluster_id" in cluster
        assert "papers" in cluster
        assert "top_terms" in cluster

def test_export_json_roundtrip(tmp_path):
    obj = _make_lc(n=6, k=2)
    out = tmp_path / "out.json"
    obj.export_json(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    total = sum(c["size"] for c in data)
    assert total == 6
