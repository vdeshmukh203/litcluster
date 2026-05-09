"""
Test suite for litcluster.

Covers: imports, data structures, text-processing primitives, clustering
pipeline, file loaders (CSV / JSONL / BibTeX), export functions, and
edge-case robustness.
"""

import csv
import json
import pathlib
import sys
import warnings

import pytest

# Ensure the repo root is on the path when running without installation.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import litcluster as lc
from litcluster import (
    Cluster,
    LitCluster,
    Paper,
    _cosine,
    _kmeans,
    _tfidf,
    _tokenise,
    _bibtex_field,
)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

def test_import_litcluster():
    assert hasattr(lc, "LitCluster")


def test_import_paper():
    assert hasattr(lc, "Paper")


def test_import_cluster():
    assert hasattr(lc, "Cluster")


def test_import_tokenise():
    assert hasattr(lc, "_tokenise")


def test_version_string():
    assert hasattr(lc, "__version__")
    assert isinstance(lc.__version__, str)


# ---------------------------------------------------------------------------
# Paper dataclass
# ---------------------------------------------------------------------------

def test_paper_creation():
    p = Paper(paper_id="p1", title="Test Paper", abstract="An abstract.")
    assert p.paper_id == "p1"
    assert p.title == "Test Paper"
    assert p.abstract == "An abstract."


def test_paper_text_combines_fields():
    p = Paper("1", "TF-IDF Clustering", abstract="Vectorisation method.", keywords="nlp tfidf")
    assert "TF-IDF Clustering" in p.text
    assert "Vectorisation method." in p.text
    assert "nlp tfidf" in p.text


def test_paper_to_dict_keys():
    p = Paper(paper_id="x", title="Hello")
    d = p.to_dict()
    for key in ("paper_id", "title", "abstract", "authors", "year", "venue", "doi", "keywords"):
        assert key in d


def test_paper_to_dict_values():
    p = Paper(paper_id="42", title="Deep Learning", abstract="Neural nets", year="2024")
    d = p.to_dict()
    assert d["paper_id"] == "42"
    assert d["year"] == "2024"


def test_paper_repr():
    p = Paper(paper_id="1", title="A Very Long Title That Exceeds Fifty Characters In Total")
    r = repr(p)
    assert "Paper" in r
    assert "1" in r


# ---------------------------------------------------------------------------
# Cluster dataclass
# ---------------------------------------------------------------------------

def test_cluster_label_with_terms():
    c = Cluster(cluster_id=0, top_terms=["machine", "learning", "neural", "network"])
    assert "Cluster 0" in c.label
    assert "machine" in c.label


def test_cluster_label_without_terms():
    c = Cluster(cluster_id=2)
    assert "Cluster 2" in c.label
    assert "—" in c.label


def test_cluster_to_dict():
    p = Paper("1", "Test")
    c = Cluster(cluster_id=1, papers=[p], top_terms=["topic"])
    d = c.to_dict()
    assert d["cluster_id"] == 1
    assert d["size"] == 1
    assert d["top_terms"] == ["topic"]
    assert len(d["papers"]) == 1


def test_cluster_repr():
    c = Cluster(cluster_id=3, papers=[Paper("1", "X")], top_terms=["a", "b"])
    assert "Cluster" in repr(c)
    assert "3" in repr(c)


# ---------------------------------------------------------------------------
# _tokenise
# ---------------------------------------------------------------------------

def test_tokenise_returns_list():
    assert isinstance(_tokenise("hello world"), list)


def test_tokenise_min_length():
    tokens = _tokenise("a ab abc abcd")
    assert "a" not in tokens
    assert "ab" not in tokens
    assert "abc" in tokens
    assert "abcd" in tokens


def test_tokenise_removes_stopwords():
    tokens = _tokenise("the and or but in on at for of with")
    assert len(tokens) == 0


def test_tokenise_lowercases():
    tokens = _tokenise("MACHINE Learning Neural")
    assert "machine" in tokens
    assert "learning" in tokens


def test_tokenise_empty():
    assert _tokenise("") == []


def test_tokenise_numbers_excluded():
    # Only alpha tokens matched
    tokens = _tokenise("123 456 abc")
    assert "123" not in tokens
    assert "456" not in tokens
    assert "abc" in tokens


# ---------------------------------------------------------------------------
# _tfidf
# ---------------------------------------------------------------------------

def test_tfidf_basic_shape():
    docs = [["machine", "learning"], ["clustering", "kmeans"]]
    vectors, vocab = _tfidf(docs)
    assert len(vectors) == 2
    assert len(vocab) == 4
    assert all(isinstance(v, dict) for v in vectors)


def test_tfidf_empty_corpus():
    vectors, vocab = _tfidf([])
    assert vectors == []
    assert vocab == []


def test_tfidf_single_doc():
    vectors, vocab = _tfidf([["hello", "world"]])
    assert len(vectors) == 1
    assert len(vocab) == 2


def test_tfidf_weights_positive():
    docs = [["neural", "network", "deep"], ["cluster", "kmeans", "distance"]]
    vectors, _ = _tfidf(docs)
    for vec in vectors:
        assert all(v > 0 for v in vec.values())


def test_tfidf_shared_term_lower_idf():
    """A term shared across all docs should have lower weight than a unique one."""
    docs = [
        ["shared", "unique_a"],
        ["shared", "unique_b"],
    ]
    vectors, _ = _tfidf(docs)
    # 'shared' appears in both docs; unique terms appear in one
    assert vectors[0]["shared"] < vectors[0]["unique_a"]


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------

def test_cosine_identical_vectors():
    v = {"machine": 1.0, "learning": 0.5}
    assert abs(_cosine(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal_vectors():
    a = {"machine": 1.0}
    b = {"learning": 1.0}
    assert _cosine(a, b) == 0.0


def test_cosine_empty_vector():
    assert _cosine({}, {"machine": 1.0}) == 0.0
    assert _cosine({"machine": 1.0}, {}) == 0.0


def test_cosine_partial_overlap():
    a = {"x": 1.0, "y": 1.0}
    b = {"y": 1.0, "z": 1.0}
    sim = _cosine(a, b)
    assert 0.0 < sim < 1.0


# ---------------------------------------------------------------------------
# _kmeans
# ---------------------------------------------------------------------------

def test_kmeans_empty():
    assert _kmeans([], k=3) == []


def test_kmeans_k_capped_at_n():
    docs = [["apple"]]
    vectors, _ = _tfidf(docs)
    labels = _kmeans(vectors, k=10, seed=0)
    assert len(labels) == 1


def test_kmeans_label_count():
    docs = [["machine", "learning"], ["deep", "network"], ["cluster", "kmeans"]]
    vectors, _ = _tfidf(docs)
    labels = _kmeans(vectors, k=2, seed=42)
    assert len(labels) == 3
    assert set(labels) <= {0, 1}


def test_kmeans_reproducible():
    docs = [
        ["machine", "learning", "neural"],
        ["clustering", "kmeans", "unsupervised"],
        ["deep", "network", "convolutional"],
        ["text", "classification", "sentiment"],
    ]
    vectors, _ = _tfidf(docs)
    l1 = _kmeans(vectors, k=2, seed=7)
    l2 = _kmeans(vectors, k=2, seed=7)
    assert l1 == l2


# ---------------------------------------------------------------------------
# _bibtex_field
# ---------------------------------------------------------------------------

def test_bibtex_field_braces():
    entry = 'title = {Deep Learning Methods},'
    assert _bibtex_field(entry, 'title') == 'Deep Learning Methods'


def test_bibtex_field_quotes():
    entry = 'title = "Quoted Title",'
    assert _bibtex_field(entry, 'title') == 'Quoted Title'


def test_bibtex_field_numeric():
    entry = 'year = 2024,'
    assert _bibtex_field(entry, 'year') == '2024'


def test_bibtex_field_nested_braces():
    entry = 'title = {A {LaTeX} Title},'
    assert _bibtex_field(entry, 'title') == 'A {LaTeX} Title'


def test_bibtex_field_missing():
    assert _bibtex_field('author = {Smith}', 'title') == ''


def test_bibtex_field_case_insensitive():
    entry = 'TITLE = {Hello},'
    assert _bibtex_field(entry, 'title') == 'Hello'


# ---------------------------------------------------------------------------
# LitCluster — construction & validation
# ---------------------------------------------------------------------------

def test_litcluster_invalid_k():
    with pytest.raises(ValueError, match="k must be"):
        LitCluster(k=0)


def test_litcluster_invalid_max_iter():
    with pytest.raises(ValueError, match="max_iter"):
        LitCluster(max_iter=0)


def test_litcluster_invalid_min_term_freq():
    with pytest.raises(ValueError, match="min_term_freq"):
        LitCluster(min_term_freq=0)


def test_litcluster_repr():
    obj = LitCluster(k=3)
    assert "LitCluster" in repr(obj)
    assert "k=3" in repr(obj)


# ---------------------------------------------------------------------------
# LitCluster — add_paper & fit
# ---------------------------------------------------------------------------

def _make_papers():
    return [
        Paper("1", "Machine Learning Survey",
              "Overview of supervised and unsupervised machine learning algorithms."),
        Paper("2", "Deep Neural Networks",
              "Neural network architectures for image classification and object detection."),
        Paper("3", "K-means Clustering Algorithm",
              "Clustering methods for unsupervised data grouping and partitioning."),
        Paper("4", "Unsupervised Clustering Techniques",
              "Approaches for high-dimensional data clustering and analysis."),
        Paper("5", "Natural Language Processing",
              "Text classification and sentiment analysis using transformer models."),
        Paper("6", "Transformer Language Models",
              "Large language model fine-tuning for downstream NLP tasks."),
    ]


def test_add_paper_returns_self():
    obj = LitCluster(k=2, min_term_freq=1)
    p = Paper("1", "Test")
    result = obj.add_paper(p)
    assert result is obj


def test_add_paper_appends():
    obj = LitCluster(k=2, min_term_freq=1)
    obj.add_paper(Paper("1", "A"))
    obj.add_paper(Paper("2", "B"))
    assert len(obj.papers) == 2


def test_fit_produces_clusters():
    obj = LitCluster(k=3, seed=42, min_term_freq=1)
    for p in _make_papers():
        obj.add_paper(p)
    obj.fit()
    assert len(obj.clusters) == 3


def test_fit_all_papers_assigned():
    papers = _make_papers()
    obj = LitCluster(k=2, seed=42, min_term_freq=1)
    for p in papers:
        obj.add_paper(p)
    obj.fit()
    total = sum(len(c.papers) for c in obj.clusters)
    assert total == len(papers)


def test_fit_top_terms_populated():
    obj = LitCluster(k=2, seed=42, min_term_freq=1)
    for p in _make_papers():
        obj.add_paper(p)
    obj.fit()
    for c in obj.clusters:
        assert len(c.top_terms) > 0


def test_fit_empty_corpus():
    obj = LitCluster(k=5)
    obj.fit()
    assert obj.clusters == []


def test_fit_warns_on_empty_token_lists():
    """Papers filtered to empty token lists should trigger a UserWarning."""
    obj = LitCluster(k=2, min_term_freq=100)  # freq threshold too high
    for p in _make_papers():
        obj.add_paper(p)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        obj.fit()
    assert any(issubclass(warning.category, UserWarning) for warning in w)


def test_fit_method_chaining():
    obj = LitCluster(k=2, seed=0, min_term_freq=1)
    for p in _make_papers():
        obj.add_paper(p)
    result = obj.fit()
    assert result is obj


def test_fit_k_greater_than_n():
    obj = LitCluster(k=20, seed=42, min_term_freq=1)
    obj.add_paper(Paper("1", "Only paper", abstract="Single document test."))
    obj.fit()
    assert len(obj.clusters) == 1


# ---------------------------------------------------------------------------
# LitCluster — summary
# ---------------------------------------------------------------------------

def test_summary_before_fit():
    obj = LitCluster()
    s = obj.summary()
    assert "fit" in s.lower() or "cluster" in s.lower()


def test_summary_after_fit():
    obj = LitCluster(k=2, seed=42, min_term_freq=1)
    for p in _make_papers():
        obj.add_paper(p)
    obj.fit()
    s = obj.summary()
    assert "papers" in s.lower()
    assert "clusters" in s.lower()
    assert str(len(_make_papers())) in s


# ---------------------------------------------------------------------------
# LitCluster — file loaders
# ---------------------------------------------------------------------------

def test_from_csv(tmp_path):
    csv_file = tmp_path / "papers.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["paper_id", "title", "abstract", "year"]
        )
        writer.writeheader()
        writer.writerow({"paper_id": "1", "title": "Hello", "abstract": "World", "year": "2024"})
        writer.writerow({"paper_id": "2", "title": "Foo", "abstract": "Bar", "year": "2023"})

    obj = LitCluster.from_csv(csv_file)
    assert len(obj.papers) == 2
    assert obj.papers[0].title == "Hello"
    assert obj.papers[1].year == "2023"


def test_from_csv_missing_optional_columns(tmp_path):
    csv_file = tmp_path / "minimal.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["title"])
        writer.writeheader()
        writer.writerow({"title": "Minimal Paper"})

    obj = LitCluster.from_csv(csv_file)
    assert len(obj.papers) == 1
    assert obj.papers[0].abstract == ""


def test_from_jsonl(tmp_path):
    jsonl_file = tmp_path / "papers.jsonl"
    jsonl_file.write_text(
        json.dumps({"paper_id": "a", "title": "Alpha", "abstract": "First."}) + "\n"
        + json.dumps({"paper_id": "b", "title": "Beta",  "abstract": "Second."}) + "\n"
        + "\n",  # blank line should be skipped
        encoding="utf-8",
    )
    obj = LitCluster.from_jsonl(jsonl_file)
    assert len(obj.papers) == 2
    assert obj.papers[0].title == "Alpha"


def test_from_bibtex(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text(
        "@article{smith2024,\n"
        "  title  = {Deep Learning for Science},\n"
        "  abstract = {We propose a neural approach.},\n"
        "  author = {Smith, Jane},\n"
        "  year   = {2024},\n"
        "  journal = {Nature},\n"
        "}\n",
        encoding="utf-8",
    )
    obj = LitCluster.from_bibtex(bib)
    assert len(obj.papers) == 1
    assert obj.papers[0].title == "Deep Learning for Science"
    assert obj.papers[0].year == "2024"
    assert obj.papers[0].venue == "Nature"
    assert obj.papers[0].authors == "Smith, Jane"


def test_from_bibtex_multiple_entries(tmp_path):
    bib = tmp_path / "multi.bib"
    bib.write_text(
        "@article{a,title={Alpha},year={2020}}\n"
        "@inproceedings{b,title={Beta},year={2021}}\n",
        encoding="utf-8",
    )
    obj = LitCluster.from_bibtex(bib)
    assert len(obj.papers) == 2


def test_from_bibtex_nested_braces(tmp_path):
    bib = tmp_path / "nested.bib"
    bib.write_text(
        "@article{x,\n"
        "  title = {A {Nested} Title},\n"
        "  year = 2022,\n"
        "}\n",
        encoding="utf-8",
    )
    obj = LitCluster.from_bibtex(bib)
    assert "Nested" in obj.papers[0].title


# ---------------------------------------------------------------------------
# LitCluster — exports
# ---------------------------------------------------------------------------

def test_export_csv(tmp_path):
    obj = LitCluster(k=2, seed=42, min_term_freq=1)
    for p in _make_papers():
        obj.add_paper(p)
    obj.fit()

    out = tmp_path / "results.csv"
    obj.export_csv(out)
    assert out.exists()

    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(_make_papers())
    assert "cluster_id" in rows[0]
    assert "title" in rows[0]


def test_export_json(tmp_path):
    obj = LitCluster(k=2, seed=42, min_term_freq=1)
    for p in _make_papers():
        obj.add_paper(p)
    obj.fit()

    out = tmp_path / "results.json"
    obj.export_json(out)
    assert out.exists()

    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 2
    for cluster in data:
        assert "cluster_id" in cluster
        assert "papers" in cluster
        assert "top_terms" in cluster


def test_export_csv_path_accepts_string(tmp_path):
    obj = LitCluster(k=2, seed=0, min_term_freq=1)
    for p in _make_papers():
        obj.add_paper(p)
    obj.fit()
    out = str(tmp_path / "str_path.csv")
    obj.export_csv(out)
    assert pathlib.Path(out).exists()
