"""
Tests for litcluster — covers text processing, data structures, I/O,
the full clustering pipeline, and all export formats.
"""

import csv
import json
import pathlib
import sys

import pytest

# Resolve litcluster.py from the repository root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import litcluster as lc_module
from litcluster import (
    Cluster,
    LitCluster,
    Paper,
    _bibtex_field,
    _cosine,
    _kmeans,
    _tfidf,
    _tokenise,
)


# ---------------------------------------------------------------------------
# _tokenise
# ---------------------------------------------------------------------------

def test_tokenise_returns_list():
    assert isinstance(_tokenise("hello world test"), list)


def test_tokenise_lowercases():
    tokens = _tokenise("Machine Learning Neural")
    assert all(t == t.lower() for t in tokens)


def test_tokenise_removes_stopwords():
    tokens = _tokenise("the quick brown fox")
    assert "the" not in tokens


def test_tokenise_min_length_three():
    tokens = _tokenise("a ab abc abcd")
    assert "a" not in tokens
    assert "ab" not in tokens
    assert "abc" in tokens
    assert "abcd" in tokens


def test_tokenise_empty_string():
    assert _tokenise("") == []


def test_tokenise_non_alpha_ignored():
    tokens = _tokenise("123 !@# hello")
    assert "123" not in tokens
    assert "hello" in tokens


# ---------------------------------------------------------------------------
# _tfidf
# ---------------------------------------------------------------------------

def test_tfidf_empty_input():
    vectors, vocab = _tfidf([])
    assert vectors == []
    assert vocab == []


def test_tfidf_single_doc():
    vectors, vocab = _tfidf([["machine", "learning"]])
    assert len(vectors) == 1
    assert "machine" in vectors[0]
    assert "learning" in vectors[0]


def test_tfidf_two_docs():
    docs = [["machine", "learning", "neural"], ["biology", "protein", "cell"]]
    vectors, vocab = _tfidf(docs)
    assert len(vectors) == 2
    assert len(vocab) == 6


def test_tfidf_weights_positive():
    vectors, _ = _tfidf([["deep", "learning"], ["protein", "folding"]])
    for vec in vectors:
        assert all(v > 0 for v in vec.values())


def test_tfidf_rare_terms_get_high_idf():
    # Both terms appear once in doc[0] (equal TF), but "unique" is absent
    # from all other docs so IDF drives its score higher than "common".
    docs = [["unique", "common"]] + [["common", "other"]] * 9
    vectors, _ = _tfidf(docs)
    assert vectors[0].get("unique", 0) > vectors[0].get("common", 0)


def test_tfidf_empty_doc_gives_empty_vector():
    vectors, vocab = _tfidf([["alpha", "beta"], []])
    assert vectors[1] == {}


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------

def test_cosine_identical_vectors():
    v = {"a": 1.0, "b": 2.0}
    assert abs(_cosine(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal_vectors():
    assert _cosine({"x": 1.0}, {"y": 1.0}) == 0.0


def test_cosine_empty_first():
    assert _cosine({}, {"a": 1.0}) == 0.0


def test_cosine_empty_second():
    assert _cosine({"a": 1.0}, {}) == 0.0


def test_cosine_both_empty():
    assert _cosine({}, {}) == 0.0


def test_cosine_range():
    a = {"a": 1.0, "b": 0.5}
    b = {"a": 0.5, "c": 1.0}
    sim = _cosine(a, b)
    assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# _kmeans
# ---------------------------------------------------------------------------

def test_kmeans_empty_input():
    assert _kmeans([], k=3) == []


def test_kmeans_single_vector():
    labels = _kmeans([{"a": 1.0}], k=3)
    assert labels == [0]


def test_kmeans_k_clamped_to_n():
    vecs = [{"a": float(i)} for i in range(3)]
    labels = _kmeans(vecs, k=100)
    assert len(labels) == 3
    assert all(0 <= l < 3 for l in labels)


def test_kmeans_reproducible():
    vecs = [{"a": float(i), "b": float(i % 3)} for i in range(10)]
    l1 = _kmeans(vecs, k=3, seed=0)
    l2 = _kmeans(vecs, k=3, seed=0)
    assert l1 == l2


def test_kmeans_different_seeds_may_differ():
    vecs = [{"a": float(i)} for i in range(10)]
    l1 = _kmeans(vecs, k=3, seed=0)
    l2 = _kmeans(vecs, k=3, seed=99)
    # Not guaranteed but highly likely with different seeds
    # (just ensure they run without error)
    assert len(l1) == len(l2) == 10


def test_kmeans_covers_all_indices():
    vecs = [{"a": float(i)} for i in range(6)]
    labels = _kmeans(vecs, k=3, seed=42)
    assert len(labels) == 6
    assert all(isinstance(l, int) for l in labels)


# ---------------------------------------------------------------------------
# _bibtex_field
# ---------------------------------------------------------------------------

def test_bibtex_field_braces():
    entry = "@article{key, title = {Hello World}, year = {2023}}"
    assert _bibtex_field(entry, "title") == "Hello World"
    assert _bibtex_field(entry, "year") == "2023"


def test_bibtex_field_double_quotes():
    entry = '@article{key, title = "Quoted Title"}'
    assert _bibtex_field(entry, "title") == "Quoted Title"


def test_bibtex_field_nested_braces():
    entry = "@article{key, title = {Advances in {Machine Learning}}}"
    result = _bibtex_field(entry, "title")
    # Should include the inner braces content
    assert "Machine Learning" in result
    assert "Advances" in result


def test_bibtex_field_numeric():
    entry = "@article{key, year = 2024}"
    assert _bibtex_field(entry, "year") == "2024"


def test_bibtex_field_missing_returns_empty():
    entry = "@article{key, title = {Test}}"
    assert _bibtex_field(entry, "abstract") == ""


def test_bibtex_field_case_insensitive():
    entry = "@article{key, TITLE = {Upper}}"
    assert _bibtex_field(entry, "title") == "Upper"


# ---------------------------------------------------------------------------
# Paper dataclass
# ---------------------------------------------------------------------------

def test_paper_text_combines_fields():
    p = Paper(paper_id="1", title="Neural Networks", abstract="Deep learning.", keywords="ML")
    assert "Neural Networks" in p.text
    assert "Deep learning." in p.text
    assert "ML" in p.text


def test_paper_text_skips_empty_fields():
    p = Paper(paper_id="1", title="Title", abstract="", keywords="")
    assert p.text == "Title"


def test_paper_to_dict_keys():
    p = Paper(paper_id="x", title="T", abstract="A")
    d = p.to_dict()
    for key in ("paper_id", "title", "abstract", "authors", "year", "venue", "doi", "keywords"):
        assert key in d


# ---------------------------------------------------------------------------
# Cluster dataclass
# ---------------------------------------------------------------------------

def test_cluster_label_includes_top_terms():
    c = Cluster(cluster_id=0, top_terms=["neural", "network", "deep"])
    assert "neural" in c.label
    assert "0" in c.label


def test_cluster_label_no_terms():
    c = Cluster(cluster_id=1)
    assert "1" in c.label
    assert "—" in c.label


def test_cluster_to_dict_structure():
    c = Cluster(cluster_id=2, papers=[], top_terms=["alpha", "beta"])
    d = c.to_dict()
    assert d["cluster_id"] == 2
    assert d["size"] == 0
    assert d["top_terms"] == ["alpha", "beta"]
    assert "label" in d
    assert "papers" in d


# ---------------------------------------------------------------------------
# Helpers for integration tests
# ---------------------------------------------------------------------------

def _make_papers(n: int = 10) -> list[Paper]:
    """Return a list of synthetic papers with distinct topic vocabulary."""
    corpus = [
        ("Neural networks for image classification using deep learning "
         "convolutional architectures and backpropagation training",
         "deep learning neural convolutional"),
        ("Protein folding prediction using molecular dynamics simulation "
         "and energy minimization algorithms",
         "protein folding biology molecular"),
        ("Climate change impact on biodiversity and ecosystem services "
         "in tropical rainforests",
         "climate ecosystem biodiversity ecology"),
        ("Quantum computing algorithms for combinatorial optimization "
         "using variational circuits",
         "quantum computing algorithm optimization"),
        ("Genome sequencing analysis using bioinformatics pipelines "
         "and variant calling workflows",
         "genome sequencing bioinformatics variant"),
        ("Autonomous robot navigation using reinforcement learning "
         "and sensor fusion techniques",
         "robot navigation reinforcement sensor"),
        ("Solar energy photovoltaic cell efficiency improvements "
         "using perovskite materials",
         "solar photovoltaic energy efficiency"),
        ("Cancer detection using convolutional neural networks "
         "on histopathology images",
         "cancer detection neural histopathology"),
        ("Natural language processing for abstractive summarization "
         "using transformer architectures",
         "language processing transformer summarization"),
        ("Novel battery materials for energy storage applications "
         "using lithium-ion chemistry",
         "battery energy storage materials lithium"),
    ]
    papers = []
    for i, (text, kw) in enumerate(corpus[:n]):
        papers.append(Paper(
            paper_id=str(i),
            title=text[:60],
            abstract=text,
            keywords=kw,
            authors=f"Author {i}",
            year=str(2020 + i % 5),
        ))
    return papers


def _make_lc(n: int = 10, k: int = 3) -> LitCluster:
    obj = LitCluster(k=k, min_term_freq=1, seed=0)
    obj.papers = _make_papers(n)
    return obj


# ---------------------------------------------------------------------------
# LitCluster — fit()
# ---------------------------------------------------------------------------

def test_fit_populates_clusters():
    lc = _make_lc()
    lc.fit()
    assert len(lc.clusters) > 0


def test_fit_all_papers_assigned():
    lc = _make_lc()
    lc.fit()
    total = sum(len(c.papers) for c in lc.clusters)
    assert total == len(lc.papers)


def test_fit_clusters_have_top_terms():
    lc = _make_lc()
    lc.fit()
    for c in lc.clusters:
        assert len(c.top_terms) > 0


def test_fit_returns_self():
    lc = _make_lc()
    assert lc.fit() is lc


def test_fit_empty_corpus_returns_self():
    lc = LitCluster()
    result = lc.fit()
    assert result is lc
    assert lc.clusters == []


def test_fit_k_greater_than_papers():
    lc = LitCluster(k=100, min_term_freq=1, seed=0)
    lc.papers = _make_papers(3)
    lc.fit()
    # k is clamped to n; all papers assigned
    assert sum(len(c.papers) for c in lc.clusters) == 3


def test_fit_min_term_freq_filters_vocabulary():
    lc = LitCluster(k=2, min_term_freq=5, seed=0)
    lc.papers = _make_papers(10)
    lc.fit()
    # With high min_term_freq some rare terms may be absent from top_terms
    # but clustering should still complete
    assert len(lc.clusters) > 0


# ---------------------------------------------------------------------------
# LitCluster — summary()
# ---------------------------------------------------------------------------

def test_summary_before_fit():
    lc = _make_lc()
    s = lc.summary()
    assert "fit()" in s


def test_summary_after_fit():
    lc = _make_lc()
    lc.fit()
    s = lc.summary()
    assert "papers" in s
    assert "clusters" in s


def test_summary_lists_cluster_ids():
    lc = _make_lc(k=2)
    lc.fit()
    s = lc.summary()
    assert "[0]" in s or "[1]" in s


# ---------------------------------------------------------------------------
# LitCluster — loaders
# ---------------------------------------------------------------------------

def test_from_csv(tmp_path):
    p = tmp_path / "papers.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["paper_id", "title", "abstract", "authors",
                    "year", "venue", "doi", "keywords"])
        w.writerow(["1", "Deep Learning", "Neural networks.", "Smith", "2020",
                    "NeurIPS", "10.1/ex", "ML"])
        w.writerow(["2", "Protein Folding", "Biology methods.", "Jones", "2021",
                    "Nature", "", "bio"])
    obj = LitCluster.from_csv(p, k=2, min_term_freq=1)
    assert len(obj.papers) == 2
    assert obj.papers[0].paper_id == "1"
    assert obj.papers[0].doi == "10.1/ex"


def test_from_csv_missing_columns_default_empty(tmp_path):
    p = tmp_path / "minimal.csv"
    with p.open("w", newline="") as fh:
        csv.writer(fh).writerows([["title"], ["Only Title"]])
    obj = LitCluster.from_csv(p, k=2, min_term_freq=1)
    assert len(obj.papers) == 1
    assert obj.papers[0].abstract == ""


def test_from_jsonl(tmp_path):
    p = tmp_path / "papers.jsonl"
    with p.open("w") as fh:
        fh.write(json.dumps({"title": "Test paper", "abstract": "Abstract text."}) + "\n")
        fh.write("\n")  # blank line should be skipped
        fh.write(json.dumps({"title": "Another paper", "abstract": "More text."}) + "\n")
    obj = LitCluster.from_jsonl(p, k=2, min_term_freq=1)
    assert len(obj.papers) == 2


def test_from_bibtex(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text(
        "@article{smith2020,\n"
        "  title = {Deep Learning for Vision},\n"
        "  abstract = {Neural network architecture.},\n"
        "  author = {Smith, John},\n"
        "  year = {2020},\n"
        "  journal = {CVPR},\n"
        "}\n"
        "@inproceedings{jones2021,\n"
        "  title = {Protein Structure Prediction},\n"
        "  abstract = {Biological sequence analysis.},\n"
        "  author = {Jones, Alice},\n"
        "  year = {2021},\n"
        "  booktitle = {NeurIPS},\n"
        "}\n",
        encoding="utf-8",
    )
    obj = LitCluster.from_bibtex(bib, k=2, min_term_freq=1)
    assert len(obj.papers) == 2
    assert obj.papers[0].title == "Deep Learning for Vision"
    assert obj.papers[1].authors == "Jones, Alice"
    assert obj.papers[1].venue == "NeurIPS"


def test_from_bibtex_nested_braces(tmp_path):
    bib = tmp_path / "nested.bib"
    bib.write_text(
        "@article{k1,\n"
        "  title = {Advances in {Machine Learning}},\n"
        "  year = {2023},\n"
        "}\n",
        encoding="utf-8",
    )
    obj = LitCluster.from_bibtex(bib, k=1, min_term_freq=1)
    assert len(obj.papers) == 1
    assert "Machine Learning" in obj.papers[0].title


def test_from_bibtex_booktitle_fallback(tmp_path):
    bib = tmp_path / "conf.bib"
    bib.write_text(
        "@inproceedings{conf1,\n"
        "  title = {Conference Paper},\n"
        "  booktitle = {ICML 2024},\n"
        "  year = {2024},\n"
        "}\n",
        encoding="utf-8",
    )
    obj = LitCluster.from_bibtex(bib, k=1, min_term_freq=1)
    assert obj.papers[0].venue == "ICML 2024"


# ---------------------------------------------------------------------------
# LitCluster — exports
# ---------------------------------------------------------------------------

def test_export_csv(tmp_path):
    lc = _make_lc()
    lc.fit()
    out = tmp_path / "out.csv"
    lc.export_csv(out)
    assert out.exists()
    with out.open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0][0] == "cluster_id"
    assert len(rows) == len(lc.papers) + 1  # header + one row per paper


def test_export_csv_contains_all_papers(tmp_path):
    lc = _make_lc()
    lc.fit()
    out = tmp_path / "all.csv"
    lc.export_csv(out)
    with out.open() as fh:
        data_rows = list(csv.DictReader(fh))
    paper_ids_out = {r["paper_id"] for r in data_rows}
    paper_ids_in = {p.paper_id for p in lc.papers}
    assert paper_ids_out == paper_ids_in


def test_export_json(tmp_path):
    lc = _make_lc()
    lc.fit()
    out = tmp_path / "out.json"
    lc.export_json(out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert len(data) == len(lc.clusters)
    for entry in data:
        assert "cluster_id" in entry
        assert "papers" in entry
        assert "top_terms" in entry


def test_export_html(tmp_path):
    lc = _make_lc()
    lc.fit()
    out = tmp_path / "out.html"
    lc.export_html(out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "litcluster" in content
    # Each cluster should appear
    for c in lc.clusters:
        assert str(c.cluster_id) in content


def test_export_html_contains_paper_titles(tmp_path):
    lc = _make_lc(n=5)
    lc.fit()
    out = tmp_path / "titles.html"
    lc.export_html(out)
    content = out.read_text(encoding="utf-8")
    for p in lc.papers:
        assert p.title[:20] in content


def test_export_html_special_chars_escaped(tmp_path):
    lc = LitCluster(k=1, min_term_freq=1, seed=0)
    lc.papers = [Paper(
        paper_id="x",
        title='<script>alert("xss")</script>',
        abstract="Safe abstract content here.",
    )]
    lc.fit()
    out = tmp_path / "xss.html"
    lc.export_html(out)
    content = out.read_text(encoding="utf-8")
    assert "<script>alert" not in content
    assert "&lt;script&gt;" in content


# ---------------------------------------------------------------------------
# Module-level attributes (JOSS requirement: importable public API)
# ---------------------------------------------------------------------------

def test_module_version():
    assert hasattr(lc_module, "__version__")
    assert isinstance(lc_module.__version__, str)


def test_module_exports_litcluster():
    assert hasattr(lc_module, "LitCluster")


def test_module_exports_paper():
    assert hasattr(lc_module, "Paper")


def test_module_exports_cluster():
    assert hasattr(lc_module, "Cluster")


def test_module_exports_tokenise():
    assert hasattr(lc_module, "_tokenise")
