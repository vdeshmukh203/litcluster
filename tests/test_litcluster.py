"""
Comprehensive test suite for litcluster.
Covers tokenisation, TF-IDF, cosine similarity, k-means, BibTeX parsing,
data structures, loaders, fit(), exporters, and edge cases.
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys
import tempfile
import textwrap

# Ensure repo root is importable
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import litcluster as lc_mod
from litcluster import (
    LitCluster,
    Paper,
    Cluster,
    _tokenise,
    _tfidf,
    _cosine,
    _kmeans,
    _parse_bibtex_field,
)

# ---------------------------------------------------------------------------
# Fixtures / shared data
# ---------------------------------------------------------------------------

SAMPLE_BIB = textwrap.dedent("""\
    @article{smith2023deep,
      author   = {Smith, John and Doe, Jane},
      title    = {Deep Learning for Natural Language Processing},
      abstract = {We study deep neural networks for text classification tasks.},
      year     = {2023},
      journal  = {Machine Learning Journal},
      doi      = {10.1234/mlj.2023},
      keywords = {deep learning, neural networks, NLP},
    }
    @article{jones2023graph,
      author   = {Jones, Alice},
      title    = {Graph Neural Networks in Computer Vision},
      abstract = {We apply graph neural networks to image recognition tasks.},
      year     = {2023},
      journal  = {Vision Conference},
      keywords = {graph neural networks, computer vision, images},
    }
    @article{lee2022transformers,
      author   = {Lee, Bob},
      title    = {Transformers for Machine Translation},
      abstract = {Transformer architectures achieve state of the art results.},
      year     = {2022},
      journal  = {NLP Symposium},
      keywords = {transformer, machine translation, attention},
    }
    @comment{this should be skipped entirely}
    @string{conf = {Some Conference}}
""")

SAMPLE_CSV = (
    "paper_id,title,abstract,authors,year,venue,doi,keywords\n"
    "p1,Deep Learning,Study of neural networks,Smith J.,2023,MLJ,,"
    "deep learning\n"
    "p2,NLP Methods,Text processing and classification,Jones A.,2022,ACL,,"
    "nlp text\n"
    "p3,Vision Models,Image classification with convolutional networks,Lee B.,"
    "2021,CVPR,,computer vision\n"
)

SAMPLE_JSONL = (
    '{"paper_id":"p1","title":"Deep Learning",'
    '"abstract":"Neural network study for text","year":"2023"}\n'
    '{"paper_id":"p2","title":"NLP Research",'
    '"abstract":"Text classification methods","year":"2022"}\n'
    '{"paper_id":"p3","title":"Computer Vision",'
    '"abstract":"Image recognition using deep convolutional models","year":"2021"}\n'
)


def _bib_tmpfile():
    """Write SAMPLE_BIB to a temp file and return its Path."""
    f = tempfile.NamedTemporaryFile(
        suffix=".bib", mode="w", delete=False, encoding="utf-8"
    )
    f.write(SAMPLE_BIB)
    f.close()
    return pathlib.Path(f.name)


def _csv_tmpfile():
    f = tempfile.NamedTemporaryFile(
        suffix=".csv", mode="w", delete=False, encoding="utf-8"
    )
    f.write(SAMPLE_CSV)
    f.close()
    return pathlib.Path(f.name)


def _jsonl_tmpfile():
    f = tempfile.NamedTemporaryFile(
        suffix=".jsonl", mode="w", delete=False, encoding="utf-8"
    )
    f.write(SAMPLE_JSONL)
    f.close()
    return pathlib.Path(f.name)


# ===========================================================================
# 1. Tokenisation
# ===========================================================================

def test_tokenise_returns_list():
    assert isinstance(_tokenise("Hello World Test"), list)


def test_tokenise_lowercases():
    tokens = _tokenise("Neural Networks Deep")
    assert "neural" in tokens
    assert "networks" in tokens
    assert "deep" in tokens


def test_tokenise_removes_stopwords():
    tokens = _tokenise("the and or but with for of")
    assert tokens == []


def test_tokenise_min_length_three():
    tokens = _tokenise("a ab abc abcd")
    assert "a" not in tokens
    assert "ab" not in tokens
    assert "abc" in tokens
    assert "abcd" in tokens


def test_tokenise_only_alpha():
    # Regex r"[a-zA-Z]{3,}" extracts the alpha run from mixed strings
    tokens = _tokenise("word123 abc! test_case")
    assert "word" in tokens    # alpha run extracted from "word123"
    assert "abc" in tokens     # alpha run from "abc!"
    assert "test" in tokens    # first run from "test_case"
    assert "case" in tokens    # second run from "test_case"


def test_tokenise_empty_string():
    assert _tokenise("") == []


def test_tokenise_non_alpha_filtered():
    tokens = _tokenise("123 456 !@#")
    assert tokens == []


# ===========================================================================
# 2. TF-IDF
# ===========================================================================

def test_tfidf_empty_input():
    vecs, vocab = _tfidf([])
    assert vecs == []
    assert vocab == []


def test_tfidf_single_document():
    vecs, vocab = _tfidf([["machine", "learning"]])
    assert len(vecs) == 1
    assert set(vocab) == {"learning", "machine"}
    assert vecs[0]["machine"] > 0
    assert vecs[0]["learning"] > 0


def test_tfidf_multiple_documents():
    docs = [["neural", "networks"], ["deep", "learning"], ["neural", "deep"]]
    vecs, vocab = _tfidf(docs)
    assert len(vecs) == 3
    assert len(vocab) == 4
    assert vecs[0]["neural"] > 0
    assert vecs[1]["deep"] > 0


def test_tfidf_vectors_are_sparse_dicts():
    vecs, _ = _tfidf([["alpha"], ["beta"]])
    assert isinstance(vecs[0], dict)
    # Each doc only has its own term
    assert "alpha" in vecs[0]
    assert "beta" not in vecs[0]


def test_tfidf_rare_term_has_higher_idf():
    # "rare" appears in 1 of 3 docs; "common" appears in all 3
    docs = [
        ["common", "rare"],
        ["common"],
        ["common"],
    ]
    vecs, _ = _tfidf(docs)
    assert vecs[0]["rare"] > vecs[0]["common"]


def test_tfidf_empty_token_list_per_doc():
    vecs, vocab = _tfidf([[], ["word"]])
    assert len(vecs) == 2
    assert vecs[0] == {}
    assert "word" in vecs[1]


# ===========================================================================
# 3. Cosine similarity
# ===========================================================================

def test_cosine_identical_vectors():
    v = {"a": 1.0, "b": 2.0}
    assert abs(_cosine(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal_vectors():
    assert _cosine({"a": 1.0}, {"b": 1.0}) == 0.0


def test_cosine_empty_vectors():
    assert _cosine({}, {"a": 1.0}) == 0.0
    assert _cosine({"a": 1.0}, {}) == 0.0
    assert _cosine({}, {}) == 0.0


def test_cosine_in_unit_interval():
    v1 = {"a": 0.8, "b": 0.5}
    v2 = {"a": 0.3, "b": 0.9, "c": 0.1}
    sim = _cosine(v1, v2)
    assert 0.0 <= sim <= 1.0


def test_cosine_symmetry():
    v1 = {"x": 1.0, "y": 2.0}
    v2 = {"x": 3.0, "z": 1.0}
    assert abs(_cosine(v1, v2) - _cosine(v2, v1)) < 1e-12


# ===========================================================================
# 4. K-means
# ===========================================================================

def test_kmeans_empty():
    assert _kmeans([], k=3) == []


def test_kmeans_single_vector():
    labels = _kmeans([{"a": 1.0}], k=1)
    assert labels == [0]


def test_kmeans_k_gt_n_capped():
    vecs = [{"a": 1.0}, {"b": 1.0}]
    labels = _kmeans(vecs, k=10)
    assert len(labels) == 2
    assert all(0 <= lbl < 2 for lbl in labels)


def test_kmeans_labels_count_matches_input():
    vecs = [{"a": float(i)} for i in range(8)]
    labels = _kmeans(vecs, k=3)
    assert len(labels) == 8


def test_kmeans_labels_in_valid_range():
    vecs = [{"a": float(i), "b": float(i % 2)} for i in range(12)]
    labels = _kmeans(vecs, k=4)
    assert all(0 <= lbl < 4 for lbl in labels)


def test_kmeans_deterministic_with_seed():
    vecs = [{"a": float(i), "b": float(i * 3)} for i in range(20)]
    assert _kmeans(vecs, k=3, seed=7) == _kmeans(vecs, k=3, seed=7)


def test_kmeans_different_seeds_may_differ():
    vecs = [{"t": float(i)} for i in range(30)]
    l1 = _kmeans(vecs, k=5, seed=1)
    l2 = _kmeans(vecs, k=5, seed=999)
    # Can't guarantee they differ, but structure must be valid
    assert len(l1) == len(l2) == 30


# ===========================================================================
# 5. BibTeX field parser
# ===========================================================================

def test_parse_brace_field():
    entry = "title = {Neural Networks and Deep Learning},"
    assert _parse_bibtex_field(entry, "title") == "Neural Networks and Deep Learning"


def test_parse_quote_field():
    entry = 'year = "2023",'
    assert _parse_bibtex_field(entry, "year") == "2023"


def test_parse_bare_number():
    entry = "year = 2023,"
    assert _parse_bibtex_field(entry, "year") == "2023"


def test_parse_nested_braces():
    entry = "title = {A Study of {NLP} Methods},"
    result = _parse_bibtex_field(entry, "title")
    assert result == "A Study of {NLP} Methods"


def test_parse_missing_field_returns_empty():
    entry = "author = {Smith, John},"
    assert _parse_bibtex_field(entry, "title") == ""


def test_parse_multiline_abstract():
    entry = "abstract = {This is a long\n  multiline\n  abstract.},"
    result = _parse_bibtex_field(entry, "abstract")
    assert "multiline" in result
    assert "abstract" in result


def test_parse_case_insensitive():
    entry = "TITLE = {Case Test},"
    assert _parse_bibtex_field(entry, "title") == "Case Test"


# ===========================================================================
# 6. Paper dataclass
# ===========================================================================

def test_paper_text_concatenation():
    p = Paper(paper_id="1", title="Deep Learning",
               abstract="Neural nets.", keywords="ml ai")
    assert "Deep Learning" in p.text
    assert "Neural nets." in p.text
    assert "ml ai" in p.text


def test_paper_to_dict_keys():
    p = Paper(paper_id="p1", title="T", abstract="A", year="2023")
    d = p.to_dict()
    for key in ("paper_id", "title", "abstract", "authors", "year",
                "venue", "doi", "keywords"):
        assert key in d


def test_paper_repr():
    p = Paper(paper_id="x", title="Some Title")
    assert "Some Title" in repr(p)


def test_paper_defaults():
    p = Paper(paper_id="1", title="T")
    assert p.abstract == ""
    assert p.doi == ""


# ===========================================================================
# 7. Cluster dataclass
# ===========================================================================

def test_cluster_label_contains_id():
    c = Cluster(cluster_id=3, top_terms=["alpha", "beta"])
    assert "3" in c.label
    assert "alpha" in c.label


def test_cluster_label_empty_terms():
    c = Cluster(cluster_id=0)
    assert "Cluster 0" in c.label


def test_cluster_to_dict_structure():
    p = Paper(paper_id="p1", title="T")
    c = Cluster(cluster_id=2, papers=[p], top_terms=["x", "y"])
    d = c.to_dict()
    assert d["cluster_id"] == 2
    assert d["size"] == 1
    assert len(d["papers"]) == 1
    assert d["top_terms"] == ["x", "y"]


def test_cluster_repr():
    c = Cluster(cluster_id=1, papers=[Paper("p1", "T")], top_terms=["a"])
    r = repr(c)
    assert "1" in r
    assert "size=1" in r


# ===========================================================================
# 8. LitCluster construction and validation
# ===========================================================================

def test_litcluster_invalid_k():
    raised = False
    try:
        LitCluster(k=0)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for k=0"


def test_litcluster_invalid_max_iter():
    raised = False
    try:
        LitCluster(max_iter=0)
    except ValueError:
        raised = True
    assert raised


def test_litcluster_invalid_min_term_freq():
    raised = False
    try:
        LitCluster(min_term_freq=0)
    except ValueError:
        raised = True
    assert raised


def test_litcluster_defaults():
    lc = LitCluster()
    assert lc.k == 5
    assert lc.seed == 42
    assert lc.max_iter == 100
    assert lc.min_term_freq == 2


# ===========================================================================
# 9. Paper management methods
# ===========================================================================

def test_add_paper():
    lc = LitCluster(k=1)
    lc.add_paper(Paper("p1", "Test"))
    assert len(lc.papers) == 1


def test_add_paper_clears_clusters():
    bib = _bib_tmpfile()
    try:
        lc = LitCluster.from_bibtex(bib, k=2, min_term_freq=1).fit()
        assert len(lc.clusters) > 0
        lc.add_paper(Paper("new", "New paper"))
        assert len(lc.clusters) == 0  # invalidated
    finally:
        bib.unlink()


def test_clear():
    lc = LitCluster(k=2)
    lc.papers.append(Paper("p1", "Test"))
    lc.clear()
    assert len(lc.papers) == 0
    assert len(lc.clusters) == 0


# ===========================================================================
# 10. Loaders
# ===========================================================================

def test_from_bibtex_paper_count():
    bib = _bib_tmpfile()
    try:
        lc = LitCluster.from_bibtex(bib)
        assert len(lc.papers) == 3  # @comment and @string skipped
    finally:
        bib.unlink()


def test_from_bibtex_fields():
    bib = _bib_tmpfile()
    try:
        lc = LitCluster.from_bibtex(bib)
        p = lc.papers[0]
        assert p.paper_id == "smith2023deep"
        assert "Deep Learning" in p.title
        assert "Smith" in p.authors
        assert p.year == "2023"
        assert p.doi == "10.1234/mlj.2023"
    finally:
        bib.unlink()


def test_from_csv_paper_count():
    csv_path = _csv_tmpfile()
    try:
        lc = LitCluster.from_csv(csv_path)
        assert len(lc.papers) == 3
    finally:
        csv_path.unlink()


def test_from_csv_fields():
    csv_path = _csv_tmpfile()
    try:
        lc = LitCluster.from_csv(csv_path)
        p = lc.papers[0]
        assert p.paper_id == "p1"
        assert "Deep Learning" in p.title
        assert p.year == "2023"
    finally:
        csv_path.unlink()


def test_from_jsonl_paper_count():
    jl = _jsonl_tmpfile()
    try:
        lc = LitCluster.from_jsonl(jl)
        assert len(lc.papers) == 3
    finally:
        jl.unlink()


def test_from_jsonl_fields():
    jl = _jsonl_tmpfile()
    try:
        lc = LitCluster.from_jsonl(jl)
        p = lc.papers[0]
        assert p.paper_id == "p1"
        assert "Deep Learning" in p.title
    finally:
        jl.unlink()


def test_from_jsonl_skips_blank_lines():
    f = tempfile.NamedTemporaryFile(
        suffix=".jsonl", mode="w", delete=False, encoding="utf-8"
    )
    f.write('{"paper_id":"a","title":"A"}\n\n{"paper_id":"b","title":"B"}\n')
    f.close()
    path = pathlib.Path(f.name)
    try:
        lc = LitCluster.from_jsonl(path)
        assert len(lc.papers) == 2
    finally:
        path.unlink()


# ===========================================================================
# 11. fit()
# ===========================================================================

def test_fit_raises_on_empty():
    lc = LitCluster(k=3)
    raised = False
    try:
        lc.fit()
    except ValueError:
        raised = True
    assert raised


def test_fit_produces_clusters():
    bib = _bib_tmpfile()
    try:
        lc = LitCluster.from_bibtex(bib, k=2, min_term_freq=1).fit()
        assert len(lc.clusters) >= 1
        assert len(lc.clusters) <= 2
    finally:
        bib.unlink()


def test_fit_all_papers_assigned():
    bib = _bib_tmpfile()
    try:
        lc = LitCluster.from_bibtex(bib, k=2, min_term_freq=1).fit()
        assigned = sum(len(c.papers) for c in lc.clusters)
        assert assigned == len(lc.papers)
    finally:
        bib.unlink()


def test_fit_k_gt_n_capped():
    csv_path = _csv_tmpfile()
    try:
        lc = LitCluster.from_csv(csv_path, k=100, min_term_freq=1).fit()
        assert len(lc.clusters) <= 3
    finally:
        csv_path.unlink()


def test_fit_deterministic():
    bib = _bib_tmpfile()
    try:
        lc1 = LitCluster.from_bibtex(bib, k=2, seed=42, min_term_freq=1).fit()
        lc2 = LitCluster.from_bibtex(bib, k=2, seed=42, min_term_freq=1).fit()
        labels1 = sorted(
            (p.paper_id, c.cluster_id)
            for c in lc1.clusters for p in c.papers
        )
        labels2 = sorted(
            (p.paper_id, c.cluster_id)
            for c in lc2.clusters for p in c.papers
        )
        assert labels1 == labels2
    finally:
        bib.unlink()


def test_fit_progress_callback():
    bib = _bib_tmpfile()
    messages = []
    try:
        LitCluster.from_bibtex(bib, k=2, min_term_freq=1).fit(
            progress=messages.append
        )
    finally:
        bib.unlink()
    assert len(messages) > 0
    assert any("Done" in m for m in messages)


def test_fit_top_terms():
    bib = _bib_tmpfile()
    try:
        lc = LitCluster.from_bibtex(bib, k=2, min_term_freq=1).fit()
        for c in lc.clusters:
            assert isinstance(c.top_terms, list)
            assert all(isinstance(t, str) for t in c.top_terms)
    finally:
        bib.unlink()


def test_fit_min_freq_filtering():
    bib = _bib_tmpfile()
    try:
        # min_term_freq=1: all terms kept
        lc1 = LitCluster.from_bibtex(bib, k=2, min_term_freq=1).fit()
        # min_term_freq=10: almost all terms filtered (rare corpus)
        lc2 = LitCluster.from_bibtex(bib, k=2, min_term_freq=10).fit()
        # Both should produce valid clusters regardless
        assert len(lc1.clusters) >= 1
        assert len(lc2.clusters) >= 1
    finally:
        bib.unlink()


# ===========================================================================
# 12. summary()
# ===========================================================================

def test_summary_contains_paper_count():
    bib = _bib_tmpfile()
    try:
        lc = LitCluster.from_bibtex(bib, k=2, min_term_freq=1).fit()
        s = lc.summary()
        assert "3" in s  # 3 papers
    finally:
        bib.unlink()


def test_summary_contains_cluster_label():
    bib = _bib_tmpfile()
    try:
        lc = LitCluster.from_bibtex(bib, k=2, min_term_freq=1).fit()
        s = lc.summary()
        assert "Cluster" in s
    finally:
        bib.unlink()


# ===========================================================================
# 13. export_csv()
# ===========================================================================

def test_export_csv_creates_file(tmp_path):
    bib = _bib_tmpfile()
    try:
        lc = LitCluster.from_bibtex(bib, k=2, min_term_freq=1).fit()
        out = tmp_path / "out.csv"
        lc.export_csv(out)
        assert out.exists()
    finally:
        bib.unlink()


def test_export_csv_row_count(tmp_path):
    bib = _bib_tmpfile()
    try:
        lc = LitCluster.from_bibtex(bib, k=2, min_term_freq=1).fit()
        out = tmp_path / "out.csv"
        lc.export_csv(out)
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 3  # one row per paper
    finally:
        bib.unlink()


def test_export_csv_required_columns(tmp_path):
    csv_path = _csv_tmpfile()
    try:
        lc = LitCluster.from_csv(csv_path, k=2, min_term_freq=1).fit()
        out = tmp_path / "out.csv"
        lc.export_csv(out)
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        for col in ("cluster_id", "cluster_label", "paper_id", "title"):
            assert col in rows[0], f"Missing column: {col}"
    finally:
        csv_path.unlink()


# ===========================================================================
# 14. export_json()
# ===========================================================================

def test_export_json_creates_file(tmp_path):
    bib = _bib_tmpfile()
    try:
        lc = LitCluster.from_bibtex(bib, k=2, min_term_freq=1).fit()
        out = tmp_path / "out.json"
        lc.export_json(out)
        assert out.exists()
    finally:
        bib.unlink()


def test_export_json_valid_structure(tmp_path):
    bib = _bib_tmpfile()
    try:
        lc = LitCluster.from_bibtex(bib, k=2, min_term_freq=1).fit()
        out = tmp_path / "out.json"
        lc.export_json(out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert "cluster_id" in data[0]
        assert "papers" in data[0]
        assert "top_terms" in data[0]
    finally:
        bib.unlink()


def test_export_json_paper_count(tmp_path):
    bib = _bib_tmpfile()
    try:
        lc = LitCluster.from_bibtex(bib, k=2, min_term_freq=1).fit()
        out = tmp_path / "out.json"
        lc.export_json(out)
        data = json.loads(out.read_text(encoding="utf-8"))
        total = sum(d["size"] for d in data)
        assert total == 3
    finally:
        bib.unlink()


# ===========================================================================
# 15. export_html()
# ===========================================================================

def test_export_html_creates_file(tmp_path):
    bib = _bib_tmpfile()
    try:
        lc = LitCluster.from_bibtex(bib, k=2, min_term_freq=1).fit()
        out = tmp_path / "out.html"
        lc.export_html(out)
        assert out.exists()
    finally:
        bib.unlink()


def test_export_html_is_valid_html(tmp_path):
    bib = _bib_tmpfile()
    try:
        lc = LitCluster.from_bibtex(bib, k=2, min_term_freq=1).fit()
        out = tmp_path / "out.html"
        lc.export_html(out)
        html = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
    finally:
        bib.unlink()


def test_export_html_contains_cluster_info(tmp_path):
    bib = _bib_tmpfile()
    try:
        lc = LitCluster.from_bibtex(bib, k=2, min_term_freq=1).fit()
        out = tmp_path / "out.html"
        lc.export_html(out)
        html = out.read_text(encoding="utf-8")
        assert "Cluster" in html
        assert "3 papers" in html
    finally:
        bib.unlink()


def test_export_html_doi_links(tmp_path):
    bib = _bib_tmpfile()
    try:
        lc = LitCluster.from_bibtex(bib, k=2, min_term_freq=1).fit()
        out = tmp_path / "out.html"
        lc.export_html(out)
        html = out.read_text(encoding="utf-8")
        # smith2023deep has DOI 10.1234/mlj.2023
        assert "10.1234/mlj.2023" in html
    finally:
        bib.unlink()


# ===========================================================================
# 16. Module-level attributes
# ===========================================================================

def test_version_attribute():
    assert hasattr(lc_mod, "__version__")
    assert isinstance(lc_mod.__version__, str)
    parts = lc_mod.__version__.split(".")
    assert len(parts) == 3


def test_public_api_exports():
    for name in ("LitCluster", "Paper", "Cluster"):
        assert hasattr(lc_mod, name), f"Missing public symbol: {name}"
