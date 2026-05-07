"""Comprehensive test suite for litcluster."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import litcluster as lc


# ---------------------------------------------------------------------------
# Fixtures / shared sample data
# ---------------------------------------------------------------------------

SAMPLE_PAPERS = [
    {
        "paper_id": "p1",
        "title": "Deep learning for natural language processing",
        "abstract": "Neural networks for text classification and sentiment analysis.",
        "authors": "Alice Smith",
        "year": "2020",
        "venue": "NeurIPS",
        "doi": "10.1234/nlp1",
        "keywords": "deep learning, NLP, neural networks",
    },
    {
        "paper_id": "p2",
        "title": "Transformer models in NLP",
        "abstract": "Attention mechanisms and BERT for language understanding tasks.",
        "authors": "Bob Jones",
        "year": "2021",
        "venue": "ACL",
        "doi": "10.1234/nlp2",
        "keywords": "transformers, BERT, attention",
    },
    {
        "paper_id": "p3",
        "title": "Graph neural networks for molecular property prediction",
        "abstract": "Using GNNs to predict chemical properties of molecules.",
        "authors": "Carol Lee",
        "year": "2020",
        "venue": "ICML",
        "doi": "10.1234/chem1",
        "keywords": "graph neural networks, chemistry, molecules",
    },
    {
        "paper_id": "p4",
        "title": "Molecular dynamics simulations of proteins",
        "abstract": "Force field methods for protein structure simulation.",
        "authors": "Dave Brown",
        "year": "2019",
        "venue": "JCTC",
        "doi": "10.1234/chem2",
        "keywords": "molecular dynamics, proteins, force field",
    },
    {
        "paper_id": "p5",
        "title": "Reinforcement learning from human feedback",
        "abstract": "RLHF methods for aligning large language models with human values.",
        "authors": "Eve Wilson",
        "year": "2022",
        "venue": "NeurIPS",
        "doi": "10.1234/rl1",
        "keywords": "reinforcement learning, RLHF, alignment",
    },
    {
        "paper_id": "p6",
        "title": "Protein folding prediction with deep learning",
        "abstract": "AlphaFold and structure prediction using transformer neural networks.",
        "authors": "Frank Garcia",
        "year": "2021",
        "venue": "Nature",
        "doi": "10.1234/chem3",
        "keywords": "protein folding, AlphaFold, deep learning",
    },
]

SAMPLE_BIB = """\
@article{smith2020,
  title = {Deep learning for text},
  author = {Smith, Alice},
  year = {2020},
  journal = {NeurIPS},
  abstract = {Neural networks for natural language processing tasks.},
  keywords = {deep learning, NLP},
  doi = {10.1234/a1},
}

@article{jones2021,
  title = {Transformer models},
  author = {Jones, Bob},
  year = {2021},
  journal = {ACL},
  abstract = {Attention mechanisms and BERT for text understanding.},
  keywords = {transformers, BERT},
  doi = {10.1234/a2},
}

@inproceedings{lee2020,
  title = {Graph neural networks for chemistry},
  author = {Lee, Carol},
  year = {2020},
  booktitle = {ICML},
  abstract = {Predicting molecular properties with graph networks.},
  keywords = {GNN, chemistry},
}
"""


def _make_fitted(k: int = 2, **kw) -> lc.LitCluster:
    """Return a fitted LitCluster built from SAMPLE_PAPERS."""
    obj = lc.LitCluster(k=k, min_term_freq=1, **kw)
    for p in SAMPLE_PAPERS:
        obj.papers.append(lc.Paper(**p))
    return obj.fit()


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

class TestImports:
    def test_litcluster_class(self):
        assert hasattr(lc, "LitCluster")

    def test_paper_class(self):
        assert hasattr(lc, "Paper")

    def test_cluster_class(self):
        assert hasattr(lc, "Cluster")

    def test_version(self):
        assert hasattr(lc, "__version__")
        assert lc.__version__

    def test_all_exports(self):
        for name in lc.__all__:
            assert hasattr(lc, name)


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

class TestTokenise:
    def test_basic(self):
        tokens = lc._tokenise("hello world test foo bar")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_stopwords_removed(self):
        tokens = lc._tokenise("the quick brown fox")
        assert "the" not in tokens

    def test_short_words_excluded(self):
        tokens = lc._tokenise("a an ab abc abcd")
        assert "a"  not in tokens
        assert "an" not in tokens
        assert "ab" not in tokens
        assert "abc" in tokens

    def test_case_normalised(self):
        tokens = lc._tokenise("Deep LEARNING NLP")
        assert "deep" in tokens
        assert "learning" in tokens

    def test_empty_string(self):
        assert lc._tokenise("") == []

    def test_numbers_excluded(self):
        tokens = lc._tokenise("paper 2024 results analysis")
        assert "2024" not in tokens


class TestTfidf:
    def test_basic(self):
        docs = [["deep", "learning", "nlp"], ["transformers", "bert", "nlp"]]
        vectors, vocab = lc._tfidf(docs)
        assert len(vectors) == 2
        assert "nlp" in vocab

    def test_empty_input(self):
        vectors, vocab = lc._tfidf([])
        assert vectors == []
        assert vocab == []

    def test_min_freq_filters_terms(self):
        docs = [["deep", "learning"], ["transformers", "bert"]]
        _, vocab = lc._tfidf(docs, min_freq=2)
        assert vocab == []

    def test_min_freq_keeps_shared_terms(self):
        docs = [["nlp", "deep"], ["nlp", "bert"]]
        _, vocab = lc._tfidf(docs, min_freq=2)
        assert "nlp" in vocab

    def test_vector_values_nonnegative(self):
        docs = [["a", "b", "c"], ["b", "c", "d"]]
        vectors, _ = lc._tfidf(docs)
        for vec in vectors:
            for v in vec.values():
                assert v >= 0.0

    def test_all_empty_docs_after_filter(self):
        docs = [["rare1"], ["rare2"]]
        vectors, vocab = lc._tfidf(docs, min_freq=2)
        assert vocab == []
        assert len(vectors) == 2
        for vec in vectors:
            assert vec == {}


class TestCosine:
    def test_identical_vectors(self):
        v = {"a": 1.0, "b": 2.0}
        assert abs(lc._cosine(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        assert lc._cosine({"x": 1.0}, {"y": 1.0}) == 0.0

    def test_empty_vector(self):
        assert lc._cosine({}, {"a": 1.0}) == 0.0
        assert lc._cosine({"a": 1.0}, {}) == 0.0

    def test_result_in_range(self):
        a = {"a": 1.0, "b": 0.5}
        b = {"a": 0.3, "b": 1.0, "c": 0.2}
        sim = lc._cosine(a, b)
        assert 0.0 <= sim <= 1.0


class TestKmeans:
    def test_two_distinct_groups(self):
        docs = [["alpha", "beta"], ["alpha", "beta"], ["gamma", "delta"], ["gamma", "delta"]]
        vectors, _ = lc._tfidf(docs)
        labels = lc._kmeans(vectors, k=2)
        assert len(labels) == 4
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_empty_input(self):
        assert lc._kmeans([], k=3) == []

    def test_k_clamped_to_n(self):
        docs = [["a"], ["b"]]
        vectors, _ = lc._tfidf(docs)
        labels = lc._kmeans(vectors, k=100)
        assert len(set(labels)) <= 2

    def test_reproducible_with_same_seed(self):
        docs = [["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]]
        vectors, _ = lc._tfidf(docs)
        l1 = lc._kmeans(vectors, k=2, seed=42)
        l2 = lc._kmeans(vectors, k=2, seed=42)
        assert l1 == l2

    def test_single_document(self):
        vectors, _ = lc._tfidf([["nlp"]])
        labels = lc._kmeans(vectors, k=5)
        assert labels == [0]


# ---------------------------------------------------------------------------
# BibTeX field extractor
# ---------------------------------------------------------------------------

class TestBibtexField:
    def test_braced_value(self):
        entry = "@article{key, title = {Hello World}, year = {2020}}"
        assert lc._bibtex_field(entry, "title") == "Hello World"
        assert lc._bibtex_field(entry, "year") == "2020"

    def test_quoted_value(self):
        entry = '@article{key, title = "Hello World"}'
        assert lc._bibtex_field(entry, "title") == "Hello World"

    def test_nested_braces(self):
        entry = "@article{key, title = {A {nested} title}}"
        assert lc._bibtex_field(entry, "title") == "A {nested} title"

    def test_missing_field_returns_empty(self):
        entry = "@article{key, title = {Foo}}"
        assert lc._bibtex_field(entry, "abstract") == ""

    def test_multiline_abstract(self):
        entry = "@article{key,\n  abstract = {Line one.\nLine two.},\n}"
        result = lc._bibtex_field(entry, "abstract")
        assert "Line one" in result
        assert "Line two" in result

    def test_bare_numeric_value(self):
        entry = "@article{key, year = 2023, title = {T}}"
        assert lc._bibtex_field(entry, "year") == "2023"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TestPaper:
    def test_text_property_combines_fields(self):
        p = lc.Paper("id1", "Title words", abstract="Abstract text", keywords="key terms")
        assert "Title words" in p.text
        assert "Abstract text" in p.text
        assert "key terms" in p.text

    def test_to_dict_keys(self):
        p = lc.Paper("id1", "Test Title", abstract="Test abstract")
        d = p.to_dict()
        assert d["paper_id"] == "id1"
        assert d["title"] == "Test Title"
        assert d["abstract"] == "Test abstract"

    def test_defaults_are_empty_strings(self):
        p = lc.Paper("id1", "Title")
        assert p.abstract == ""
        assert p.authors == ""
        assert p.doi == ""


class TestCluster:
    def test_label_includes_top_terms(self):
        c = lc.Cluster(1, top_terms=["nlp", "deep", "learning", "neural"])
        assert "nlp" in c.label
        assert "Cluster 1" in c.label

    def test_label_only_shows_first_three_terms(self):
        c = lc.Cluster(0, top_terms=["alpha", "beta", "gamma", "delta", "epsilon"])
        assert "delta" not in c.label
        assert "epsilon" not in c.label

    def test_label_with_empty_terms(self):
        c = lc.Cluster(2)
        assert "Cluster 2" in c.label
        assert "uncategorised" in c.label

    def test_to_dict_structure(self):
        p = lc.Paper("p1", "Title")
        c = lc.Cluster(0, papers=[p], top_terms=["nlp"])
        d = c.to_dict()
        assert d["cluster_id"] == 0
        assert d["size"] == 1
        assert len(d["papers"]) == 1
        assert d["top_terms"] == ["nlp"]


# ---------------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------------

class TestFromCsv:
    def test_loads_all_papers(self, tmp_path):
        f = tmp_path / "papers.csv"
        with f.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(SAMPLE_PAPERS[0].keys()))
            w.writeheader()
            w.writerows(SAMPLE_PAPERS)
        obj = lc.LitCluster.from_csv(f)
        assert len(obj.papers) == len(SAMPLE_PAPERS)
        assert obj.papers[0].title == SAMPLE_PAPERS[0]["title"]

    def test_missing_columns_default_to_empty(self, tmp_path):
        f = tmp_path / "partial.csv"
        f.write_text("title\nPaper One\nPaper Two\n", encoding="utf-8")
        obj = lc.LitCluster.from_csv(f)
        assert len(obj.papers) == 2
        assert obj.papers[0].abstract == ""

    def test_string_path_accepted(self, tmp_path):
        f = tmp_path / "papers.csv"
        with f.open("w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=list(SAMPLE_PAPERS[0].keys())).writeheader()
        lc.LitCluster.from_csv(str(f))


class TestFromJsonl:
    def test_loads_all_papers(self, tmp_path):
        f = tmp_path / "papers.jsonl"
        f.write_text(
            "\n".join(json.dumps(p) for p in SAMPLE_PAPERS), encoding="utf-8"
        )
        obj = lc.LitCluster.from_jsonl(f)
        assert len(obj.papers) == len(SAMPLE_PAPERS)

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "papers.jsonl"
        lines = [json.dumps(SAMPLE_PAPERS[0]), "", json.dumps(SAMPLE_PAPERS[1])]
        f.write_text("\n".join(lines), encoding="utf-8")
        obj = lc.LitCluster.from_jsonl(f)
        assert len(obj.papers) == 2


class TestFromBibtex:
    def test_loads_all_entries(self, tmp_path):
        f = tmp_path / "refs.bib"
        f.write_text(SAMPLE_BIB, encoding="utf-8")
        obj = lc.LitCluster.from_bibtex(f)
        assert len(obj.papers) == 3

    def test_entry_key_used_as_paper_id(self, tmp_path):
        f = tmp_path / "refs.bib"
        f.write_text(SAMPLE_BIB, encoding="utf-8")
        obj = lc.LitCluster.from_bibtex(f)
        assert obj.papers[0].paper_id == "smith2020"

    def test_title_extracted(self, tmp_path):
        f = tmp_path / "refs.bib"
        f.write_text(SAMPLE_BIB, encoding="utf-8")
        obj = lc.LitCluster.from_bibtex(f)
        assert "deep learning" in obj.papers[0].title.lower()

    def test_booktitle_used_as_venue(self, tmp_path):
        f = tmp_path / "refs.bib"
        f.write_text(SAMPLE_BIB, encoding="utf-8")
        obj = lc.LitCluster.from_bibtex(f)
        assert obj.papers[2].venue == "ICML"


# ---------------------------------------------------------------------------
# Fitting and clustering
# ---------------------------------------------------------------------------

class TestFit:
    def test_produces_requested_clusters(self):
        obj = _make_fitted(k=2)
        assert len(obj.clusters) == 2

    def test_all_papers_assigned(self):
        obj = _make_fitted(k=2)
        total = sum(len(c.papers) for c in obj.clusters)
        assert total == len(SAMPLE_PAPERS)

    def test_top_terms_nonempty(self):
        obj = _make_fitted(k=2)
        for c in obj.clusters:
            assert len(c.top_terms) > 0

    def test_empty_input_returns_self(self):
        obj = lc.LitCluster(k=3).fit()
        assert obj.clusters == []

    def test_k_larger_than_n_clamped(self):
        obj = lc.LitCluster(k=100, min_term_freq=1)
        for p in SAMPLE_PAPERS[:2]:
            obj.papers.append(lc.Paper(**p))
        obj.fit()
        assert len(obj.clusters) <= 2

    def test_reproducible_with_same_seed(self):
        obj1 = _make_fitted(k=3, seed=7)
        obj2 = _make_fitted(k=3, seed=7)
        assert obj1._labels == obj2._labels

    def test_method_chaining(self):
        obj = lc.LitCluster(k=2, min_term_freq=1)
        for p in SAMPLE_PAPERS:
            obj.papers.append(lc.Paper(**p))
        result = obj.fit()
        assert result is obj


# ---------------------------------------------------------------------------
# Quality metric
# ---------------------------------------------------------------------------

class TestSilhouette:
    def test_returns_float_in_valid_range(self):
        obj = _make_fitted(k=2)
        s = obj.silhouette()
        assert isinstance(s, float)
        assert -1.0 <= s <= 1.0

    def test_single_cluster_returns_zero(self):
        obj = _make_fitted(k=1)
        assert obj.silhouette() == 0.0

    def test_unfitted_returns_zero(self):
        obj = lc.LitCluster(k=2)
        assert obj.silhouette() == 0.0


# ---------------------------------------------------------------------------
# Export methods
# ---------------------------------------------------------------------------

class TestExportCsv:
    def test_creates_file(self, tmp_path):
        f = tmp_path / "out.csv"
        _make_fitted().export_csv(f)
        assert f.is_file()

    def test_correct_row_count(self, tmp_path):
        f = tmp_path / "out.csv"
        _make_fitted().export_csv(f)
        rows = list(csv.DictReader(f.open()))
        assert len(rows) == len(SAMPLE_PAPERS)

    def test_required_columns_present(self, tmp_path):
        f = tmp_path / "out.csv"
        _make_fitted().export_csv(f)
        rows = list(csv.DictReader(f.open()))
        for col in ("cluster_id", "cluster_label", "paper_id", "title"):
            assert col in rows[0]


class TestExportJson:
    def test_creates_valid_json(self, tmp_path):
        f = tmp_path / "out.json"
        _make_fitted().export_json(f)
        data = json.loads(f.read_text())
        assert isinstance(data, list)

    def test_cluster_structure(self, tmp_path):
        f = tmp_path / "out.json"
        obj = _make_fitted(k=2)
        obj.export_json(f)
        data = json.loads(f.read_text())
        assert len(data) == 2
        assert "cluster_id" in data[0]
        assert "papers" in data[0]
        assert "top_terms" in data[0]


class TestExportHtml:
    def test_creates_valid_html(self, tmp_path):
        f = tmp_path / "out.html"
        _make_fitted().export_html(f)
        html = f.read_text()
        assert "<!DOCTYPE html>" in html
        assert "<table>" in html

    def test_contains_cluster_labels(self, tmp_path):
        f = tmp_path / "out.html"
        obj = _make_fitted(k=2)
        obj.export_html(f)
        html = f.read_text()
        assert "Cluster" in html

    def test_doi_links(self, tmp_path):
        f = tmp_path / "out.html"
        _make_fitted().export_html(f)
        html = f.read_text()
        assert "https://doi.org/" in html

    def test_html_escaping(self, tmp_path):
        obj = lc.LitCluster(k=1, min_term_freq=1)
        obj.papers.append(
            lc.Paper("x", "<script>alert('xss')</script>", abstract="safe")
        )
        obj.fit()
        f = tmp_path / "out.html"
        obj.export_html(f)
        html = f.read_text()
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


class TestSummary:
    def test_contains_paper_count(self):
        s = _make_fitted().summary()
        assert str(len(SAMPLE_PAPERS)) in s

    def test_contains_cluster_count(self):
        obj = _make_fitted(k=2)
        s = obj.summary()
        assert "2" in s

    def test_unfitted_message(self):
        s = lc.LitCluster(k=2).summary()
        assert "fit" in s.lower()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="k must be"):
            lc.LitCluster(k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError):
            lc.LitCluster(k=-1)

    def test_max_iter_zero_raises(self):
        with pytest.raises(ValueError, match="max_iter"):
            lc.LitCluster(max_iter=0)

    def test_min_term_freq_zero_raises(self):
        with pytest.raises(ValueError, match="min_term_freq"):
            lc.LitCluster(min_term_freq=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def _write_jsonl(self, path: Path) -> None:
        path.write_text(
            "\n".join(json.dumps(p) for p in SAMPLE_PAPERS), encoding="utf-8"
        )

    def test_summary_output(self, tmp_path, capsys):
        f = tmp_path / "papers.jsonl"
        self._write_jsonl(f)
        ret = lc.main([str(f), "-k", "2", "--min-freq", "1"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "papers" in captured.out.lower()

    def test_csv_output(self, tmp_path):
        f = tmp_path / "papers.jsonl"
        self._write_jsonl(f)
        out = tmp_path / "clusters.csv"
        ret = lc.main([str(f), "-k", "2", "--format", "csv",
                       "-o", str(out), "--min-freq", "1"])
        assert ret == 0
        assert out.is_file()
        rows = list(csv.DictReader(out.open()))
        assert len(rows) == len(SAMPLE_PAPERS)

    def test_json_output(self, tmp_path):
        f = tmp_path / "papers.jsonl"
        self._write_jsonl(f)
        out = tmp_path / "clusters.json"
        ret = lc.main([str(f), "-k", "2", "--format", "json",
                       "-o", str(out), "--min-freq", "1"])
        assert ret == 0
        data = json.loads(out.read_text())
        assert len(data) == 2

    def test_html_output(self, tmp_path):
        f = tmp_path / "papers.jsonl"
        self._write_jsonl(f)
        out = tmp_path / "clusters.html"
        ret = lc.main([str(f), "-k", "2", "--format", "html",
                       "-o", str(out), "--min-freq", "1"])
        assert ret == 0
        assert "<html" in out.read_text().lower()

    def test_missing_file_returns_error(self, tmp_path):
        ret = lc.main(["nonexistent_file.csv"])
        assert ret == 1

    def test_bibtex_input(self, tmp_path, capsys):
        f = tmp_path / "refs.bib"
        f.write_text(SAMPLE_BIB, encoding="utf-8")
        ret = lc.main([str(f), "-k", "2", "--min-freq", "1"])
        assert ret == 0

    def test_cli_alias(self):
        assert lc._cli is lc.main
