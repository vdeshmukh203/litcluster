"""
Comprehensive tests for litcluster.

Run with:  pytest tests/ -v
"""

import csv
import io
import json
import math
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the project root is on the path so tests work without installation.
sys.path.insert(0, str(Path(__file__).parent.parent))
import litcluster as lc

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_PAPERS = [
    {
        "paper_id": "p1",
        "title": "Deep learning for image recognition",
        "abstract": "We present a convolutional neural network for image classification.",
        "authors": "Smith, J.",
        "year": "2020",
        "venue": "CVPR",
        "doi": "10.1000/xyz001",
        "keywords": "neural network image classification",
    },
    {
        "paper_id": "p2",
        "title": "Object detection with neural networks",
        "abstract": "Neural networks enable accurate object detection in images.",
        "authors": "Jones, A.",
        "year": "2021",
        "venue": "CVPR",
        "doi": "10.1000/xyz002",
        "keywords": "object detection neural network",
    },
    {
        "paper_id": "p3",
        "title": "Transformer models for NLP",
        "abstract": "Self-attention transformer architectures achieve state-of-the-art NLP results.",
        "authors": "Brown, T.",
        "year": "2020",
        "venue": "NeurIPS",
        "doi": "10.1000/xyz003",
        "keywords": "transformer attention NLP",
    },
    {
        "paper_id": "p4",
        "title": "BERT language model fine-tuning",
        "abstract": "Fine-tuning BERT language models on downstream NLP tasks.",
        "authors": "Lee, K.",
        "year": "2019",
        "venue": "EMNLP",
        "doi": "10.1000/xyz004",
        "keywords": "BERT language model NLP",
    },
    {
        "paper_id": "p5",
        "title": "Protein structure prediction",
        "abstract": "Predicting protein folding structures using computational methods.",
        "authors": "Wang, X.",
        "year": "2021",
        "venue": "Nature",
        "doi": "10.1000/xyz005",
        "keywords": "protein structure folding",
    },
    {
        "paper_id": "p6",
        "title": "Molecular dynamics simulation",
        "abstract": "Simulating molecular dynamics for protein folding and drug discovery.",
        "authors": "Chen, L.",
        "year": "2022",
        "venue": "PLOS",
        "doi": "10.1000/xyz006",
        "keywords": "molecular dynamics protein simulation",
    },
]

SAMPLE_BIB = r"""
@article{smith2020deep,
  title     = {Deep learning for image recognition},
  abstract  = {Convolutional neural networks excel at image classification tasks.},
  author    = {Smith, John},
  year      = {2020},
  journal   = {CVPR},
  doi       = {10.1000/abc001},
  keywords  = {deep learning, image, neural network},
}

@inproceedings{jones2021object,
  title     = {Object detection with neural networks},
  abstract  = {Neural networks enable robust object detection.},
  author    = {Jones, Alice},
  year      = {2021},
  booktitle = {ECCV},
  doi       = {10.1000/abc002},
}

@article{brown2020transformer,
  title    = {Transformer models for natural language processing},
  abstract = {Self-attention transformers achieve state-of-the-art NLP benchmarks.},
  author   = {Brown, Tom},
  year     = {2020},
  journal  = {NeurIPS},
}
"""


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

class TestTokenise:
    def test_basic(self):
        tokens = lc._tokenise("hello world test foo")
        assert isinstance(tokens, list)
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty(self):
        assert lc._tokenise("") == []
        assert lc._tokenise(None) == []

    def test_stopwords_removed(self):
        tokens = lc._tokenise("the quick brown fox")
        assert "the" not in tokens
        assert "brown" in tokens or "quick" in tokens

    def test_short_words_removed(self):
        tokens = lc._tokenise("a bb ccc dddd")
        assert "a" not in tokens
        assert "bb" not in tokens
        assert "ccc" in tokens
        assert "dddd" in tokens

    def test_lowercase(self):
        tokens = lc._tokenise("Neural Networks NLP")
        assert all(t == t.lower() for t in tokens)

    def test_non_alpha_stripped(self):
        tokens = lc._tokenise("deep-learning, NLP! 2024")
        assert all(t.isalpha() for t in tokens)


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

class TestTfidf:
    def test_basic(self):
        docs = [["cat", "dog", "cat"], ["fish", "cat"]]
        vecs, vocab = lc._tfidf(docs, min_freq=1)
        assert len(vecs) == 2
        assert "cat" in vocab
        assert all(isinstance(v, dict) for v in vecs)

    def test_empty_corpus(self):
        vecs, vocab = lc._tfidf([])
        assert vecs == []
        assert vocab == []

    def test_min_freq(self):
        docs = [["common", "rare"], ["common", "another"]]
        vecs, vocab = lc._tfidf(docs, min_freq=2)
        assert "common" in vocab
        assert "rare" not in vocab

    def test_scores_positive(self):
        docs = [["neural", "network"], ["protein", "structure"]]
        vecs, vocab = lc._tfidf(docs, min_freq=1)
        for vec in vecs:
            assert all(v > 0 for v in vec.values())

    def test_idf_penalises_common_terms(self):
        docs = [["cat", "dog"], ["cat", "fish"], ["cat", "bird"]]
        vecs, vocab = lc._tfidf(docs, min_freq=1)
        # "cat" appears in every doc → low IDF → lower score relative to rare terms
        # Each doc has "cat" plus one other term; unique terms get higher IDF
        for vec in vecs:
            unique_score = max(
                v for t, v in vec.items() if t != "cat"
            )
            cat_score = vec.get("cat", 0)
            assert unique_score > cat_score


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

class TestCosine:
    def test_identical(self):
        v = {"a": 1.0, "b": 2.0}
        assert abs(lc._cosine(v, v) - 1.0) < 1e-9

    def test_orthogonal(self):
        a = {"x": 1.0}
        b = {"y": 1.0}
        assert lc._cosine(a, b) == 0.0

    def test_empty(self):
        assert lc._cosine({}, {"a": 1.0}) == 0.0
        assert lc._cosine({"a": 1.0}, {}) == 0.0

    def test_symmetry(self):
        a = {"x": 1.0, "y": 2.0}
        b = {"y": 1.0, "z": 1.0}
        assert abs(lc._cosine(a, b) - lc._cosine(b, a)) < 1e-9

    def test_range(self):
        import random
        rng = random.Random(0)
        for _ in range(20):
            a = {str(i): rng.random() for i in range(5)}
            b = {str(i): rng.random() for i in range(5)}
            sim = lc._cosine(a, b)
            assert -1e-9 <= sim <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# k-means
# ---------------------------------------------------------------------------

class TestKmeans:
    def test_basic_clusters(self):
        docs = [["cat", "dog"], ["cat", "bird"], ["fish", "water"]]
        vecs, _ = lc._tfidf(docs, min_freq=1)
        labels = lc._kmeans(vecs, k=2, seed=0)
        assert len(labels) == 3
        assert all(isinstance(l, int) for l in labels)
        assert set(labels) <= {0, 1}

    def test_k_capped_at_n(self):
        docs = [["cat"], ["dog"]]
        vecs, _ = lc._tfidf(docs, min_freq=1)
        labels = lc._kmeans(vecs, k=10, seed=0)
        assert len(set(labels)) <= 2

    def test_empty_input(self):
        assert lc._kmeans([], k=3) == []

    def test_single_document(self):
        vecs, _ = lc._tfidf([["hello", "world"]], min_freq=1)
        labels = lc._kmeans(vecs, k=3, seed=0)
        assert labels == [0]

    def test_reproducible(self):
        docs = [["neural", "network"]] * 4 + [["protein", "structure"]] * 4
        vecs, _ = lc._tfidf(docs, min_freq=1)
        l1 = lc._kmeans(vecs, k=2, seed=7)
        l2 = lc._kmeans(vecs, k=2, seed=7)
        assert l1 == l2


# ---------------------------------------------------------------------------
# Paper dataclass
# ---------------------------------------------------------------------------

class TestPaper:
    def test_defaults(self):
        p = lc.Paper(paper_id="1", title="Test")
        assert p.abstract == ""
        assert p.doi == ""

    def test_text_property(self):
        p = lc.Paper(paper_id="1", title="Neural", abstract="Networks", keywords="deep")
        assert "Neural" in p.text
        assert "Networks" in p.text
        assert "deep" in p.text

    def test_to_dict(self):
        p = lc.Paper(paper_id="1", title="Test", year="2020")
        d = p.to_dict()
        assert d["paper_id"] == "1"
        assert d["year"] == "2020"

    def test_from_dict(self):
        d = {"paper_id": "42", "title": "Hello", "year": "2022"}
        p = lc.Paper.from_dict(d)
        assert p.paper_id == "42"
        assert p.title == "Hello"
        assert p.year == "2022"

    def test_from_dict_defaults(self):
        p = lc.Paper.from_dict({}, index=5)
        assert p.paper_id == "5"
        assert p.title == ""


# ---------------------------------------------------------------------------
# Cluster dataclass
# ---------------------------------------------------------------------------

class TestCluster:
    def test_label(self):
        c = lc.Cluster(cluster_id=0, top_terms=["neural", "network", "image"])
        assert "0" in c.label
        assert "neural" in c.label

    def test_label_no_terms(self):
        c = lc.Cluster(cluster_id=1)
        assert "unlabelled" in c.label

    def test_to_dict(self):
        p = lc.Paper(paper_id="1", title="Test")
        c = lc.Cluster(cluster_id=0, papers=[p], top_terms=["cat"])
        d = c.to_dict()
        assert d["cluster_id"] == 0
        assert d["size"] == 1
        assert d["top_terms"] == ["cat"]
        assert len(d["papers"]) == 1


# ---------------------------------------------------------------------------
# LitCluster — construction and input parsers
# ---------------------------------------------------------------------------

class TestLitClusterInit:
    def test_defaults(self):
        obj = lc.LitCluster()
        assert obj.k == 5
        assert obj.seed == 42

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            lc.LitCluster(k=0)

    def test_invalid_max_iter(self):
        with pytest.raises(ValueError):
            lc.LitCluster(max_iter=0)

    def test_invalid_min_freq(self):
        with pytest.raises(ValueError):
            lc.LitCluster(min_term_freq=0)


class TestFromList:
    def test_basic(self):
        obj = lc.LitCluster.from_list(SAMPLE_PAPERS, k=2)
        assert len(obj.papers) == len(SAMPLE_PAPERS)

    def test_empty(self):
        obj = lc.LitCluster.from_list([])
        assert obj.papers == []


class TestFromCsv:
    def test_round_trip(self, tmp_path):
        p = tmp_path / "papers.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(SAMPLE_PAPERS[0].keys()))
            w.writeheader()
            w.writerows(SAMPLE_PAPERS)
        obj = lc.LitCluster.from_csv(p, k=2)
        assert len(obj.papers) == len(SAMPLE_PAPERS)
        assert obj.papers[0].title == SAMPLE_PAPERS[0]["title"]

    def test_missing_columns(self, tmp_path):
        p = tmp_path / "minimal.csv"
        p.write_text("title,abstract\nNeural Networks,Deep learning study\n")
        obj = lc.LitCluster.from_csv(p, k=1)
        assert len(obj.papers) == 1
        assert obj.papers[0].doi == ""


class TestFromJsonl:
    def test_round_trip(self, tmp_path):
        p = tmp_path / "papers.jsonl"
        lines = "\n".join(json.dumps(row) for row in SAMPLE_PAPERS)
        p.write_text(lines, encoding="utf-8")
        obj = lc.LitCluster.from_jsonl(p, k=2)
        assert len(obj.papers) == len(SAMPLE_PAPERS)

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "papers.jsonl"
        p.write_text(
            json.dumps(SAMPLE_PAPERS[0]) + "\n\n" + json.dumps(SAMPLE_PAPERS[1]) + "\n"
        )
        obj = lc.LitCluster.from_jsonl(p, k=1)
        assert len(obj.papers) == 2


class TestFromBibtex:
    def test_round_trip(self, tmp_path):
        p = tmp_path / "refs.bib"
        p.write_text(SAMPLE_BIB, encoding="utf-8")
        obj = lc.LitCluster.from_bibtex(p, k=2)
        assert len(obj.papers) == 3
        titles = [p.title for p in obj.papers]
        assert any("image" in t.lower() for t in titles)

    def test_nested_braces(self, tmp_path):
        bib = r"""@article{test,
  title = {Nested {Braces} Test},
  year  = {2021},
}"""
        p = tmp_path / "test.bib"
        p.write_text(bib)
        obj = lc.LitCluster.from_bibtex(p)
        assert "Nested" in obj.papers[0].title
        assert "Braces" in obj.papers[0].title

    def test_booktitle_fallback(self, tmp_path):
        bib = r"""@inproceedings{conf2021,
  title     = {Conference Paper},
  booktitle = {Proceedings of the Workshop},
  year      = {2021},
}"""
        p = tmp_path / "conf.bib"
        p.write_text(bib)
        obj = lc.LitCluster.from_bibtex(p)
        assert obj.papers[0].venue == "Proceedings of the Workshop"


# ---------------------------------------------------------------------------
# LitCluster.fit()
# ---------------------------------------------------------------------------

class TestFit:
    def test_basic(self):
        obj = lc.LitCluster.from_list(SAMPLE_PAPERS, k=3, min_term_freq=1)
        obj.fit()
        assert len(obj.clusters) <= 3
        assert sum(len(c.papers) for c in obj.clusters) == len(SAMPLE_PAPERS)

    def test_no_papers(self):
        obj = lc.LitCluster()
        obj.fit()
        assert obj.clusters == []

    def test_labels_assigned(self):
        obj = lc.LitCluster.from_list(SAMPLE_PAPERS, k=2, min_term_freq=1)
        obj.fit()
        assert len(obj._labels) == len(SAMPLE_PAPERS)

    def test_top_terms_non_empty(self):
        obj = lc.LitCluster.from_list(SAMPLE_PAPERS, k=2, min_term_freq=1)
        obj.fit()
        for c in obj.clusters:
            assert isinstance(c.top_terms, list)
            assert len(c.top_terms) > 0

    def test_chaining(self):
        clusters = lc.LitCluster.from_list(SAMPLE_PAPERS, k=2, min_term_freq=1).fit().clusters
        assert len(clusters) >= 1

    def test_k_larger_than_papers(self):
        few = SAMPLE_PAPERS[:2]
        obj = lc.LitCluster.from_list(few, k=10, min_term_freq=1)
        obj.fit()
        assert sum(len(c.papers) for c in obj.clusters) == 2

    def test_reproducible(self):
        def run():
            return lc.LitCluster.from_list(
                SAMPLE_PAPERS, k=3, seed=99, min_term_freq=1
            ).fit()._labels
        assert run() == run()

    def test_summary_markdown(self):
        obj = lc.LitCluster.from_list(SAMPLE_PAPERS, k=2, min_term_freq=1).fit()
        summary = obj.summary()
        assert "| #" in summary
        assert "Papers" in summary


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

class TestExportCsv:
    def test_creates_file(self, tmp_path):
        obj = lc.LitCluster.from_list(SAMPLE_PAPERS, k=2, min_term_freq=1).fit()
        out = tmp_path / "out.csv"
        obj.export_csv(out)
        assert out.exists()

    def test_header_and_rows(self, tmp_path):
        obj = lc.LitCluster.from_list(SAMPLE_PAPERS, k=2, min_term_freq=1).fit()
        out = tmp_path / "out.csv"
        obj.export_csv(out)
        with out.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        assert rows[0][0] == "cluster_id"
        assert len(rows) == len(SAMPLE_PAPERS) + 1  # header + data


class TestExportJson:
    def test_creates_file(self, tmp_path):
        obj = lc.LitCluster.from_list(SAMPLE_PAPERS, k=2, min_term_freq=1).fit()
        out = tmp_path / "out.json"
        obj.export_json(out)
        assert out.exists()

    def test_valid_json(self, tmp_path):
        obj = lc.LitCluster.from_list(SAMPLE_PAPERS, k=2, min_term_freq=1).fit()
        out = tmp_path / "out.json"
        obj.export_json(out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert all("cluster_id" in c for c in data)
        assert all("papers" in c for c in data)

    def test_all_papers_present(self, tmp_path):
        obj = lc.LitCluster.from_list(SAMPLE_PAPERS, k=2, min_term_freq=1).fit()
        out = tmp_path / "out.json"
        obj.export_json(out)
        data = json.loads(out.read_text(encoding="utf-8"))
        total = sum(len(c["papers"]) for c in data)
        assert total == len(SAMPLE_PAPERS)


class TestExportHtml:
    def test_creates_file(self, tmp_path):
        obj = lc.LitCluster.from_list(SAMPLE_PAPERS, k=2, min_term_freq=1).fit()
        out = tmp_path / "report.html"
        obj.export_html(out, title="Test Report")
        assert out.exists()

    def test_self_contained(self, tmp_path):
        obj = lc.LitCluster.from_list(SAMPLE_PAPERS, k=2, min_term_freq=1).fit()
        out = tmp_path / "report.html"
        obj.export_html(out, title="Test Report")
        html = out.read_text(encoding="utf-8")
        # Should be self-contained — no external http references
        assert "http://" not in html or "doi.org" in html
        assert "<html" in html
        assert "</html>" in html

    def test_title_embedded(self, tmp_path):
        obj = lc.LitCluster.from_list(SAMPLE_PAPERS, k=2, min_term_freq=1).fit()
        out = tmp_path / "report.html"
        obj.export_html(out, title="My Unique Title 12345")
        html = out.read_text(encoding="utf-8")
        assert "My Unique Title 12345" in html

    def test_data_embedded(self, tmp_path):
        obj = lc.LitCluster.from_list(SAMPLE_PAPERS, k=2, min_term_freq=1).fit()
        out = tmp_path / "report.html"
        obj.export_html(out)
        html = out.read_text(encoding="utf-8")
        # Paper titles from sample data should appear in the embedded JSON
        assert "image recognition" in html.lower()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCli:
    def test_summary_stdout(self, tmp_path, capsys):
        p = tmp_path / "papers.csv"
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(SAMPLE_PAPERS[0].keys()))
            w.writeheader()
            w.writerows(SAMPLE_PAPERS)
        rc = lc.main([str(p), "-k", "2", "--min-freq", "1", "--format", "summary"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "LitCluster" in out

    def test_csv_output(self, tmp_path):
        p = tmp_path / "papers.csv"
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(SAMPLE_PAPERS[0].keys()))
            w.writeheader()
            w.writerows(SAMPLE_PAPERS)
        out = tmp_path / "clusters.csv"
        rc = lc.main([str(p), "-k", "2", "--min-freq", "1", "--format", "csv", "-o", str(out)])
        assert rc == 0
        assert out.exists()

    def test_json_output(self, tmp_path):
        p = tmp_path / "papers.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in SAMPLE_PAPERS))
        out = tmp_path / "clusters.json"
        rc = lc.main([str(p), "-k", "2", "--min-freq", "1", "--format", "json", "-o", str(out)])
        assert rc == 0
        assert out.exists()

    def test_html_output(self, tmp_path):
        p = tmp_path / "refs.bib"
        p.write_text(SAMPLE_BIB, encoding="utf-8")
        out = tmp_path / "report.html"
        rc = lc.main([str(p), "-k", "2", "--min-freq", "1", "--format", "html", "-o", str(out)])
        assert rc == 0
        assert out.exists()
        assert "<html" in out.read_text(encoding="utf-8")

    def test_missing_file(self, capsys):
        rc = lc.main(["nonexistent.csv"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "not found" in err.lower()

    def test_bibtex_input(self, tmp_path, capsys):
        p = tmp_path / "refs.bib"
        p.write_text(SAMPLE_BIB, encoding="utf-8")
        rc = lc.main([str(p), "-k", "2", "--min-freq", "1"])
        assert rc == 0


# ---------------------------------------------------------------------------
# BibTeX field extraction
# ---------------------------------------------------------------------------

class TestBibtexField:
    def test_braced_value(self):
        entry = "  title = {Hello World},"
        assert lc._bibtex_field(entry, "title") == "Hello World"

    def test_nested_braces(self):
        entry = "  title = {Hello {Nested} World},"
        val = lc._bibtex_field(entry, "title")
        assert "Hello" in val
        assert "Nested" in val

    def test_quoted_value(self):
        entry = '  title = "Hello World",'
        assert lc._bibtex_field(entry, "title") == "Hello World"

    def test_bare_year(self):
        entry = "  year = 2023,"
        assert lc._bibtex_field(entry, "year") == "2023"

    def test_missing_field(self):
        entry = "  title = {Hello},"
        assert lc._bibtex_field(entry, "doi") == ""

    def test_case_insensitive(self):
        entry = "  TITLE = {Hello},"
        assert lc._bibtex_field(entry, "title") == "Hello"

    def test_multiline(self):
        entry = "  abstract = {This is\na multiline\nabstract.},"
        val = lc._bibtex_field(entry, "abstract")
        assert "multiline" in val
