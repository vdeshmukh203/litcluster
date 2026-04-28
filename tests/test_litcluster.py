"""Tests for litcluster.py — JOSS quality checks."""

from __future__ import annotations

import csv
import json
import math
import sys
import textwrap
from pathlib import Path

import pytest

# Allow running from repo root without installation
sys.path.insert(0, str(Path(__file__).parent.parent))
import litcluster as lc
from litcluster import (
    LitCluster, Paper, Cluster,
    _tokenise, _tfidf, _cosine, _kmeans, _bibtex_field,
    __version__,
)


# ---------------------------------------------------------------------------
# _tokenise
# ---------------------------------------------------------------------------

class TestTokenise:
    def test_basic(self):
        tokens = _tokenise("Machine learning neural network")
        assert "machine" in tokens
        assert "learning" in tokens
        assert "neural" in tokens
        assert "network" in tokens

    def test_stopwords_removed(self):
        tokens = _tokenise("the cat sat on the mat")
        assert "the" not in tokens
        assert "on" not in tokens

    def test_short_tokens_excluded(self):
        tokens = _tokenise("go do it ok no")  # all < 3 chars
        assert tokens == []

    def test_case_normalisation(self):
        assert _tokenise("CLUSTERING") == _tokenise("clustering")

    def test_non_alpha_stripped(self):
        tokens = _tokenise("deep-learning: 2024 (special)")
        assert "deep" in tokens
        assert "learning" in tokens
        assert "2024" not in tokens

    def test_empty_string(self):
        assert _tokenise("") == []

    def test_returns_list(self):
        assert isinstance(_tokenise("hello world test"), list)


# ---------------------------------------------------------------------------
# _tfidf
# ---------------------------------------------------------------------------

class TestTfidf:
    def test_empty_input(self):
        vecs, vocab = _tfidf([])
        assert vecs == []
        assert vocab == []

    def test_single_document(self):
        vecs, vocab = _tfidf([["cat", "dog", "cat"]])
        assert len(vecs) == 1
        assert "cat" in vecs[0]
        assert "dog" in vecs[0]

    def test_vector_count_matches_documents(self):
        docs = [["a", "b"], ["b", "c"], ["c", "d"]]
        vecs, vocab = _tfidf(docs)
        assert len(vecs) == 3

    def test_vocab_sorted(self):
        docs = [["zebra", "apple", "mango"]]
        _, vocab = _tfidf(docs)
        assert vocab == sorted(vocab)

    def test_rare_term_has_higher_idf(self):
        # "rare" appears only in one document; "common" appears in all three
        docs = [
            ["common", "rare"],
            ["common"],
            ["common"],
        ]
        vecs, _ = _tfidf(docs)
        # For doc 0: tf("rare") == tf("common") == 0.5, but idf("rare") > idf("common")
        assert vecs[0]["rare"] > vecs[0]["common"]

    def test_all_values_non_negative(self):
        docs = [["alpha", "beta"], ["beta", "gamma"]]
        vecs, _ = _tfidf(docs)
        for vec in vecs:
            for v in vec.values():
                assert v >= 0


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------

class TestCosine:
    def test_identical_vectors(self):
        v = {"a": 1.0, "b": 2.0}
        assert abs(_cosine(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        a = {"x": 1.0}
        b = {"y": 1.0}
        assert _cosine(a, b) == 0.0

    def test_zero_vector(self):
        a = {"x": 0.0}
        b = {"x": 1.0}
        assert _cosine(a, b) == 0.0

    def test_empty_vectors(self):
        assert _cosine({}, {}) == 0.0

    def test_symmetry(self):
        a = {"a": 1.0, "b": 3.0}
        b = {"b": 2.0, "c": 1.0}
        assert abs(_cosine(a, b) - _cosine(b, a)) < 1e-12

    def test_range(self):
        a = {"x": 2.0, "y": 1.0}
        b = {"x": 1.0, "y": 3.0}
        sim = _cosine(a, b)
        assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# _kmeans
# ---------------------------------------------------------------------------

class TestKmeans:
    def _make_vecs(self):
        """Two well-separated clusters of 4 docs each."""
        group_a = [{"cat": 1.0, "dog": 0.8} for _ in range(4)]
        group_b = [{"deep": 1.0, "neural": 0.9} for _ in range(4)]
        return group_a + group_b

    def test_empty_input(self):
        assert _kmeans([], k=3) == []

    def test_k_larger_than_n_clips(self):
        vecs = [{"x": 1.0}, {"y": 1.0}]
        labels = _kmeans(vecs, k=10)
        assert len(labels) == 2

    def test_k_equals_1(self):
        vecs = [{"a": 1.0}, {"b": 1.0}, {"c": 1.0}]
        labels = _kmeans(vecs, k=1)
        assert all(lbl == 0 for lbl in labels)

    def test_label_count_matches_input(self):
        vecs = self._make_vecs()
        labels = _kmeans(vecs, k=2)
        assert len(labels) == len(vecs)

    def test_separable_clusters(self):
        vecs = self._make_vecs()
        labels = _kmeans(vecs, k=2, seed=0)
        # First 4 should share a label; last 4 should share the other
        assert len(set(labels[:4])) == 1
        assert len(set(labels[4:])) == 1
        assert labels[0] != labels[4]

    def test_reproducible_with_seed(self):
        vecs = self._make_vecs()
        assert _kmeans(vecs, k=2, seed=7) == _kmeans(vecs, k=2, seed=7)

    def test_different_seeds_may_differ(self):
        vecs = [{"a": float(i)} for i in range(10)]
        r1 = _kmeans(vecs, k=3, seed=1)
        r2 = _kmeans(vecs, k=3, seed=2)
        # Not a guaranteed difference, but with very different seeds on
        # spread-out data they should diverge
        assert isinstance(r1, list) and isinstance(r2, list)


# ---------------------------------------------------------------------------
# _bibtex_field
# ---------------------------------------------------------------------------

class TestBibtexField:
    def test_brace_delimited(self):
        entry = '@article{key, title = {My Paper Title}, year = {2020}}'
        assert _bibtex_field(entry, 'title') == 'My Paper Title'

    def test_quote_delimited(self):
        entry = '@article{key, title = "My Paper Title", year = "2020"}'
        assert _bibtex_field(entry, 'title') == 'My Paper Title'

    def test_bare_year(self):
        entry = '@article{key, year = 2021, author = {Smith}}'
        assert _bibtex_field(entry, 'year') == '2021'

    def test_nested_braces(self):
        entry = '@article{key, title = {The {ABC} Method}, year = {2020}}'
        assert _bibtex_field(entry, 'title') == 'The {ABC} Method'

    def test_missing_field_returns_empty(self):
        entry = '@article{key, title = {Foo}}'
        assert _bibtex_field(entry, 'abstract') == ''

    def test_case_insensitive(self):
        entry = '@article{key, TITLE = {Caps Title}}'
        assert _bibtex_field(entry, 'title') == 'Caps Title'


# ---------------------------------------------------------------------------
# Paper
# ---------------------------------------------------------------------------

class TestPaper:
    def test_text_property(self):
        p = Paper(paper_id="1", title="Deep Learning",
                  abstract="Neural networks", keywords="classification")
        assert "Deep Learning" in p.text
        assert "Neural networks" in p.text
        assert "classification" in p.text

    def test_to_dict_keys(self):
        p = Paper(paper_id="1", title="T")
        d = p.to_dict()
        for key in ("paper_id", "title", "abstract", "authors", "year", "venue", "doi", "keywords"):
            assert key in d

    def test_defaults_are_empty_strings(self):
        p = Paper(paper_id="x", title="T")
        assert p.abstract == ""
        assert p.doi == ""


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------

class TestCluster:
    def test_label_format(self):
        c = Cluster(cluster_id=2, top_terms=["alpha", "beta", "gamma", "delta"])
        assert c.label == "Cluster 2: alpha, beta, gamma"

    def test_to_dict_includes_papers(self):
        papers = [Paper(paper_id="1", title="A"), Paper(paper_id="2", title="B")]
        c = Cluster(cluster_id=0, papers=papers, top_terms=["foo"])
        d = c.to_dict()
        assert d["size"] == 2
        assert len(d["papers"]) == 2

    def test_empty_top_terms_label(self):
        c = Cluster(cluster_id=0)
        assert "Cluster 0" in c.label


# ---------------------------------------------------------------------------
# LitCluster — construction and validation
# ---------------------------------------------------------------------------

class TestLitClusterInit:
    def test_defaults(self):
        lc = LitCluster()
        assert lc.k == 5
        assert lc.seed == 42

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k"):
            LitCluster(k=0)

    def test_invalid_max_iter_raises(self):
        with pytest.raises(ValueError, match="max_iter"):
            LitCluster(max_iter=0)

    def test_invalid_min_term_freq_raises(self):
        with pytest.raises(ValueError, match="min_term_freq"):
            LitCluster(min_term_freq=0)

    def test_fit_empty_papers_returns_self(self):
        obj = LitCluster()
        result = obj.fit()
        assert result is obj
        assert obj.clusters == []


# ---------------------------------------------------------------------------
# LitCluster — loaders (via tmp files)
# ---------------------------------------------------------------------------

class TestLitClusterLoaders:
    def test_from_csv(self, tmp_path):
        f = tmp_path / "papers.csv"
        f.write_text(
            "paper_id,title,abstract,authors,year,venue,doi,keywords\n"
            "1,Machine Learning,An intro to ML,Smith,2020,ICML,,supervised\n"
            "2,Deep Networks,A paper on deep nets,Jones,2021,NeurIPS,,unsupervised\n",
            encoding="utf-8",
        )
        obj = LitCluster.from_csv(f)
        assert len(obj.papers) == 2
        assert obj.papers[0].title == "Machine Learning"

    def test_from_csv_missing_columns(self, tmp_path):
        f = tmp_path / "minimal.csv"
        f.write_text("title\nOnly Title\n", encoding="utf-8")
        obj = LitCluster.from_csv(f)
        assert len(obj.papers) == 1
        assert obj.papers[0].abstract == ""

    def test_from_jsonl(self, tmp_path):
        f = tmp_path / "papers.jsonl"
        f.write_text(
            '{"paper_id":"p1","title":"Clustering Methods","abstract":"K-means and DBSCAN"}\n'
            '{"paper_id":"p2","title":"Topic Modelling","abstract":"LDA and NMF approaches"}\n',
            encoding="utf-8",
        )
        obj = LitCluster.from_jsonl(f)
        assert len(obj.papers) == 2
        assert obj.papers[1].title == "Topic Modelling"

    def test_from_jsonl_skips_blank_lines(self, tmp_path):
        f = tmp_path / "papers.jsonl"
        f.write_text(
            '{"title":"A"}\n\n{"title":"B"}\n',
            encoding="utf-8",
        )
        obj = LitCluster.from_jsonl(f)
        assert len(obj.papers) == 2

    def test_from_bibtex(self, tmp_path):
        bib = textwrap.dedent("""\
            @article{smith2020,
              title  = {A Survey of Deep Learning},
              author = {Smith, J.},
              year   = {2020},
              journal = {Nature},
              abstract = {Deep learning has transformed AI.},
            }
            @inproceedings{jones2021,
              title   = {Efficient Transformers},
              author  = {Jones, A.},
              year    = {2021},
              booktitle = {NeurIPS},
              abstract  = {We propose efficient transformer models.},
            }
        """)
        f = tmp_path / "refs.bib"
        f.write_text(bib, encoding="utf-8")
        obj = LitCluster.from_bibtex(f)
        assert len(obj.papers) == 2
        titles = {p.title for p in obj.papers}
        assert "A Survey of Deep Learning" in titles
        assert "Efficient Transformers" in titles

    def test_from_bibtex_nested_braces(self, tmp_path):
        bib = textwrap.dedent("""\
            @article{key,
              title = {The {BERT} Model for NLP},
              year  = {2019},
            }
        """)
        f = tmp_path / "refs.bib"
        f.write_text(bib, encoding="utf-8")
        obj = LitCluster.from_bibtex(f)
        assert "BERT" in obj.papers[0].title


# ---------------------------------------------------------------------------
# LitCluster — fit and clustering behaviour
# ---------------------------------------------------------------------------

class TestLitClusterFit:
    def _make_lc(self, n_per_topic=6, k=2):
        """Two clearly distinct topic groups."""
        papers = []
        for i in range(n_per_topic):
            papers.append(Paper(
                paper_id=f"ml{i}",
                title="Machine Learning Classification",
                abstract="Support vector machines decision trees random forests supervised learning",
            ))
        for i in range(n_per_topic):
            papers.append(Paper(
                paper_id=f"bio{i}",
                title="Protein Structure Prediction",
                abstract="Amino acids folding molecular dynamics bioinformatics genomics",
            ))
        obj = LitCluster(k=k, seed=0, min_term_freq=1)
        obj.papers = papers
        return obj

    def test_fit_returns_self(self):
        obj = self._make_lc()
        assert obj.fit() is obj

    def test_correct_cluster_count(self):
        obj = self._make_lc(k=2).fit()
        assert len(obj.clusters) == 2

    def test_cluster_sizes_sum_to_total(self):
        obj = self._make_lc(n_per_topic=5, k=2).fit()
        total = sum(len(c.papers) for c in obj.clusters)
        assert total == 10

    def test_top_terms_populated(self):
        obj = self._make_lc(k=2).fit()
        for c in obj.clusters:
            assert len(c.top_terms) > 0

    def test_separable_topics_assigned_correctly(self):
        obj = self._make_lc(n_per_topic=8, k=2).fit()
        for cluster in obj.clusters:
            ids = {p.paper_id[:2] for p in cluster.papers}
            assert len(ids) == 1, "mixed topics ended up in the same cluster"

    def test_k_larger_than_papers_clamps(self):
        obj = LitCluster(k=20, min_term_freq=1)
        obj.papers = [Paper(paper_id=str(i), title=f"Paper {i}") for i in range(3)]
        obj.fit()
        assert len(obj.clusters) <= 3

    def test_min_term_freq_1_uses_all_terms(self):
        obj = LitCluster(k=1, min_term_freq=1)
        obj.papers = [
            Paper(paper_id="a", title="unique rare obscure"),
            Paper(paper_id="b", title="common shared frequent"),
        ]
        obj.fit()
        assert len(obj.clusters) == 1


# ---------------------------------------------------------------------------
# LitCluster — export
# ---------------------------------------------------------------------------

class TestLitClusterExport:
    def _fitted_lc(self):
        obj = LitCluster(k=2, seed=0, min_term_freq=1)
        obj.papers = [
            Paper(paper_id="1", title="Neural Network Deep Learning", abstract="Backprop gradient"),
            Paper(paper_id="2", title="Neural Network Classification", abstract="SVM features"),
            Paper(paper_id="3", title="Protein Folding Structure", abstract="Molecular dynamics"),
            Paper(paper_id="4", title="Genome Sequencing Analysis", abstract="Bioinformatics pipeline"),
        ]
        return obj.fit()

    def test_export_csv_creates_file(self, tmp_path):
        out = tmp_path / "out.csv"
        self._fitted_lc().export_csv(out)
        assert out.exists()

    def test_export_csv_has_header(self, tmp_path):
        out = tmp_path / "out.csv"
        self._fitted_lc().export_csv(out)
        rows = list(csv.reader(out.read_text(encoding="utf-8").splitlines()))
        assert rows[0][0] == "cluster_id"
        assert "title" in rows[0]

    def test_export_csv_row_count(self, tmp_path):
        out = tmp_path / "out.csv"
        lc_obj = self._fitted_lc()
        lc_obj.export_csv(out)
        rows = list(csv.reader(out.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 1 + len(lc_obj.papers)  # header + one row per paper

    def test_export_json_creates_file(self, tmp_path):
        out = tmp_path / "out.json"
        self._fitted_lc().export_json(out)
        assert out.exists()

    def test_export_json_valid_structure(self, tmp_path):
        out = tmp_path / "out.json"
        lc_obj = self._fitted_lc()
        lc_obj.export_json(out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        for cluster in data:
            assert "cluster_id" in cluster
            assert "papers" in cluster
            assert "top_terms" in cluster

    def test_summary_contains_paper_count(self):
        lc_obj = self._fitted_lc()
        s = lc_obj.summary()
        assert "4" in s

    def test_summary_contains_cluster_count(self):
        lc_obj = self._fitted_lc()
        s = lc_obj.summary()
        assert "2" in s


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_summary(self, tmp_path, capsys):
        f = tmp_path / "papers.csv"
        f.write_text(
            "title,abstract\n"
            + "\n".join(
                f"Paper {i},Abstract about topic {i % 2}" for i in range(10)
            ),
            encoding="utf-8",
        )
        from litcluster import main
        rc = main([str(f), "-k", "2", "--min-freq", "1"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "papers" in out.lower() or "cluster" in out.lower()

    def test_cli_missing_file(self, capsys):
        from litcluster import main
        rc = main(["nonexistent_file.csv"])
        assert rc == 1

    def test_cli_json_output(self, tmp_path):
        f = tmp_path / "papers.csv"
        f.write_text(
            "title,abstract\n"
            + "\n".join(f"Paper {i},Text {i}" for i in range(6)),
            encoding="utf-8",
        )
        out = tmp_path / "result.json"
        from litcluster import main
        rc = main([str(f), "-k", "2", "--format", "json", "-o", str(out), "--min-freq", "1"])
        assert rc == 0
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, list)

    def test_cli_csv_output(self, tmp_path):
        f = tmp_path / "papers.csv"
        f.write_text(
            "title,abstract\n" + "\n".join(f"Paper {i},Text {i}" for i in range(6)),
            encoding="utf-8",
        )
        out = tmp_path / "result.csv"
        from litcluster import main
        rc = main([str(f), "-k", "2", "--format", "csv", "-o", str(out), "--min-freq", "1"])
        assert rc == 0
        assert out.exists()


# ---------------------------------------------------------------------------
# Module-level attributes (JOSS: importable, versioned)
# ---------------------------------------------------------------------------

def test_version_string():
    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)

def test_public_api():
    for name in ("LitCluster", "Paper", "Cluster"):
        assert hasattr(lc, name)

def test_private_helpers_accessible():
    for name in ("_tokenise", "_tfidf", "_cosine", "_kmeans"):
        assert hasattr(lc, name)
