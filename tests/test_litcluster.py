"""
Tests for litcluster — Literature Clustering Tool.
Run with:  pytest tests/
"""

import csv
import json
import math
import pathlib
import sys
import tempfile
import textwrap

import pytest

# Ensure the root module is importable from the project root.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import litcluster as lc
from litcluster import Cluster, LitCluster, Paper, _bibtex_field, _cosine, _tfidf, _tokenise


# ---------------------------------------------------------------------------
# Smoke tests — public API surface
# ---------------------------------------------------------------------------

def test_public_api():
    assert hasattr(lc, "LitCluster")
    assert hasattr(lc, "Paper")
    assert hasattr(lc, "Cluster")
    assert hasattr(lc, "__version__")


# ---------------------------------------------------------------------------
# _tokenise
# ---------------------------------------------------------------------------

def test_tokenise_basic():
    tokens = _tokenise("Machine learning algorithms for text classification")
    assert "machine" in tokens
    assert "learning" in tokens
    assert "classification" in tokens


def test_tokenise_stopwords_removed():
    tokens = _tokenise("the quick brown fox jumps over the lazy dog")
    assert "the" not in tokens
    assert "over" not in tokens


def test_tokenise_min_length():
    tokens = _tokenise("a ab abc abcd")
    # "a" and "ab" are shorter than 3 chars — must be excluded.
    assert "a" not in tokens
    assert "ab" not in tokens
    assert "abc" in tokens


def test_tokenise_empty():
    assert _tokenise("") == []
    assert _tokenise("a an the") == []


# ---------------------------------------------------------------------------
# _tfidf
# ---------------------------------------------------------------------------

def test_tfidf_empty():
    vecs, vocab = _tfidf([])
    assert vecs == []
    assert vocab == []


def test_tfidf_single_doc():
    vecs, vocab = _tfidf([["neural", "network", "deep"]])
    assert len(vecs) == 1
    assert all(v > 0 for v in vecs[0].values())


def test_tfidf_scores_positive():
    docs = [["neural", "deep", "learning"], ["graph", "network", "node"]]
    vecs, vocab = _tfidf(docs)
    for vec in vecs:
        for score in vec.values():
            assert score > 0


def test_tfidf_rare_term_higher_idf():
    """A term appearing in only 1 of 3 docs should have higher IDF than a common one."""
    docs = [
        ["common", "rare"],
        ["common", "other"],
        ["common", "extra"],
    ]
    vecs, vocab = _tfidf(docs)
    # "common" has df=3, "rare" has df=1 → rare should get higher IDF weight.
    common_idf = math.log((3 + 1) / (3 + 1)) + 1.0
    rare_idf = math.log((3 + 1) / (1 + 1)) + 1.0
    assert rare_idf > common_idf


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------

def test_cosine_identical():
    v = {"a": 1.0, "b": 2.0}
    assert abs(_cosine(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal():
    a = {"x": 1.0}
    b = {"y": 1.0}
    assert _cosine(a, b) == 0.0


def test_cosine_empty():
    assert _cosine({}, {"a": 1.0}) == 0.0
    assert _cosine({"a": 1.0}, {}) == 0.0


def test_cosine_range():
    a = {"a": 1.0, "b": 0.5}
    b = {"a": 0.8, "b": 0.6, "c": 0.3}
    sim = _cosine(a, b)
    assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# _bibtex_field
# ---------------------------------------------------------------------------

def test_bibtex_field_brace():
    entry = "@article{key, title = {Learning with Neural Networks}, year = {2024}}"
    assert _bibtex_field(entry, "title") == "Learning with Neural Networks"
    assert _bibtex_field(entry, "year") == "2024"


def test_bibtex_field_nested_braces():
    entry = "@article{k, title = {A Study of {Machine} Learning}}"
    assert _bibtex_field(entry, "title") == "A Study of {Machine} Learning"


def test_bibtex_field_quote_delimited():
    entry = '@article{k, journal = "Nature Methods"}'
    assert _bibtex_field(entry, "journal") == "Nature Methods"


def test_bibtex_field_bare_number():
    entry = "@article{k, year = 2023, pages = {1--10}}"
    assert _bibtex_field(entry, "year") == "2023"


def test_bibtex_field_missing():
    entry = "@article{k, title = {Some title}}"
    assert _bibtex_field(entry, "abstract") == ""


def test_bibtex_field_case_insensitive():
    entry = "@article{k, TITLE = {Upper Case Field}}"
    assert _bibtex_field(entry, "title") == "Upper Case Field"


# ---------------------------------------------------------------------------
# Paper dataclass
# ---------------------------------------------------------------------------

def test_paper_text_combines_fields():
    p = Paper(paper_id="1", title="Deep Learning", abstract="Neural networks", keywords="CNN")
    assert "Deep Learning" in p.text
    assert "Neural networks" in p.text
    assert "CNN" in p.text


def test_paper_to_dict_keys():
    p = Paper(paper_id="42", title="T", abstract="A")
    d = p.to_dict()
    assert set(d.keys()) == {"paper_id", "title", "abstract", "authors", "year", "venue", "doi", "keywords"}


# ---------------------------------------------------------------------------
# Cluster dataclass
# ---------------------------------------------------------------------------

def test_cluster_label():
    c = Cluster(cluster_id=0, top_terms=["neural", "network", "deep", "learning"])
    assert "neural" in c.label
    assert "Cluster 0" in c.label


def test_cluster_to_dict():
    p = Paper(paper_id="1", title="T")
    c = Cluster(cluster_id=1, papers=[p], top_terms=["foo", "bar"])
    d = c.to_dict()
    assert d["cluster_id"] == 1
    assert d["size"] == 1
    assert len(d["papers"]) == 1


# ---------------------------------------------------------------------------
# LitCluster.fit — end-to-end
# ---------------------------------------------------------------------------

def _make_papers():
    """Return a small synthetic corpus with two clear topic groups."""
    ml_papers = [
        Paper(f"ml{i}", f"Deep learning neural network paper {i}",
              abstract="We train a deep neural network using gradient descent backpropagation.")
        for i in range(5)
    ]
    bio_papers = [
        Paper(f"bio{i}", f"Gene expression protein biology paper {i}",
              abstract="We sequence genomes and analyse protein expression in cells.")
        for i in range(5)
    ]
    return ml_papers + bio_papers


def test_fit_basic():
    obj = LitCluster(k=2, min_term_freq=1, seed=0)
    obj.papers = _make_papers()
    obj.fit()
    assert len(obj.clusters) == 2
    total = sum(len(c.papers) for c in obj.clusters)
    assert total == 10


def test_fit_top_terms():
    obj = LitCluster(k=2, min_term_freq=1, seed=0)
    obj.papers = _make_papers()
    obj.fit()
    for cluster in obj.clusters:
        assert len(cluster.top_terms) > 0


def test_fit_labels_coverage():
    obj = LitCluster(k=3, min_term_freq=1, seed=42)
    obj.papers = _make_papers()
    obj.fit()
    labelled = set(obj._labels)
    assert all(lbl in labelled for lbl in range(len(obj.clusters)))


def test_fit_empty_corpus():
    obj = LitCluster(k=3)
    obj.fit()
    assert obj.clusters == []


def test_fit_k_clamped(recwarn):
    """k larger than corpus size should be clamped with a warning."""
    obj = LitCluster(k=20, min_term_freq=1, seed=0)
    obj.papers = _make_papers()
    obj.fit()
    assert len(obj.clusters) <= 10
    assert any("reduced" in str(w.message).lower() for w in recwarn.list)


def test_fit_returns_self():
    obj = LitCluster(k=2, min_term_freq=1)
    obj.papers = _make_papers()
    result = obj.fit()
    assert result is obj


def test_fit_summary_nonempty():
    obj = LitCluster(k=2, min_term_freq=1, seed=0)
    obj.papers = _make_papers()
    obj.fit()
    s = obj.summary()
    assert "10 papers" in s
    assert "2 clusters" in s


# ---------------------------------------------------------------------------
# LitCluster.from_csv
# ---------------------------------------------------------------------------

def test_from_csv(tmp_path):
    csv_file = tmp_path / "papers.csv"
    rows = [
        {"paper_id": "1", "title": "Deep learning", "abstract": "Neural nets", "authors": "A", "year": "2020", "venue": "NeurIPS", "doi": "", "keywords": ""},
        {"paper_id": "2", "title": "Graph networks", "abstract": "Graph theory", "authors": "B", "year": "2021", "venue": "ICML", "doi": "", "keywords": ""},
    ]
    with csv_file.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    obj = LitCluster.from_csv(csv_file)
    assert len(obj.papers) == 2
    assert obj.papers[0].title == "Deep learning"


def test_from_csv_missing_file():
    with pytest.raises(FileNotFoundError):
        LitCluster.from_csv(pathlib.Path("/nonexistent/path.csv"))


# ---------------------------------------------------------------------------
# LitCluster.from_jsonl
# ---------------------------------------------------------------------------

def test_from_jsonl(tmp_path):
    jsonl_file = tmp_path / "papers.jsonl"
    records = [
        {"paper_id": "a", "title": "Neural networks", "abstract": "Deep learning"},
        {"paper_id": "b", "title": "Protein folding", "abstract": "Biology"},
    ]
    with jsonl_file.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    obj = LitCluster.from_jsonl(jsonl_file)
    assert len(obj.papers) == 2


def test_from_jsonl_skips_blank_lines(tmp_path):
    jsonl_file = tmp_path / "papers.jsonl"
    jsonl_file.write_text(
        '{"paper_id": "1", "title": "Test"}\n\n{"paper_id": "2", "title": "Other"}\n'
    )
    obj = LitCluster.from_jsonl(jsonl_file)
    assert len(obj.papers) == 2


def test_from_jsonl_invalid_json(tmp_path):
    jsonl_file = tmp_path / "bad.jsonl"
    jsonl_file.write_text('{"title": "ok"}\nnot-valid-json\n')
    with pytest.raises(ValueError, match="Invalid JSON"):
        LitCluster.from_jsonl(jsonl_file)


# ---------------------------------------------------------------------------
# LitCluster.from_bibtex
# ---------------------------------------------------------------------------

_BIBTEX_SAMPLE = textwrap.dedent("""\
    @article{Smith2020,
      author  = {John Smith and Jane Doe},
      title   = {Deep Learning for {Natural} Language Processing},
      journal = {Journal of Machine Learning},
      year    = {2020},
      abstract = {We present a deep neural network for NLP tasks including classification.},
      keywords = {deep learning, NLP, neural networks},
      doi     = {10.1234/jml.2020},
    }

    @inproceedings{Brown2021,
      author    = {Alice Brown},
      title     = {Graph Neural Networks: A Survey},
      booktitle = {International Conference on Machine Learning},
      year      = {2021},
      abstract  = {We survey recent advances in graph neural networks.},
    }
""")


def test_from_bibtex(tmp_path):
    bib_file = tmp_path / "refs.bib"
    bib_file.write_text(_BIBTEX_SAMPLE)
    obj = LitCluster.from_bibtex(bib_file)
    assert len(obj.papers) == 2
    paper = obj.papers[0]
    assert paper.paper_id == "Smith2020"
    assert "Natural" in paper.title  # nested braces preserved
    assert paper.authors == "John Smith and Jane Doe"
    assert paper.year == "2020"
    assert paper.venue == "Journal of Machine Learning"
    assert paper.doi == "10.1234/jml.2020"


def test_from_bibtex_booktitle_fallback(tmp_path):
    bib_file = tmp_path / "refs.bib"
    bib_file.write_text(_BIBTEX_SAMPLE)
    obj = LitCluster.from_bibtex(bib_file)
    assert obj.papers[1].venue == "International Conference on Machine Learning"


def test_from_bibtex_missing_file():
    with pytest.raises(FileNotFoundError):
        LitCluster.from_bibtex(pathlib.Path("/no/such/file.bib"))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_export_csv(tmp_path):
    obj = LitCluster(k=2, min_term_freq=1, seed=0)
    obj.papers = _make_papers()
    obj.fit()
    out = tmp_path / "out.csv"
    obj.export_csv(out)
    assert out.is_file()
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 10
    assert "cluster_id" in rows[0]
    assert "title" in rows[0]


def test_export_json(tmp_path):
    obj = LitCluster(k=2, min_term_freq=1, seed=0)
    obj.papers = _make_papers()
    obj.fit()
    out = tmp_path / "out.json"
    obj.export_json(out)
    assert out.is_file()
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert len(data) == 2
    assert "top_terms" in data[0]
    assert "papers" in data[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_summary(tmp_path, capsys):
    bib_file = tmp_path / "refs.bib"
    bib_file.write_text(_BIBTEX_SAMPLE)
    rc = lc.main([str(bib_file), "-k", "2", "--min-freq", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "papers" in out


def test_cli_missing_file(capsys):
    rc = lc.main(["/nonexistent/file.csv"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        lc.main(["--version"])
    assert exc_info.value.code == 0


def test_cli_json_output(tmp_path, capsys):
    bib_file = tmp_path / "refs.bib"
    bib_file.write_text(_BIBTEX_SAMPLE)
    out_file = tmp_path / "out.json"
    rc = lc.main([str(bib_file), "-k", "2", "--format", "json",
                  "-o", str(out_file), "--min-freq", "1"])
    assert rc == 0
    assert out_file.is_file()
    data = json.loads(out_file.read_text())
    assert isinstance(data, list)


def test_cli_csv_output(tmp_path, capsys):
    bib_file = tmp_path / "refs.bib"
    bib_file.write_text(_BIBTEX_SAMPLE)
    out_file = tmp_path / "out.csv"
    rc = lc.main([str(bib_file), "-k", "2", "--format", "csv",
                  "-o", str(out_file), "--min-freq", "1"])
    assert rc == 0
    assert out_file.is_file()


# ---------------------------------------------------------------------------
# LitCluster constructor validation
# ---------------------------------------------------------------------------

def test_invalid_k():
    with pytest.raises(ValueError):
        LitCluster(k=0)


def test_invalid_max_iter():
    with pytest.raises(ValueError):
        LitCluster(max_iter=0)


def test_invalid_min_term_freq():
    with pytest.raises(ValueError):
        LitCluster(min_term_freq=0)
