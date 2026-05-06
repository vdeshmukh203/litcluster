# litcluster

**Topic-based clustering of scientific literature using TF-IDF and k-means.**

`litcluster` groups a collection of academic papers into thematic clusters.
It works entirely with the Python standard library — no external dependencies
are required.  Input can be a **CSV file**, **JSONL file**, or **BibTeX file**;
output can be a text summary, a CSV, or a JSON file.
A **tkinter GUI** is also provided for interactive exploration.

---

## Installation

```bash
pip install .
```

Or run directly from the repository without installing:

```bash
python litcluster.py papers.bib -k 5
```

---

## Quick start

### Command-line interface

```bash
# Print a cluster summary (default)
litcluster refs.bib -k 5

# Write cluster assignments to CSV
litcluster papers.csv -k 6 --format csv --output clusters.csv

# Write full cluster data (with abstracts) to JSON
litcluster papers.jsonl -k 4 --format json -o clusters.json
```

All options:

| Flag | Default | Description |
|------|---------|-------------|
| `-k`, `--clusters` | `5` | Number of clusters |
| `--format` | `summary` | Output format: `summary`, `csv`, `json` |
| `-o`, `--output` | — | Output file (stdout for summary if omitted) |
| `--seed` | `42` | Random seed for reproducibility |
| `--max-iter` | `100` | Maximum k-means iterations |
| `--min-freq` | `2` | Minimum document frequency for vocabulary terms |

### Graphical user interface

```bash
python litcluster_gui.py
```

The GUI lets you open a file, adjust parameters, run clustering, browse
clusters and papers, read abstracts, and export results — all without
touching the command line.

### Python API

```python
from litcluster import LitCluster

# Load from CSV (columns: paper_id, title, abstract, authors, year, venue, doi, keywords)
lc = LitCluster.from_csv("papers.csv", k=5)
lc.fit()
print(lc.summary())

# Load from BibTeX
lc = LitCluster.from_bibtex("refs.bib", k=4, min_term_freq=1)
lc.fit()
lc.export_json("clusters.json")

# Load from JSONL
lc = LitCluster.from_jsonl("papers.jsonl", k=3, seed=0)
lc.fit()
lc.export_csv("clusters.csv")

# Inspect results programmatically
for cluster in lc.clusters:
    print(cluster.label, "—", len(cluster.papers), "papers")
    print("Top terms:", cluster.top_terms[:5])
    for paper in cluster.papers:
        print(" •", paper.title)
```

---

## Input formats

### CSV

Required column: `title`.  All other columns are optional.

```csv
paper_id,title,abstract,authors,year,venue,doi,keywords
1,Deep Learning for NLP,"...",Smith A.,2021,ACL,,neural;transformers
```

### JSONL

One JSON object per line with the same fields as CSV.

```jsonl
{"paper_id": "1", "title": "Deep Learning for NLP", "abstract": "..."}
{"paper_id": "2", "title": "BERT: Pre-training...", "abstract": "..."}
```

### BibTeX

Standard `.bib` files; `title`, `abstract`, `author`, `year`, `journal`,
`booktitle`, `doi`, and `keywords` fields are extracted automatically.

> **Note**: The BibTeX parser uses regular expressions and supports flat
> field values only.  Nested braces inside field values may be truncated.

---

## Algorithm

1. **Tokenisation** — Lower-case, alphabetic tokens of length ≥ 3, with a
   built-in English stopword list applied.
2. **TF-IDF vectorisation** — Smooth IDF weighting (sklearn-compatible).
   Terms below `--min-freq` document frequency are discarded to reduce noise.
3. **k-means clustering** — Lloyd's algorithm with cosine similarity.
   Centroids are reinitialised if a cluster becomes empty.
4. **Top-term extraction** — Each cluster is labelled with its highest-scoring
   TF-IDF terms across member papers.

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Project structure

```
litcluster/
├── litcluster.py        # Core library and CLI
├── litcluster_gui.py    # Tkinter GUI (stdlib only)
├── tests/
│   └── test_litcluster.py
├── paper.md             # JOSS manuscript
├── paper.bib
└── pyproject.toml
```

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use `litcluster` in research, please cite the accompanying JOSS paper
(see [CITATION.cff](CITATION.cff)).
