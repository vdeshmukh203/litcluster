# litcluster

[![CI](https://github.com/vdeshmukh203/litcluster/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/litcluster/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)

**litcluster** is a pure-Python tool for automatically clustering academic papers by topic using TF-IDF and k-means.  Given a BibTeX file, a CSV, or a JSONL collection of paper metadata, it groups papers into thematic clusters and labels each cluster with its most distinctive terms.

Results can be exported as:

- **Interactive HTML** — collapsible cluster cards, click-to-reveal abstracts, real-time search filter
- **CSV** — flat table with cluster assignments for use in spreadsheets
- **JSON** — full structured output including top terms and abstracts

An optional **GUI** (built with `tkinter`, standard-library only) provides a point-and-click interface.

**No external dependencies** — only Python ≥ 3.8 standard library required.

---

## Installation

```bash
# From PyPI (once published)
pip install litcluster

# Or directly from source
git clone https://github.com/vdeshmukh203/litcluster.git
cd litcluster
pip install -e .
```

---

## Quick start

### Command line

```bash
# Cluster a BibTeX file into 5 groups and print a summary
litcluster refs.bib

# Choose the number of clusters and export to HTML
litcluster refs.bib -k 8 --format html -o report.html

# Export to CSV
litcluster papers.csv -k 6 --format csv -o clusters.csv

# Export to JSON
litcluster papers.jsonl -k 4 --format json
```

### Python API

```python
from pathlib import Path
from litcluster import LitCluster

# Load from BibTeX
lc = LitCluster.from_bibtex(Path("refs.bib"), k=6, seed=42)
lc.fit()

# Print a plain-text summary
print(lc.summary())

# Export formats
lc.export_html(Path("report.html"))   # interactive HTML
lc.export_csv(Path("clusters.csv"))   # flat CSV
lc.export_json(Path("clusters.json")) # structured JSON

# Iterate over clusters programmatically
for cluster in lc.clusters:
    print(cluster.label, "—", len(cluster.papers), "papers")
    for paper in cluster.papers:
        print(" ", paper.title, f"({paper.year})")
```

### GUI

```bash
litcluster-gui
# or, without installation:
python litcluster_gui.py
```

The GUI lets you browse for a file, tune parameters with spin-boxes, run clustering in a background thread, browse results in a tree view, and export to CSV / JSON / HTML — all without touching the command line.

---

## Input formats

| Format | Extension | Required columns / fields |
|--------|-----------|--------------------------|
| CSV | `.csv` | `title` (all others optional: `paper_id`, `abstract`, `authors`, `year`, `venue`, `doi`, `keywords`) |
| JSONL | `.jsonl` | Same field names as CSV, one JSON object per line |
| BibTeX | `.bib` | Standard BibTeX fields: `title`, `abstract`, `author`, `year`, `journal`/`booktitle`, `doi`, `keywords` |

---

## CLI reference

```
usage: litcluster [-h] [-k K] [--format {csv,json,html,summary}]
                  [--output OUTPUT] [--seed SEED] [--max-iter MAX_ITER]
                  [--min-freq MIN_FREQ]
                  input

positional arguments:
  input                 Input file (.csv, .jsonl, or .bib)

options:
  -k, --clusters K      Number of clusters (default: 5)
  --format              Output format: summary | csv | json | html (default: summary)
  -o, --output          Output file path (default: derived from input name)
  --seed                Random seed for reproducibility (default: 42)
  --max-iter            Maximum k-means iterations (default: 100)
  --min-freq            Minimum document-frequency for vocabulary terms (default: 2)
```

---

## Python API reference

### `LitCluster(k=5, max_iter=100, seed=42, min_term_freq=2)`

Main class.

| Method | Description |
|--------|-------------|
| `from_csv(path, **kwargs)` | Load papers from a CSV file |
| `from_jsonl(path, **kwargs)` | Load papers from a JSONL file |
| `from_bibtex(path, **kwargs)` | Load papers from a BibTeX `.bib` file |
| `fit()` | Run TF-IDF + k-means; populates `.clusters`; returns `self` |
| `summary()` | Return a plain-text summary string |
| `export_csv(path)` | Write cluster assignments to CSV |
| `export_json(path)` | Write full cluster data to JSON |
| `export_html(path)` | Write an interactive self-contained HTML report |

### `Paper`

Dataclass with fields: `paper_id`, `title`, `abstract`, `authors`, `year`, `venue`, `doi`, `keywords`.  The `.text` property returns the concatenation of `title`, `abstract`, and `keywords` used for vectorisation.

### `Cluster`

Dataclass with fields: `cluster_id` (int), `papers` (list of `Paper`), `top_terms` (list of str).  The `.label` property returns a short human-readable label.

---

## How it works

1. **Tokenisation** — titles, abstracts, and keywords are lowercased and split into alphabetic tokens (≥ 3 chars); stop-words and common academic filler words are removed.
2. **TF-IDF** — a sparse term-frequency × inverse-document-frequency matrix is built using the smoothed sklearn-style IDF formula `log((N+1)/(df+1)) + 1`.
3. **k-means** — Lloyd's algorithm is run on the TF-IDF vectors using cosine similarity.  Centroids are initialised by random sampling (seeded for reproducibility).
4. **Top terms** — each cluster is labelled with its highest aggregate-weight terms across member papers.

---

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

All 63 tests should pass on Python 3.8 – 3.12 with no additional dependencies.

---

## Contributing

Bug reports and pull requests are welcome.  Please open an issue before submitting large changes.

---

## Citation

If you use litcluster in published research, please cite:

```bibtex
@software{deshmukh2026litcluster,
  author    = {Deshmukh, Vaibhav},
  title     = {litcluster: Topic modelling and semantic clustering of scientific literature},
  year      = {2026},
  url       = {https://github.com/vdeshmukh203/litcluster},
  version   = {0.1.0}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
