# litcluster

[![CI](https://github.com/vdeshmukh203/litcluster/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/litcluster/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

**litcluster** clusters academic papers into thematic groups using TF-IDF
vectorisation and k-means (Lloyd's algorithm).  It accepts BibTeX, CSV, or
JSON Lines input and requires **no external dependencies** — only Python 3.8+.

---

## Installation

```bash
# From source (no pip install required — stdlib only)
git clone https://github.com/vdeshmukh203/litcluster.git
cd litcluster

# Or install as a package for the `litcluster` CLI command:
pip install .
```

## Quick start

### Command line

```bash
# Cluster a BibTeX file into 8 groups and print a summary
litcluster refs.bib -k 8

# Export to JSON
litcluster refs.bib -k 6 --format json -o clusters.json

# Export to CSV
litcluster papers.csv -k 5 --format csv -o assignments.csv

# Use a CSV or JSON Lines file
litcluster papers.jsonl -k 10 --seed 7
```

### GUI

```bash
python litcluster_gui.py
# or, after pip install:
litcluster-gui
```

The GUI lets you open a file, adjust parameters with spinboxes, inspect each
cluster's papers interactively, and export results — no command line needed.

### Python API

```python
from litcluster import LitCluster

# Load from BibTeX
lc = LitCluster.from_bibtex("refs.bib", k=6, seed=42)
lc.fit()

# Inspect results
print(lc.summary())
for cluster in lc.clusters:
    print(cluster.label, "—", len(cluster.papers), "papers")

# Export
lc.export_csv("clusters.csv")
lc.export_json("clusters.json")
```

## CLI reference

```
litcluster [-h] [-k K] [--format {csv,json,summary}]
           [--output FILE] [--seed N] [--max-iter N]
           [--min-freq N] [--version]
           input
```

| Option | Default | Description |
|---|---|---|
| `input` | — | Input file: `.bib`, `.csv`, or `.jsonl` |
| `-k`, `--clusters` | 5 | Number of clusters |
| `--format` | `summary` | Output format: `summary`, `csv`, or `json` |
| `-o`, `--output` | stdout / auto | Output file path |
| `--seed` | 42 | Random seed for reproducibility |
| `--max-iter` | 100 | Maximum k-means iterations |
| `--min-freq` | 2 | Minimum document frequency for vocabulary terms |

## Input formats

| Format | File extension | Required columns/fields |
|---|---|---|
| BibTeX | `.bib` | Any `@article`/`@inproceedings` entries — `title`, `abstract`, `keywords` used |
| CSV | `.csv` | `title`, `abstract` recommended; `paper_id`, `authors`, `year`, `venue`, `doi`, `keywords` optional |
| JSON Lines | `.jsonl` | Same field names as CSV, one JSON object per line |

## How it works

1. **Tokenise** each paper's title + abstract + keywords (lowercase, alphabetic
   tokens ≥ 3 chars, stopwords removed).
2. **Filter** terms appearing in fewer than `--min-freq` documents.
3. **TF-IDF** vectorise the corpus (smoothed IDF: `log((n+1)/(df+1)) + 1`).
4. **k-means** cluster the sparse vectors using cosine similarity.
5. **Label** each cluster with its top 10 TF-IDF terms.

## Running tests

```bash
pip install pytest
pytest tests/
```

## Citation

If you use litcluster in academic work, please cite:

```
Deshmukh, V. (2026). litcluster: Topic-based clustering of scientific
literature using TF-IDF and k-means. Journal of Open Source Software.
```

See `CITATION.cff` for machine-readable citation metadata.

## Licence

MIT — see [LICENSE](LICENSE).
