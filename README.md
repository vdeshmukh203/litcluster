# litcluster

**Topic-based clustering of scientific literature** using TF-IDF and k-means.

Given a BibTeX file, CSV table, or JSONL record set, `litcluster` tokenises
each paper's title + abstract + keywords, builds sparse TF-IDF vectors, and
partitions the corpus into *k* topical clusters using Lloyd's algorithm with
k-means++ initialisation.  No external dependencies — pure Python stdlib.

[![CI](https://github.com/vdeshmukh203/litcluster/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/litcluster/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Installation

```bash
pip install .
```

Python ≥ 3.8 required; no third-party packages needed.

---

## Quick start

### Command line

```bash
# Cluster a BibTeX file into 6 topics, print a summary
litcluster refs.bib -k 6

# Export cluster assignments to CSV
litcluster refs.bib -k 6 --format csv -o clusters.csv

# Export as JSON
litcluster refs.bib -k 6 --format json -o clusters.json

# Cluster a CSV or JSONL file
litcluster papers.csv -k 5 --seed 123 --min-freq 3
litcluster papers.jsonl -k 4
```

**All options:**

| Option | Default | Description |
|---|---|---|
| `-k / --clusters` | 5 | Number of clusters |
| `--format` | summary | Output format: `summary`, `csv`, `json` |
| `-o / --output` | stdout / auto | Output path |
| `--seed` | 42 | Random seed for reproducibility |
| `--max-iter` | 100 | Maximum k-means iterations |
| `--min-freq` | 2 | Minimum document frequency for vocabulary |
| `--version` | — | Print version and exit |

### Graphical interface

```bash
litcluster-gui
# or
python litcluster_gui.py
```

The GUI lets you browse for an input file, tune parameters, inspect cluster
summaries and paper assignments in a tabbed view, and export results — all
without touching the command line.

### Python API

```python
from pathlib import Path
from litcluster import LitCluster

# Load and cluster a BibTeX file
lc = LitCluster.from_bibtex(Path("refs.bib"), k=5, seed=42)
lc.fit()

print(lc.summary())

# Inspect clusters
for cluster in lc.clusters:
    print(cluster.label, "—", len(cluster.papers), "papers")
    for paper in cluster.papers:
        print("  •", paper.title)

# Export
lc.export_csv(Path("clusters.csv"))
lc.export_json(Path("clusters.json"))
```

---

## Input formats

### BibTeX (`.bib`)

Standard BibTeX entries.  The fields `title`, `abstract`, `author`, `year`,
`journal`/`booktitle`, `doi`, and `keywords` are extracted automatically.
Nested brace-delimited values (e.g. `title = {The {BERT} Model}`) are handled
correctly.

### CSV (`.csv`)

A header row followed by one paper per row.  Recognised column names:
`paper_id`, `title`, `abstract`, `authors`, `year`, `venue`, `doi`,
`keywords`.  Only `title` is required; missing columns default to empty
strings.

### JSONL (`.jsonl`)

One JSON object per line with the same keys as the CSV columns.

---

## How it works

1. **Tokenisation** — titles, abstracts, and keywords are lower-cased, split
   on non-alpha characters, and filtered for stopwords and tokens shorter than
   three characters.
2. **TF-IDF** — sparse term-frequency/inverse-document-frequency vectors are
   built with smoothed IDF: `idf(t) = log((N+1)/(df(t)+1)) + 1`.  Terms
   appearing in fewer than `--min-freq` documents are excluded.
3. **K-means++** — centroids are seeded proportionally to their cosine
   distance from already-chosen centroids (k-means++ criterion), then refined
   by Lloyd's algorithm until convergence or `--max-iter` iterations.
4. **Cluster labelling** — each cluster is labelled by its top-10 TF-IDF
   terms (sum of member vectors), with the top-3 used in the short label.

---

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Citation

If you use `litcluster` in published research, please cite:

```bibtex
@software{litcluster,
  author  = {Deshmukh, V.A.},
  title   = {litcluster: Topic modelling and semantic clustering of scientific literature},
  year    = {2026},
  url     = {https://github.com/vdeshmukh203/litcluster},
  license = {MIT},
}
```

---

## License

MIT — see [LICENSE](LICENSE).
