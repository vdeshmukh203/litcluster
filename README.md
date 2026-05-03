# litcluster

[![CI](https://github.com/vdeshmukh203/litcluster/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/litcluster/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)

**litcluster** clusters academic papers into thematic groups using TF-IDF vectorisation and k-means — zero external dependencies, pure Python standard library.

It is designed to help researchers organise large literature collections at the start of a systematic or scoping review, surfacing thematic clusters and their top discriminative terms automatically.

---

## Features

- Ingest papers from **BibTeX** (`.bib`), **CSV**, or **JSONL** files
- TF-IDF vectorisation with configurable rare-term filtering
- k-means clustering using cosine similarity
- Per-cluster top-term extraction for interpretable labels
- Export results as **CSV** or **JSON**
- Interactive **GUI** (tkinter, stdlib — no install required)
- **CLI** with summary, CSV, and JSON output modes
- Fully reproducible: seeded random initialisation

---

## Installation

```bash
pip install litcluster
```

Or from source:

```bash
git clone https://github.com/vdeshmukh203/litcluster.git
cd litcluster
pip install -e .
```

No external packages are required — litcluster depends only on the Python standard library.

---

## Quick Start

### GUI

```bash
litcluster-gui
# or
litcluster --gui
```

The GUI lets you browse for a file, tune parameters with spinboxes, run clustering, inspect cluster contents and individual paper metadata, and export results.

### CLI

```bash
# Print a summary to stdout
litcluster refs.bib -k 5

# Export cluster assignments as CSV
litcluster refs.bib -k 8 --format csv -o clusters.csv

# Export full cluster data (papers + top terms) as JSON
litcluster refs.bib -k 8 --format json -o clusters.json
```

### Python API

```python
from pathlib import Path
from litcluster import LitCluster

lc = LitCluster.from_bibtex(Path("refs.bib"), k=6)
lc.fit()

print(lc.summary())
lc.export_csv(Path("clusters.csv"))
lc.export_json(Path("clusters.json"))
```

---

## Input Formats

### BibTeX (`.bib`)

Standard BibTeX files exported from reference managers (Zotero, Mendeley, etc.).
litcluster reads the `title`, `abstract`, `author`, `year`, `journal`/`booktitle`,
`doi`, and `keywords` fields. Multi-line values and nested braces are handled.

### CSV (`.csv`)

Column names: `paper_id`, `title`, `abstract`, `authors`, `year`, `venue`, `doi`,
`keywords`. All columns except `title` are optional.

### JSONL (`.jsonl`)

One JSON object per line, using the same field names as CSV.

---

## Parameters

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| `k` | `-k` / `--clusters` | `5` | Number of clusters |
| `seed` | `--seed` | `42` | Random seed for reproducibility |
| `max_iter` | `--max-iter` | `100` | Maximum k-means iterations |
| `min_term_freq` | `--min-freq` | `2` | Minimum document frequency for a term to enter the vocabulary |

---

## Output

### Summary (default)

```
LitCluster: 42 papers in 5 clusters

  [0] Cluster 0: deep, learning, neural  (9 papers)
  [1] Cluster 1: climate, temperature, carbon  (8 papers)
  [2] Cluster 2: genome, protein, sequence  (10 papers)
  [3] Cluster 3: quantum, circuit, gate  (7 papers)
  [4] Cluster 4: polymer, synthesis, catalyst  (8 papers)
```

### CSV

Columns: `cluster_id`, `cluster_label`, `paper_id`, `title`, `authors`, `year`,
`venue`, `doi`.

### JSON

Array of cluster objects, each containing `cluster_id`, `size`, `top_terms`,
`label`, and a `papers` array with full metadata.

---

## Development

```bash
git clone https://github.com/vdeshmukh203/litcluster.git
cd litcluster
pip install -e ".[dev]"
pytest tests/ -v
```

---

## How It Works

1. **Load** — papers are ingested from BibTeX, CSV, or JSONL.
2. **Tokenise** — title, abstract, and keywords are lower-cased; alphabetic tokens
   of length >= 3 are retained after stopword removal.
3. **Filter** — terms appearing in fewer than `min_term_freq` documents are dropped,
   reducing noise from highly specialised or misspelled tokens.
4. **Vectorise** — each paper is represented as a sparse TF-IDF vector using
   smoothed IDF: `idf(t) = log((N+1)/(df(t)+1)) + 1`.
5. **Cluster** — Lloyd's k-means algorithm iterates assignment and centroid update
   steps using cosine similarity until convergence or `max_iter` is reached.
   Empty clusters are re-seeded to a random document.
6. **Label** — the top-`n` terms by aggregate TF-IDF score across cluster members
   are extracted as interpretable cluster labels.

---

## Citation

If you use litcluster in published work, please cite:

```bibtex
@software{deshmukh2026litcluster,
  author  = {Deshmukh, Vaibhav},
  title   = {litcluster: Topic-based clustering of scientific literature},
  year    = {2026},
  url     = {https://github.com/vdeshmukh203/litcluster},
  license = {MIT}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
