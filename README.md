# litcluster

**litcluster** is a pure-Python tool for clustering academic paper collections
by topic. Given a BibTeX (`.bib`), CSV, or JSON Lines (`.jsonl`) file, it
produces labelled topic clusters using TF-IDF vectorisation and k-means
clustering — with no external dependencies beyond the Python standard library.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

---

## Features

- **Zero dependencies** — pure Python 3.8+ stdlib only.
- **Multiple input formats** — BibTeX, CSV, JSONL.
- **Multiple output formats** — plain-text summary, CSV, JSON.
- **Reproducible** — seeded random initialisation.
- **Python API, CLI, and GUI** — use whichever fits your workflow.

---

## Installation

```bash
pip install .
```

Or for development (editable install):

```bash
pip install -e .
```

---

## Quick start

### Python API

```python
from litcluster import LitCluster

# Load a BibTeX export from your reference manager
lc = LitCluster.from_bibtex("refs.bib", k=6).fit()

# Print a plain-text summary
print(lc.summary())

# Export results
lc.export_csv("clusters.csv")
lc.export_json("clusters.json")
```

### CLI

```bash
# Summary to stdout
litcluster refs.bib -k 6

# Export CSV
litcluster refs.bib -k 6 --format csv -o clusters.csv

# Export JSON from a JSONL input
litcluster papers.jsonl -k 8 --format json -o clusters.json
```

Full option reference:

```
usage: litcluster [-h] [-k K] [--format {csv,json,summary}]
                  [--output OUTPUT] [--seed SEED] [--max-iter MAX_ITER]
                  [--min-freq MIN_FREQ]
                  input

positional arguments:
  input                 Input file: CSV, JSONL, or .bib

options:
  -k, --clusters K      Number of clusters (default: 5)
  --format              Output format: csv | json | summary (default: summary)
  -o, --output          Output file path (default: stdout or auto-named)
  --seed                Random seed (default: 42)
  --max-iter            Max k-means iterations (default: 100)
  --min-freq            Minimum document frequency for vocabulary terms (default: 2)
```

### GUI

```bash
litcluster-gui
```

The GUI lets you browse for an input file, tune parameters, run clustering,
inspect the results in tabbed panes, and export to CSV or JSON — all without
the command line.

---

## Input formats

### CSV

A header row followed by paper records. Recognised columns (all optional):

| Column | Description |
|--------|-------------|
| `paper_id` | Unique identifier (defaults to row number) |
| `title` | Paper title |
| `abstract` | Abstract text |
| `authors` | Author list |
| `year` | Publication year |
| `venue` | Journal or conference |
| `doi` | DOI |
| `keywords` | Author-supplied keywords |

### JSON Lines (`.jsonl`)

One JSON object per line with the same field names as CSV.

### BibTeX (`.bib`)

Standard BibTeX exported from Zotero, Mendeley, JabRef, etc.
Fields parsed: `title`, `abstract`, `author`, `year`, `journal`,
`booktitle`, `doi`, `keywords`.

---

## Algorithm

1. **Tokenisation** — extract alphabetic tokens ≥ 3 characters, lowercase, remove stopwords.
2. **Rare-term filtering** — drop terms appearing in fewer than `--min-freq` documents.
3. **TF-IDF vectorisation** — smoothed IDF: `log((N+1)/(df+1)) + 1`.
4. **k-means clustering** — Lloyd's algorithm with cosine similarity;
   seeded random centroid initialisation; empty clusters reinitialised automatically.
5. **Topic labelling** — top-k terms by aggregate TF-IDF score per cluster.

---

## Contributing

Bug reports and pull requests are welcome at
<https://github.com/vdeshmukh203/litcluster>.

---

## Citation

If you use litcluster in published research, please cite:

```bibtex
@software{deshmukh2026litcluster,
  author  = {Deshmukh, V.A.},
  title   = {litcluster: Topic-based clustering of scientific literature},
  year    = {2026},
  url     = {https://github.com/vdeshmukh203/litcluster},
  version = {0.1.0},
}
```

---

## License

MIT — see [LICENSE](LICENSE).
