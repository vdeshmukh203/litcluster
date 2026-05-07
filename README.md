# litcluster

[![CI](https://github.com/vdeshmukh203/litcluster/actions/workflows/ci.yml/badge.svg)](https://github.com/vdeshmukh203/litcluster/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python ≥ 3.8](https://img.shields.io/badge/python-%E2%89%A53.8-blue.svg)](https://python.org)

**litcluster** clusters scientific papers into thematic groups using TF-IDF
vectorisation and k-means, helping researchers organise literature for
systematic or scoping reviews.  It accepts BibTeX, CSV, or JSON-Lines input
and has **zero external dependencies** — pure Python standard library.

## Features

- **Multiple input formats** — BibTeX (`.bib`), CSV, JSON-Lines (`.jsonl`)
- **TF-IDF + k-means** — interpretable, fully reproducible clustering
- **Multiple output formats** — plain-text summary, CSV table, JSON, interactive HTML report
- **Silhouette-score quality metric** — helps choose the right number of clusters
- **Graphical user interface** — Tkinter GUI for non-command-line users
- **Pure stdlib** — zero dependencies, works anywhere Python 3.8+ is installed

## Installation

```bash
git clone https://github.com/vdeshmukh203/litcluster.git
cd litcluster
pip install .
```

No additional packages are required.

## Quick start

### Command line

```bash
# Cluster a BibTeX file into 5 groups and print a summary
litcluster refs.bib -k 5

# Export an interactive HTML report
litcluster refs.bib -k 5 --format html -o report.html

# Export a CSV table of cluster assignments
litcluster papers.csv -k 8 --format csv -o clusters.csv

# Export a structured JSON file
litcluster papers.jsonl -k 6 --format json -o clusters.json
```

### Graphical interface

```bash
litcluster-gui            # installed entry point
python litcluster_gui.py  # run directly
```

The GUI lets you browse for an input file, configure clustering parameters,
view cluster assignments interactively, and export results without typing
a command.

### Python API

```python
from litcluster import LitCluster

lc = LitCluster.from_bibtex("refs.bib", k=5)
lc.fit()

print(lc.summary())
print("Silhouette score:", lc.silhouette())

lc.export_html("report.html")
lc.export_csv("clusters.csv")
lc.export_json("clusters.json")

for cluster in lc.clusters:
    print(cluster.label, "—", len(cluster.papers), "papers")
```

## CLI reference

```
usage: litcluster [-h] [-k K] [--format {csv,json,html,summary}]
                  [--output OUTPUT] [--seed SEED] [--max-iter MAX_ITER]
                  [--min-freq MIN_FREQ] [--version]
                  input

positional arguments:
  input            Input file (.bib, .csv, or .jsonl)

options:
  -k, --clusters K    Number of clusters (default: 5)
  --format            Output format: summary | csv | json | html (default: summary)
  --output, -o        Output file path (default: auto-named next to input)
  --seed              Random seed for reproducibility (default: 42)
  --max-iter          Maximum k-means iterations (default: 100)
  --min-freq          Minimum document frequency for vocabulary (default: 2)
  --version           Show version and exit
```

## Input formats

### BibTeX (`.bib`)

Standard BibTeX files.  Fields used: `title`, `abstract`, `author`, `year`,
`journal` / `booktitle`, `doi`, `keywords`.  Nested braces and multi-line
field values are handled correctly.

### CSV

Must have a header row.  Recognised columns: `paper_id`, `title`, `abstract`,
`authors`, `year`, `venue`, `doi`, `keywords`.  Missing columns default to
empty strings.

### JSON-Lines (`.jsonl`)

One JSON object per line with the same fields as CSV.

## Algorithm

1. **Tokenisation** — title, abstract, and keywords are concatenated, split
   into lowercase alphabetic tokens (≥ 3 characters), and filtered for
   English stopwords.
2. **TF-IDF vectorisation** — a sparse, smoothed TF-IDF vector is computed
   for each paper; terms appearing in fewer than `min_freq` documents are
   excluded.
3. **K-means clustering** — Lloyd's algorithm on cosine distance groups
   papers into *k* thematic clusters.  The random seed is fixed for
   reproducibility.
4. **Top-term extraction** — each cluster is labelled with its highest-scoring
   TF-IDF terms aggregated across member papers.
5. **Silhouette score** — the mean silhouette coefficient (cosine distance) is
   reported as a clustering-quality indicator.

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

## Contributing

Bug reports and pull requests are welcome at
<https://github.com/vdeshmukh203/litcluster>.

## Citation

```bibtex
@software{deshmukh2026litcluster,
  author  = {Deshmukh, Vaibhav},
  title   = {litcluster: Topic-based clustering of scientific literature},
  year    = {2026},
  url     = {https://github.com/vdeshmukh203/litcluster},
  license = {MIT},
}
```

## License

MIT © Vaibhav Deshmukh. See [LICENSE](LICENSE).
