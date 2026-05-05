# litcluster

**Topic-based clustering of scientific literature — pure Python, zero dependencies.**

`litcluster` reads a BibTeX file (or CSV / JSON Lines), clusters papers by topic
using TF-IDF + k-means, and outputs a plain-text summary, structured CSV/JSON,
or an interactive HTML report.  A Tkinter GUI is also included.

## Installation

```bash
pip install litcluster
```

Python ≥ 3.8, no third-party packages required.

## Quick start

```bash
# Cluster a BibTeX library exported from Zotero / Mendeley / JabRef
litcluster refs.bib -k 8 --format html -o report.html

# CSV (columns: paper_id, title, abstract, authors, year, venue, doi, keywords)
litcluster papers.csv -k 5

# JSON Lines
litcluster papers.jsonl -k 10 --format json -o clusters.json

# Launch the graphical interface
litcluster-gui
```

## Python API

```python
from pathlib import Path
from litcluster import LitCluster

lc = LitCluster.from_bibtex(Path("refs.bib"), k=8, min_term_freq=2)
lc.fit(progress=print)        # optional progress messages

print(lc.summary())
lc.export_csv(Path("clusters.csv"))
lc.export_json(Path("clusters.json"))
lc.export_html(Path("report.html"))   # self-contained HTML report
```

## CLI reference

```
litcluster <input> [options]

Positional:
  input               .bib, .csv, or .jsonl file

Options:
  -k, --clusters N    Number of clusters          (default: 5)
  --format FMT        summary | csv | json | html  (default: summary)
  -o, --output FILE   Output file (auto-named if omitted)
  --seed N            Random seed                  (default: 42)
  --max-iter N        K-means iteration cap        (default: 100)
  --min-freq N        Minimum term document freq   (default: 2)
```

## Parameters

| Parameter      | Default | Description                                         |
|----------------|---------|-----------------------------------------------------|
| `k`            | 5       | Number of clusters                                  |
| `seed`         | 42      | Random seed for reproducible results                |
| `max_iter`     | 100     | Maximum k-means iterations                          |
| `min_term_freq`| 2       | Minimum document frequency for vocabulary inclusion |

## Output formats

| Format    | Flag              | Description                                      |
|-----------|-------------------|--------------------------------------------------|
| `summary` | `--format summary`| Human-readable cluster labels (default)          |
| `csv`     | `--format csv`    | One row per paper with cluster assignment        |
| `json`    | `--format json`   | Nested cluster → papers structure                |
| `html`    | `--format html`   | Self-contained interactive report with DOI links |

## How it works

1. **Tokenise** — lowercase, remove English stopwords, discard tokens < 3 chars
2. **TF-IDF** — compute smoothed TF-IDF vectors (sparse dict representation)
3. **k-means** — Lloyd's algorithm with cosine similarity, seeded for reproducibility
4. **Export** — summary text, CSV, JSON, or interactive HTML report

## Graphical interface

```bash
litcluster-gui       # or: python gui.py
```

The GUI lets you browse for a file, adjust clustering parameters, view the
cluster tree and paper details interactively, and export results in any format.

## Citation

If you use `litcluster` in published research, please cite:

```bibtex
@software{deshmukh2026litcluster,
  author  = {Deshmukh, Vaibhav},
  title   = {litcluster: Topic-based clustering of scientific literature},
  year    = {2026},
  url     = {https://github.com/vdeshmukh203/litcluster},
  version = {0.1.0}
}
```

## License

MIT © 2026 Vaibhav Deshmukh
