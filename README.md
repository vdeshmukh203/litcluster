# litcluster

**Topic-based clustering of scientific literature using TF-IDF and k-means.**

`litcluster` groups a collection of academic papers into thematic clusters automatically, using term frequency–inverse document frequency (TF-IDF) vectorisation and k-means clustering. It accepts BibTeX, CSV, or JSONL input and exports results as CSV, JSON, or an interactive self-contained HTML report. A Tkinter desktop GUI is also included.

All functionality uses the Python standard library only — **no external packages required**.

---

## Features

- **Zero dependencies** — pure Python 3.8+ standard library
- **Multiple input formats** — BibTeX (`.bib`), CSV, JSONL
- **Multiple output formats** — Markdown summary, CSV, JSON, interactive HTML
- **Interactive HTML report** — collapsible cluster cards, search, filter; completely offline
- **Desktop GUI** — Tkinter interface for interactive use (`litcluster-gui`)
- **Reproducible** — fixed random seed, deterministic k-means++ initialisation

---

## Installation

```bash
pip install .
```

Or directly from source:

```bash
git clone https://github.com/vdeshmukh203/litcluster.git
cd litcluster
pip install -e .
```

---

## Quick start

### Command-line interface

```bash
# Cluster a BibTeX file into 5 groups and print a Markdown summary
litcluster refs.bib -k 5

# Export an interactive HTML report
litcluster refs.bib -k 6 --format html --output report.html

# Cluster a CSV file, export to JSON
litcluster papers.csv -k 8 --format json --output clusters.json
```

**CLI options**

| Flag | Default | Description |
|------|---------|-------------|
| `-k`, `--clusters` | `5` | Number of clusters |
| `--format` | `summary` | Output format: `summary`, `csv`, `json`, `html` |
| `--output`, `-o` | auto | Output file path |
| `--seed` | `42` | Random seed |
| `--max-iter` | `100` | Maximum k-means iterations |
| `--min-freq` | `2` | Minimum document frequency for vocabulary |
| `--title` | `"Literature Cluster Report"` | Title used in HTML export |

### Desktop GUI

```bash
litcluster-gui
```

The GUI lets you browse for an input file, adjust parameters, run clustering, inspect results by cluster, search papers, and export in any format.

### Python API

```python
from pathlib import Path
import litcluster as lc

# Load from BibTeX
model = lc.LitCluster.from_bibtex(Path("refs.bib"), k=6, min_term_freq=2)
model.fit()

# Inspect results
print(model.summary())
for cluster in model.clusters:
    print(f"\nCluster {cluster.cluster_id}: {cluster.label}")
    print("Top terms:", cluster.top_terms)
    for paper in cluster.papers:
        print(f"  {paper.title} ({paper.year})")

# Export
model.export_html(Path("report.html"), title="My Literature Survey")
model.export_csv(Path("clusters.csv"))
model.export_json(Path("clusters.json"))
```

**Load from a list of dicts (programmatic use):**

```python
papers = [
    {"title": "Deep learning for vision", "abstract": "CNN-based methods...", "year": "2021"},
    {"title": "Protein folding with AlphaFold", "abstract": "Structure prediction...", "year": "2022"},
    # ...
]
model = lc.LitCluster.from_list(papers, k=4).fit()
```

---

## Input formats

### BibTeX (`.bib`)

Standard BibTeX files. `litcluster` extracts `title`, `abstract`, `author`,
`year`, `journal`/`booktitle`, `doi`, and `keywords`. Nested braces and
multi-line values are handled correctly.

```bibtex
@article{smith2021,
  title    = {Deep learning for image recognition},
  abstract = {We present a convolutional neural network...},
  author   = {Smith, J. and Jones, A.},
  year     = {2021},
  journal  = {CVPR},
  doi      = {10.1000/xyz123},
}
```

### CSV

Any CSV with a header row. Recognised column names:
`paper_id`, `title`, `abstract`, `authors`, `year`, `venue`, `doi`, `keywords`.
All columns are optional; at least `title` or `abstract` should be present.

### JSONL

One JSON object per line, with the same field names as CSV.

---

## Output formats

### Markdown summary (default)

```markdown
# LitCluster Report

**24 papers** — **4 clusters**

| # | Label                         | Papers | Top Terms                          |
|---|-------------------------------|-------:|------------------------------------|
| 0 | Cluster 0: neural, image, cnn |      8 | neural, image, cnn, detection, ...  |
| 1 | Cluster 1: protein, structure |      6 | protein, structure, folding, ...    |
```

### Interactive HTML report

A single `.html` file with:
- Header with total paper and cluster counts
- Search bar that filters papers across all clusters
- Cluster filter dropdown and colour-coded cluster pills
- Collapsible cluster cards with top-term badges and paper tables

### CSV

Two-level CSV with columns:
`cluster_id`, `cluster_label`, `paper_id`, `title`, `authors`, `year`, `venue`, `doi`

### JSON

Array of cluster objects, each with `cluster_id`, `size`, `label`, `top_terms`,
and a `papers` array of full metadata records.

---

## Algorithm

1. **Tokenisation** — extract alphabetic tokens (≥ 3 characters), lower-cased,
   with English stopwords and common academic filler words removed.
2. **TF-IDF vectorisation** — build sparse document vectors using term
   frequency × smoothed inverse document frequency; terms appearing in fewer
   than `min_term_freq` documents are discarded.
3. **k-means++ initialisation** — select initial centroids with probability
   proportional to squared cosine distance from existing centroids, reducing
   sensitivity to random initialisation.
4. **Lloyd's algorithm** — iterate assignment (cosine similarity) and centroid
   recomputation until convergence or `max_iter` is reached.
5. **Cluster labelling** — rank terms by mean normalised TF-IDF score across
   cluster members; top-3 terms form the cluster label.

---

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

72 tests cover tokenisation, TF-IDF, cosine similarity, k-means, all input
parsers, all export formats, and the CLI.

---

## Citation

If you use `litcluster` in published research, please cite:

```bibtex
@software{deshmukh2026litcluster,
  author  = {Deshmukh, Vaibhav},
  title   = {litcluster: Topic-based clustering of scientific literature},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/vdeshmukh203/litcluster},
}
```

See also `CITATION.cff`.

---

## Licence

MIT — see `LICENSE`.
