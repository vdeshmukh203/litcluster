# Changelog

All notable changes to litcluster are documented here.  
This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- Embedding-based clustering backend (sentence-transformers + HDBSCAN)
- UMAP 2D projection for HTML scatter-plot visualisation
- Semantic Scholar API integration for bulk paper fetching
- Per-cluster LLM-generated summaries

## [0.1.0] – 2026-04-23

### Added
- **Core algorithm**: TF-IDF vectorisation with configurable minimum document
  frequency, followed by Lloyd's k-means with k-means++ initialisation and
  cosine-similarity distance.
- **`Paper`** dataclass: stores `paper_id`, `title`, `abstract`, `authors`,
  `year`, `venue`, `doi`, `keywords`; includes `from_dict()` classmethod.
- **`Cluster`** dataclass: groups papers by cluster ID with ranked top-terms
  and an auto-generated label.
- **`LitCluster`** class with:
  - `from_bibtex()` — robust BibTeX parser with proper nested-brace handling
  - `from_csv()` — CSV reader (flexible column names)
  - `from_jsonl()` — JSONL reader
  - `from_list()` — programmatic construction from list of dicts
  - `fit()` — executes the full clustering pipeline; chainable
  - `export_csv()` — cluster assignments as CSV
  - `export_json()` — full cluster metadata as JSON
  - `export_html()` — self-contained interactive HTML report
  - `summary()` — Markdown-formatted summary table
- **Interactive HTML report**: offline-capable, single-file output with
  collapsible cluster cards, top-term badges, paper tables, real-time search,
  cluster filter dropdown, and colour-coded cluster pills.
- **CLI** (`litcluster`): supports `summary`, `csv`, `json`, and `html` output
  formats; all parameters configurable via flags.
- **Tkinter GUI** (`litcluster-gui`): file browser, parameter spinboxes,
  background clustering thread, tabbed results view (Summary / Clusters /
  Papers), paper search, and export dialogs.
- **Test suite**: 72 pytest tests covering tokenisation, TF-IDF, cosine
  similarity, k-means, all input parsers, all export formats, CLI, and BibTeX
  field extraction.
- MIT licence, `CITATION.cff`, `pyproject.toml` with correct entry points.
