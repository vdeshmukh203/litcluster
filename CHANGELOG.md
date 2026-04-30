# Changelog

All notable changes to litcluster are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Planned
- BERTopic backend for richer topic labels
- Semantic Scholar API integration for bulk paper fetching
- Per-cluster LLM-generated summaries
- UMAP 2-D projection for scatter-plot visualisation

## [0.1.0] - 2026-04-23

### Added
- TF-IDF vectorisation with smoothed IDF (`log((N+1)/(df+1)) + 1`)
- Lloyd's k-means clustering using cosine similarity on sparse vectors
- `Paper` and `Cluster` dataclasses
- `LitCluster` class with `from_bibtex`, `from_csv`, `from_jsonl` loaders
- Automatic top-term extraction per cluster
- `export_csv`, `export_json`, `export_html` output methods
- Self-contained interactive HTML report with collapsible cluster cards,
  click-to-reveal abstracts, and a real-time search filter
- `litcluster_gui.py`: tkinter GUI with file picker, parameter controls,
  treeview results browser, and export buttons
- CLI: `litcluster <file> [-k K] [--format summary|csv|json|html]`
- Robust BibTeX field parser handling nested curly braces
- Extended stop-word list covering common academic filler words
- 63 pytest unit and integration tests (100 % passing)
- MIT licence
