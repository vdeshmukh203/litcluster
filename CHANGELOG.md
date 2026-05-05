# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Planned
- BERTopic backend for interpretable topic labels (#1)
- Semantic Scholar API integration for bulk paper fetching (#2)
- Per-cluster LLM-generated summaries (#3)
- UMAP 2D embedding visualisation

## [0.1.0] - 2026-04-23

### Added
- TF-IDF vectorisation with smoothed IDF weighting
- k-means clustering (Lloyd's algorithm) with cosine similarity
- Seeded centroid initialisation for fully reproducible results
- Input support: BibTeX (`.bib`), CSV, JSON Lines (`.jsonl`)
- Robust BibTeX parser handling nested braces, quoted fields, and bare numbers
- Filtering of `@comment`, `@preamble`, and `@string` meta-entries
- Output formats: plain-text summary, CSV, JSON, interactive HTML report
- `export_html()` generating a self-contained, zero-dependency report
- Progress callback parameter on `LitCluster.fit()` for GUI/TUI integration
- Graphical interface (`gui.py` / `litcluster-gui`) via Tkinter (stdlib)
- CLI entry point: `litcluster input.bib -k 8 --format html`
- GUI entry point: `litcluster-gui`
- Python API: `LitCluster`, `Paper`, `Cluster`
- Comprehensive test suite (60+ tests covering all public API)
- JOSS paper (`paper.md`) with accurate description and relevant references
