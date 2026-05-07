# Changelog

All notable changes to litcluster are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

- SPECTER2 / sentence-transformer embedding backend
- HDBSCAN density-based clustering backend
- UMAP 2D projection for scatter-plot visualisation
- BERTopic integration for interpretable topic labels
- Semantic Scholar API for bulk abstract fetching

## [0.1.0] — 2026-04-23

### Added

- Smoothed TF-IDF vectorisation with configurable minimum term frequency
- Lloyd's k-means clustering on cosine distance with seeded reproducibility
- Mean silhouette-score quality metric (cosine distance)
- Robust BibTeX parser handling nested braces and multi-line field values
- CSV and JSON-Lines input parsers
- Output formats: plain-text summary, CSV, JSON, interactive self-contained HTML
- Tkinter graphical user interface (`litcluster-gui`) with:
  - File browser, parameter spinboxes, background clustering thread
  - Cluster and paper tree-views with single-cluster filtering
  - Paper detail pop-up on double-click
  - Export shortcuts for CSV, JSON, and HTML
- CLI: `litcluster input.bib -k 5 --format html`
- Python API: `LitCluster`, `Paper`, `Cluster`
- Input validation with descriptive error messages
- Comprehensive pytest test suite (unit, integration, and CLI tests)
- `--version` flag on CLI
