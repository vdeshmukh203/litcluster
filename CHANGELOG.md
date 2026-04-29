# Changelog

All notable changes to litcluster are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Planned
- Optional scikit-learn backend for large corpora
- Per-cluster LLM-generated summaries
- Semantic Scholar API integration for bulk paper fetching

## [0.1.0] - 2026-04-29

### Added
- `LitCluster` class with TF-IDF + k-means clustering pipeline
- `Paper` and `Cluster` dataclasses
- BibTeX (`.bib`), CSV, and JSONL input parsers
- Plain-text summary, CSV, and JSON export
- CLI entry point: `litcluster <file> [-k N] [--format csv|json|summary]`
- Tkinter GUI entry point: `litcluster-gui`
- Reproducible seeded k-means initialisation
- Rare-term filtering via `--min-freq` / `min_term_freq` parameter
- Comprehensive docstrings on all public classes and functions
- Full pytest test suite covering tokenisation, TF-IDF, cosine similarity,
  k-means, I/O parsers, export, and CLI
