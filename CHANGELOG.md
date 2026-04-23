# Changelog

## [Unreleased]
- BERTopic backend for interpretable topic labels (#1)
- Semantic Scholar API for bulk paper fetching (#2)
- Per-cluster LLM-generated summaries (#3)

## [0.1.0] - 2026-04-23
### Added
- SPECTER2 sentence-transformer embeddings for scientific literature
- HDBSCAN, k-means, and agglomerative clustering backends
- UMAP 2D projection for visualisation
- Interactive HTML report with cluster explorer
- BibTeX input support
- CLI: `litcluster cluster refs.bib`
- Python API: `LitCluster`, `PaperEmbedder`
