"""
litcluster: Semantic clustering and topic modelling of scientific literature.

Ingests BibTeX files, CSV tables, or JSONL records; builds sparse TF-IDF
vectors from title + abstract + keywords; and partitions papers into k
topical clusters using Lloyd's algorithm with k-means++ initialisation.
Outputs plain-text summaries, CSV, or JSON — no external dependencies.
"""

from litcluster import (  # noqa: F401
    LitCluster,
    Paper,
    Cluster,
    __version__,
    _tokenise,
    _tfidf,
    _cosine,
    _kmeans,
    _bibtex_field,
    main,
)

__all__ = [
    "LitCluster",
    "Paper",
    "Cluster",
    "__version__",
    "_tokenise",
    "_tfidf",
    "_cosine",
    "_kmeans",
    "_bibtex_field",
    "main",
]
