"""
litcluster: Topic-based clustering of scientific literature using TF-IDF and k-means.

Zero external dependencies — pure Python standard library.
"""

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

# Re-export public API from the single-module implementation.
from litcluster import (  # noqa: F401
    LitCluster,
    Paper,
    Cluster,
    _tokenise,
    _tfidf,
    _cosine,
    _kmeans,
)

__all__ = ["LitCluster", "Paper", "Cluster", "_tokenise", "_tfidf", "_cosine", "_kmeans"]
