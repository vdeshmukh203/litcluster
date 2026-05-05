"""
litcluster: Topic-based clustering of scientific literature using TF-IDF and k-means.

The canonical module is the root ``litcluster.py`` installed via
``pyproject.toml`` (``py-modules = ["litcluster"]``).  This package stub
re-exports the public API so that both ``import litcluster`` and
``from litcluster import LitCluster`` work regardless of how the package
is discovered.
"""

from __future__ import annotations

import os
import sys

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

# Ensure the repo root (containing litcluster.py) is importable.
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Re-export the full public API from the root module.
from litcluster import (  # noqa: E402, F401
    LitCluster,
    Paper,
    Cluster,
    _tokenise,
    _tfidf,
    _cosine,
    _kmeans,
    _parse_bibtex_field,
)

__all__ = [
    "LitCluster",
    "Paper",
    "Cluster",
    "__version__",
    "__author__",
    "__license__",
]
