"""
litcluster: Topic-based clustering of scientific literature.

This package stub re-exports the public API from the canonical
``litcluster`` module (``litcluster.py`` at the repository root).

Note
----
The installed package is built from ``litcluster.py`` directly
(``py-modules = ["litcluster"]`` in *pyproject.toml*).  This
``src/litcluster/`` directory is retained for IDE discoverability and
future restructuring to a full src-layout.
"""

from litcluster import (  # noqa: F401
    Cluster,
    LitCluster,
    Paper,
    _cosine,
    _kmeans,
    _tfidf,
    _tokenise,
    main,
)

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

__all__ = ["LitCluster", "Paper", "Cluster"]
