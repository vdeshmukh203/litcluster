"""
litcluster: Topic-based clustering of scientific literature using TF-IDF and k-means.

This package re-exports the public API from the top-level ``litcluster`` module
so that both ``import litcluster`` and ``from litcluster import LitCluster``
work regardless of whether the package is installed from source or used
in-place.
"""

from litcluster import (  # noqa: F401
    LitCluster,
    Paper,
    Cluster,
    __version__,
    __all__,
)

__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"
