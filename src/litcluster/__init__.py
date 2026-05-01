"""
litcluster: Topic-based clustering of scientific literature using TF-IDF and k-means.

This package shim re-exports the public API from the top-level ``litcluster``
module.  The canonical import is simply::

    import litcluster
    lc = litcluster.LitCluster.from_bibtex("refs.bib").fit()

See the project README for full documentation.
"""

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

# Re-export the public API from the root litcluster module.
# The root module is what pip installs (py-modules = ["litcluster"]).
import importlib as _imp
import sys as _sys

_root = _imp.import_module("litcluster")

LitCluster = _root.LitCluster
Paper = _root.Paper
Cluster = _root.Cluster

__all__ = ["LitCluster", "Paper", "Cluster"]
