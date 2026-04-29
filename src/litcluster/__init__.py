"""
litcluster: Topic-based clustering of scientific literature using TF-IDF and k-means.

This package re-exports the public API from the top-level ``litcluster`` module.
"""

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

# The installed package is the top-level litcluster.py module (py-modules in
# pyproject.toml).  This __init__ keeps the src/ directory importable for
# development without a full install.
import importlib as _importlib
import sys as _sys

_mod = _importlib.import_module("litcluster")
LitCluster = _mod.LitCluster
Paper = _mod.Paper
Cluster = _mod.Cluster

__all__ = ["LitCluster", "Paper", "Cluster"]
