"""
litcluster: Semantic clustering and topic modelling of scientific literature.

Ingests collections of scientific abstracts or full papers (via BibTeX files,
CSV, or JSONL), embeds them using TF-IDF vectorisation, and applies k-means
clustering to produce structured cluster summaries for literature reviews.

All functionality lives in the top-level ``litcluster`` module::

    from litcluster import LitCluster, Paper, Cluster

Zero external dependencies — pure Python standard library.
"""

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"
