"""
litcluster: Semantic clustering and topic modelling of scientific literature.

Ingests collections of scientific abstracts or full papers (via DOI lists,
BibTeX files, or arXiv IDs), embeds them using sentence-transformers, and
applies hierarchical clustering and topic modelling to produce interactive
visualisations and structured cluster summaries for systematic literature
reviews.
"""

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

from .cluster import LitCluster
from .embed import PaperEmbedder

__all__ = ["LitCluster", "PaperEmbedder"]
