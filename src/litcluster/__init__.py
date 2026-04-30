"""
litcluster: Semantic clustering and topic modelling of scientific literature.

Ingests collections of academic papers (BibTeX, CSV, or JSONL), builds
TF-IDF vectors from titles and abstracts, and applies k-means clustering
to group papers by topic.  Results can be exported as CSV, JSON, or an
interactive self-contained HTML report.
"""

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

import sys as _sys
from pathlib import Path as _Path

# The canonical implementation lives in litcluster.py at the repository root.
# When the package is used in-development (editable install or direct clone)
# we ensure the root is on sys.path so the import resolves correctly.
_root = str(_Path(__file__).parent.parent.parent)
if _root not in _sys.path:
    _sys.path.insert(0, _root)

from litcluster import (  # noqa: E402
    LitCluster,
    Paper,
    Cluster,
    _tokenise,
    _bibtex_field,
)

__all__ = ["LitCluster", "Paper", "Cluster"]
