"""
litcluster: Topic-based clustering of scientific literature.

The canonical single-file implementation lives at ``litcluster.py`` in the
project root and is installed as a py-module via ``pyproject.toml``.  This
package stub re-exports the public API so that both import styles work:

    import litcluster          # imports the root module directly
    from litcluster import ... # same
"""

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"

# Re-export the public API from the single-file implementation.
# The try/except handles the case where this src/ directory is on sys.path
# but the root litcluster.py is not (e.g. during an editable install that
# only exposes src/).
try:
    from litcluster import LitCluster, Paper, Cluster  # noqa: F401
    __all__ = ["LitCluster", "Paper", "Cluster"]
except ImportError:
    __all__ = []
