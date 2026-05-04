#!/usr/bin/env python3
"""
litcluster — Literature Clustering Tool
========================================
Cluster academic papers by topic using TF-IDF vectorisation and k-means
(Lloyd's algorithm).  All functionality uses Python's standard library only —
no third-party packages are required.

Command-line usage::

    litcluster papers.bib -k 8 --format json -o results.json
    litcluster papers.csv -k 5
    litcluster papers.jsonl --format csv -o clusters.csv

Python API::

    from litcluster import LitCluster

    lc = LitCluster.from_bibtex("refs.bib", k=6)
    lc.fit()
    print(lc.summary())
    lc.export_json("clusters.json")
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__version__ = "0.1.0"
__all__ = ["LitCluster", "Paper", "Cluster"]


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

# General-purpose stopwords plus common high-frequency academic/scientific
# terms that appear across virtually all papers and carry no discriminating
# information for clustering.
_STOPWORDS: frozenset = frozenset({
    # --- function words ---
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "this", "that", "these", "those",
    "it", "its", "we", "our", "they", "their", "as", "if", "not", "no",
    "nor", "so", "yet", "both", "either", "whether", "each", "few", "more",
    "most", "other", "some", "such", "than", "too", "very", "just", "also",
    "only", "then", "here", "there", "when", "where", "who", "which", "how",
    "all", "any", "can", "into", "through", "during", "before", "after",
    "above", "below", "between", "out", "off", "over", "under", "again",
    "further", "once", "i", "my", "me", "he", "she", "his", "her", "him",
    "you", "your", "while", "about", "against", "what", "one", "two",
    "three", "four", "five", "first", "second", "third", "last", "among",
    "within", "without", "per",
    # --- ubiquitous academic verbs / nouns ---
    "study", "studies", "research", "paper", "article", "work", "works",
    "method", "methods", "approach", "approaches", "technique", "techniques",
    "result", "results", "analysis", "analyses", "evaluation", "evaluations",
    "performance", "experiment", "experiments", "experimental",
    "show", "shows", "shown", "demonstrate", "demonstrates", "demonstrated",
    "present", "presents", "presented", "propose", "proposes", "proposed",
    "introduce", "introduces", "introduced", "describe", "describes",
    "described", "develop", "develops", "developed", "use", "used", "using",
    "provide", "provides", "provided", "based", "applied", "apply", "applies",
    "investigate", "investigates", "investigated",
    "compare", "compared", "comparison", "discuss", "discusses", "discussed",
    "consider", "considered", "existing", "novel", "new", "recent", "current",
    "previous", "prior", "several", "many", "large", "high", "low",
    "significant", "significantly", "important", "general", "specific",
    "different", "various", "however", "therefore", "thus", "hence",
    "furthermore", "moreover", "although", "despite",
    "conclusion", "conclusions", "conclude", "concluded",
    "finding", "findings", "contribution", "contributions",
    "able", "well", "good", "set", "number", "finally", "respectively",
    "data", "dataset", "datasets", "model", "models", "task", "tasks",
    "problem", "problems", "system", "systems", "framework", "frameworks",
    "algorithm", "algorithms", "implementation", "implementations",
    "test", "testing", "tested", "train", "training", "trained",
    "evaluate", "evaluated", "report", "reports", "reported",
    "aim", "aims", "goal", "goals", "objective", "objectives",
    "table", "figure", "section", "appendix", "equation", "equations",
    "respectively", "typically", "often", "usually", "commonly",
    "known", "given", "possible", "following", "related", "used",
})


def _tokenise(text: str) -> List[str]:
    """Return lowercase alphabetic tokens (≥ 3 chars) excluding stopwords.

    Parameters
    ----------
    text:
        Raw text to tokenise (title, abstract, keywords, etc.).

    Returns
    -------
    list of str
        Filtered token list ready for TF-IDF computation.
    """
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _tfidf(
    documents: List[List[str]],
) -> Tuple[List[Dict[str, float]], List[str]]:
    """Compute sparse TF-IDF vectors for a list of tokenised documents.

    Uses smoothed IDF  ``log((n+1)/(df+1)) + 1``  (sklearn convention) to
    prevent zero-division and to assign non-zero weight to every term.

    Parameters
    ----------
    documents:
        List of token lists, one per document.

    Returns
    -------
    vectors:
        List of sparse dicts ``{term: tfidf_score}`` — one per document.
    vocab:
        Sorted list of all terms in the corpus.
    """
    n = len(documents)
    if n == 0:
        return [], []

    vocab_set: set = set()
    for doc in documents:
        vocab_set.update(doc)
    vocab: List[str] = sorted(vocab_set)
    term_idx: Dict[str, int] = {t: i for i, t in enumerate(vocab)}

    # Document frequency per term
    df: List[int] = [0] * len(vocab)
    for doc in documents:
        for t in set(doc):
            idx = term_idx.get(t)
            if idx is not None:
                df[idx] += 1

    # Build sparse TF-IDF vectors
    vectors: List[Dict[str, float]] = []
    for doc in documents:
        raw_tf: Dict[int, int] = {}
        for t in doc:
            idx = term_idx.get(t)
            if idx is not None:
                raw_tf[idx] = raw_tf.get(idx, 0) + 1
        doc_len = max(len(doc), 1)
        vec: Dict[str, float] = {}
        for idx, count in raw_tf.items():
            tf_val = count / doc_len
            idf_val = math.log((n + 1) / (df[idx] + 1)) + 1.0
            vec[vocab[idx]] = tf_val * idf_val
        vectors.append(vec)

    return vectors, vocab


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    if not a or not b:
        return 0.0
    dot = sum(a.get(t, 0.0) * v for t, v in b.items())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _kmeans(
    vectors: List[Dict[str, float]],
    k: int,
    max_iter: int = 100,
    seed: int = 42,
) -> List[int]:
    """Lloyd's k-means over sparse TF-IDF vectors using cosine similarity.

    Centroids are initialised by random sampling without replacement (seeded
    for reproducibility).  Empty clusters are reinitialised to a random
    member vector rather than being left degenerate.

    Parameters
    ----------
    vectors:
        Sparse document vectors as returned by :func:`_tfidf`.
    k:
        Number of clusters.  Clamped to ``len(vectors)`` if larger.
    max_iter:
        Maximum number of Lloyd iterations.
    seed:
        RNG seed for centroid initialisation.

    Returns
    -------
    list of int
        Cluster label (0 … k-1) for each input vector.
    """
    import random

    n = len(vectors)
    if n == 0:
        return []
    k = min(k, n)

    rng = random.Random(seed)
    centroid_indices = rng.sample(range(n), k)
    centroids: List[Dict[str, float]] = [dict(vectors[i]) for i in centroid_indices]

    # Initialise labels to a sentinel that never matches a real assignment so
    # the first-iteration convergence check is never a false positive.
    labels: List[int] = [-1] * n

    for _ in range(max_iter):
        new_labels = [
            max(range(k), key=lambda c: _cosine(vec, centroids[c]))
            for vec in vectors
        ]
        if new_labels == labels:
            break
        labels = new_labels

        # Update centroids as mean of member vectors
        for c in range(k):
            members = [vectors[i] for i, lbl in enumerate(labels) if lbl == c]
            if not members:
                centroids[c] = dict(vectors[rng.randint(0, n - 1)])
                continue
            new_centroid: Dict[str, float] = {}
            for vec in members:
                for t, v in vec.items():
                    new_centroid[t] = new_centroid.get(t, 0.0) + v
            total = len(members)
            centroids[c] = {t: v / total for t, v in new_centroid.items()}

    return labels


# ---------------------------------------------------------------------------
# BibTeX field extraction
# ---------------------------------------------------------------------------

def _bibtex_field(entry: str, field: str) -> str:
    """Extract a named field value from a single BibTeX entry string.

    Handles the three legal BibTeX value syntaxes:

    * Brace-delimited  ``field = {value}``  — supports nested braces.
    * Quote-delimited  ``field = "value"``  — stops at the first unescaped
      closing double-quote.
    * Bare (numeric)   ``field = 2024``     — captured up to whitespace/comma.

    Parameters
    ----------
    entry:
        Raw text of a single ``@type{key, …}`` BibTeX entry.
    field:
        Field name to extract (case-insensitive).

    Returns
    -------
    str
        Extracted value, stripped of leading/trailing whitespace, or ``""``
        if the field is absent.
    """
    pattern = re.compile(rf'\b{re.escape(field)}\s*=\s*', re.IGNORECASE)
    m = pattern.search(entry)
    if not m:
        return ""
    pos = m.end()
    while pos < len(entry) and entry[pos] == " ":
        pos += 1
    if pos >= len(entry):
        return ""

    c = entry[pos]
    if c == "{":
        # Brace-balanced extraction — handles nested LaTeX markup.
        depth = 0
        start = pos + 1
        i = pos
        while i < len(entry):
            if entry[i] == "{":
                depth += 1
            elif entry[i] == "}":
                depth -= 1
                if depth == 0:
                    return entry[start:i].strip()
            i += 1
        return entry[start:].strip()
    elif c == '"':
        end = entry.find('"', pos + 1)
        if end == -1:
            return entry[pos + 1:].strip()
        return entry[pos + 1:end].strip()
    else:
        m2 = re.match(r'([^\s,}\n]+)', entry[pos:])
        return m2.group(1) if m2 else ""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    """Lightweight representation of a single academic paper.

    Attributes
    ----------
    paper_id:
        Unique identifier (BibTeX cite key, row index, etc.).
    title:
        Paper title.
    abstract:
        Full abstract text.
    authors:
        Author list (free-form string, e.g. ``"Smith, J. and Doe, A."``).
    year:
        Publication year as a string.
    venue:
        Journal or conference name.
    doi:
        Digital Object Identifier.
    keywords:
        Author-supplied keywords (semicolon or comma separated).
    """

    paper_id: str
    title: str
    abstract: str = ""
    authors: str = ""
    year: str = ""
    venue: str = ""
    doi: str = ""
    keywords: str = ""

    @property
    def text(self) -> str:
        """Concatenation of title, abstract, and keywords used for clustering."""
        return f"{self.title} {self.abstract} {self.keywords}"

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "keywords": self.keywords,
        }


@dataclass
class Cluster:
    """A single topic cluster produced by :meth:`LitCluster.fit`.

    Attributes
    ----------
    cluster_id:
        Integer cluster index (0-based).
    papers:
        Papers assigned to this cluster.
    top_terms:
        Up to 10 highest-scoring TF-IDF terms in the cluster, ordered by
        aggregate score (most representative first).
    """

    cluster_id: int
    papers: List[Paper] = field(default_factory=list)
    top_terms: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Short human-readable label using the three top terms."""
        terms = ", ".join(self.top_terms[:3]) if self.top_terms else "—"
        return f"Cluster {self.cluster_id}: {terms}"

    def to_dict(self) -> dict:
        """Serialise cluster and all its papers to a plain dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "size": len(self.papers),
            "top_terms": self.top_terms,
            "label": self.label,
            "papers": [p.to_dict() for p in self.papers],
        }


# ---------------------------------------------------------------------------
# Main API class
# ---------------------------------------------------------------------------

class LitCluster:
    """Cluster a collection of academic papers by topic using TF-IDF + k-means.

    Parameters
    ----------
    k:
        Number of clusters to create.  Automatically clamped to the number
        of loaded papers if ``k`` exceeds the corpus size.
    max_iter:
        Maximum number of Lloyd's algorithm iterations (default 100).
    seed:
        Random seed for reproducible centroid initialisation (default 42).
    min_term_freq:
        Minimum number of documents a term must appear in to be included in
        the vocabulary.  Increasing this value filters rare/noisy terms.

    Examples
    --------
    >>> lc = LitCluster.from_bibtex("refs.bib", k=5)
    >>> lc.fit()
    >>> print(lc.summary())
    """

    def __init__(
        self,
        k: int = 5,
        max_iter: int = 100,
        seed: int = 42,
        min_term_freq: int = 2,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be ≥ 1, got {k}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be ≥ 1, got {max_iter}")
        if min_term_freq < 1:
            raise ValueError(f"min_term_freq must be ≥ 1, got {min_term_freq}")

        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.min_term_freq = min_term_freq
        self.papers: List[Paper] = []
        self.clusters: List[Cluster] = []
        self._labels: List[int] = []
        self._vectors: List[Dict[str, float]] = []
        self._vocab: List[str] = []

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: Path, **kwargs) -> "LitCluster":
        """Load papers from a CSV file.

        The CSV must contain a header row.  Recognised column names (all
        optional except at least one of *title* or *abstract*):
        ``paper_id``, ``title``, ``abstract``, ``authors``, ``year``,
        ``venue``, ``doi``, ``keywords``.

        Parameters
        ----------
        path:
            Path to the CSV file.
        **kwargs:
            Forwarded to :class:`LitCluster` constructor
            (``k``, ``max_iter``, ``seed``, ``min_term_freq``).
        """
        obj = cls(**kwargs)
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")
        with path.open(encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                obj.papers.append(Paper(
                    paper_id=row.get("paper_id", str(i)),
                    title=row.get("title", ""),
                    abstract=row.get("abstract", ""),
                    authors=row.get("authors", ""),
                    year=row.get("year", ""),
                    venue=row.get("venue", ""),
                    doi=row.get("doi", ""),
                    keywords=row.get("keywords", ""),
                ))
        return obj

    @classmethod
    def from_jsonl(cls, path: Path, **kwargs) -> "LitCluster":
        """Load papers from a JSON Lines file (one JSON object per line).

        Each object should have the same keys as the CSV columns above.

        Parameters
        ----------
        path:
            Path to the ``.jsonl`` file.
        **kwargs:
            Forwarded to :class:`LitCluster` constructor.
        """
        obj = cls(**kwargs)
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"JSONL file not found: {path}")
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {i}: {exc}") from exc
                obj.papers.append(Paper(
                    paper_id=row.get("paper_id", str(i)),
                    title=row.get("title", ""),
                    abstract=row.get("abstract", ""),
                    authors=row.get("authors", ""),
                    year=row.get("year", ""),
                    venue=row.get("venue", ""),
                    doi=row.get("doi", ""),
                    keywords=row.get("keywords", ""),
                ))
        return obj

    @classmethod
    def from_bibtex(cls, path: Path, **kwargs) -> "LitCluster":
        """Load papers from a BibTeX (``.bib``) file.

        Extracts ``title``, ``abstract``, ``author``, ``year``,
        ``journal``/``booktitle``, ``doi``, and ``keywords`` fields.
        Supports brace-delimited, quote-delimited, and bare field values,
        including nested braces in LaTeX markup.

        Parameters
        ----------
        path:
            Path to the BibTeX file.
        **kwargs:
            Forwarded to :class:`LitCluster` constructor.
        """
        obj = cls(**kwargs)
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"BibTeX file not found: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        entries = re.split(r'(?=@\w+\s*\{)', text)
        for i, entry in enumerate(entries):
            entry = entry.strip()
            if not entry or not entry.startswith("@"):
                continue
            m_key = re.match(r'@\w+\s*\{\s*(\S+?)\s*[,}]', entry)
            key = m_key.group(1) if m_key else str(i)
            obj.papers.append(Paper(
                paper_id=key,
                title=_bibtex_field(entry, "title"),
                abstract=_bibtex_field(entry, "abstract"),
                authors=_bibtex_field(entry, "author"),
                year=_bibtex_field(entry, "year"),
                venue=(
                    _bibtex_field(entry, "journal")
                    or _bibtex_field(entry, "booktitle")
                ),
                doi=_bibtex_field(entry, "doi"),
                keywords=_bibtex_field(entry, "keywords"),
            ))
        return obj

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def fit(self) -> "LitCluster":
        """Run the TF-IDF + k-means clustering pipeline.

        Steps:

        1. Tokenise each paper's ``text`` property.
        2. Remove terms appearing in fewer than ``min_term_freq`` documents.
        3. Compute sparse TF-IDF vectors.
        4. Apply k-means (Lloyd's algorithm, cosine similarity).
        5. Extract the top 10 TF-IDF terms per cluster.
        6. Populate :attr:`clusters`.

        Returns
        -------
        LitCluster
            Returns ``self`` to allow method chaining.
        """
        if not self.papers:
            return self

        tokens_list = [_tokenise(p.text) for p in self.papers]

        if self.min_term_freq > 1:
            freq: Dict[str, int] = {}
            for tokens in tokens_list:
                for t in set(tokens):
                    freq[t] = freq.get(t, 0) + 1
            tokens_list = [
                [t for t in tokens if freq.get(t, 0) >= self.min_term_freq]
                for tokens in tokens_list
            ]

        # Warn if many documents have empty token lists after filtering
        empty_count = sum(1 for t in tokens_list if not t)
        if empty_count == len(self.papers):
            raise ValueError(
                "All papers produced empty token lists. "
                "Try lowering --min-freq or check that the input contains "
                "title/abstract text."
            )

        self._vectors, self._vocab = _tfidf(tokens_list)
        effective_k = min(self.k, len(self.papers))
        if effective_k < self.k:
            import warnings
            warnings.warn(
                f"k={self.k} reduced to {effective_k} (fewer papers than clusters).",
                stacklevel=2,
            )
        self._labels = _kmeans(self._vectors, effective_k, self.max_iter, self.seed)

        cluster_map: Dict[int, List[Paper]] = {}
        for paper, lbl in zip(self.papers, self._labels):
            cluster_map.setdefault(lbl, []).append(paper)

        self.clusters = [
            Cluster(
                cluster_id=cid,
                papers=cluster_map[cid],
                top_terms=self._top_terms_for_cluster(cid, n=10),
            )
            for cid in sorted(cluster_map)
        ]
        return self

    def _top_terms_for_cluster(self, cid: int, n: int = 10) -> List[str]:
        """Return the *n* highest aggregate-TF-IDF terms for cluster *cid*."""
        member_vecs = [
            self._vectors[i]
            for i, lbl in enumerate(self._labels)
            if lbl == cid
        ]
        if not member_vecs:
            return []
        scores: Dict[str, float] = {}
        for vec in member_vecs:
            for t, v in vec.items():
                scores[t] = scores.get(t, 0.0) + v
        return sorted(scores, key=lambda t: -scores[t])[:n]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, path: Path) -> None:
        """Write cluster assignments to a CSV file.

        Each row represents one paper and includes its cluster ID, cluster
        label, and all metadata fields.

        Parameters
        ----------
        path:
            Destination file path.
        """
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "cluster_id", "cluster_label",
                "paper_id", "title", "authors", "year", "venue", "doi",
            ])
            for cluster in self.clusters:
                for p in cluster.papers:
                    writer.writerow([
                        cluster.cluster_id, cluster.label,
                        p.paper_id, p.title, p.authors, p.year, p.venue, p.doi,
                    ])

    def export_json(self, path: Path) -> None:
        """Write full cluster data (including paper metadata) to a JSON file.

        Parameters
        ----------
        path:
            Destination file path.
        """
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                [c.to_dict() for c in self.clusters],
                fh,
                indent=2,
                ensure_ascii=False,
            )

    def summary(self) -> str:
        """Return a human-readable plain-text summary of clustering results."""
        lines = [
            f"litcluster v{__version__}",
            f"{len(self.papers)} papers → {len(self.clusters)} clusters",
            "",
        ]
        for c in self.clusters:
            lines.append(
                f"  [{c.cluster_id:2d}]  {len(c.papers):4d} papers  "
                f"({', '.join(c.top_terms[:5])})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="litcluster",
        description=(
            "Cluster academic papers by topic using TF-IDF + k-means. "
            "Accepts BibTeX (.bib), CSV, or JSON Lines (.jsonl) input."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  litcluster refs.bib -k 8\n"
            "  litcluster papers.csv --format json -o clusters.json\n"
            "  litcluster papers.jsonl -k 10 --format csv\n"
        ),
    )
    parser.add_argument("input", help="Input file: .bib, .csv, or .jsonl")
    parser.add_argument(
        "-k", "--clusters", type=int, default=5, dest="k",
        metavar="K",
        help="Number of clusters (default: 5)",
    )
    parser.add_argument(
        "--format", choices=["csv", "json", "summary"], default="summary",
        help="Output format (default: summary)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        metavar="FILE",
        help="Output file path (default: stdout for summary, auto-named for csv/json)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=100,
        metavar="N",
        help="Maximum k-means iterations (default: 100)",
    )
    parser.add_argument(
        "--min-freq", type=int, default=2,
        metavar="N",
        help="Minimum document frequency for vocabulary terms (default: 2)",
    )
    parser.add_argument(
        "--version", action="version", version=f"litcluster {__version__}",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Entry point for the ``litcluster`` command-line tool.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    args = _parse_args(argv)
    path = Path(args.input)
    if not path.is_file():
        print(f"litcluster: error: file not found: {path}", file=sys.stderr)
        return 1

    kwargs = dict(
        k=args.k,
        max_iter=args.max_iter,
        seed=args.seed,
        min_term_freq=args.min_freq,
    )

    try:
        suffix = path.suffix.lower()
        if suffix == ".bib":
            lc = LitCluster.from_bibtex(path, **kwargs)
        elif suffix == ".jsonl":
            lc = LitCluster.from_jsonl(path, **kwargs)
        else:
            lc = LitCluster.from_csv(path, **kwargs)

        if not lc.papers:
            print("litcluster: warning: no papers loaded from input file.", file=sys.stderr)
            return 1

        lc.fit()
    except (FileNotFoundError, ValueError) as exc:
        print(f"litcluster: error: {exc}", file=sys.stderr)
        return 1

    if args.format == "summary":
        text = lc.summary()
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"Summary written to {args.output}")
        else:
            print(text)
    elif args.format == "csv":
        out = Path(args.output) if args.output else path.with_suffix(".clusters.csv")
        lc.export_csv(out)
        print(f"Clusters written to {out}")
    elif args.format == "json":
        out = Path(args.output) if args.output else path.with_suffix(".clusters.json")
        lc.export_json(out)
        print(f"Clusters written to {out}")

    return 0


# Alias expected by pyproject.toml entry-point ``litcluster = "litcluster:_cli"``
_cli = main


if __name__ == "__main__":
    sys.exit(main())
