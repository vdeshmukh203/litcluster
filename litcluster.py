#!/usr/bin/env python3
"""
litcluster — Literature Clustering Tool

Clusters academic papers by topic using TF-IDF + k-means (pure Python stdlib).
Accepts BibTeX (.bib), CSV, and JSONL input files; exports CSV, JSON, or plain-text
summary output.

Typical usage::

    # Python API
    from litcluster import LitCluster
    lc = LitCluster.from_bibtex("refs.bib", k=6).fit()
    print(lc.summary())
    lc.export_csv("clusters.csv")

    # CLI
    $ litcluster refs.bib -k 6 --format csv -o clusters.csv
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

__all__ = ["Paper", "Cluster", "LitCluster"]


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'this', 'that', 'these', 'those',
    'it', 'its', 'we', 'our', 'they', 'their', 'as', 'if', 'not', 'no',
    'nor', 'so', 'yet', 'both', 'either', 'whether', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'than', 'too', 'very', 'just', 'also',
    'only', 'then', 'here', 'there', 'when', 'where', 'who', 'which', 'how',
    'all', 'any', 'can', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'out', 'off', 'over', 'under', 'again',
    'further', 'once', 'i', 'my', 'me', 'he', 'she', 'his', 'her', 'him',
    'you', 'your',
})


def _tokenise(text: str) -> List[str]:
    """Lowercase, extract alphabetic tokens of length ≥ 3, and remove stopwords.

    Parameters
    ----------
    text:
        Raw input string (title, abstract, keywords).

    Returns
    -------
    List[str]
        Filtered list of word tokens.
    """
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _tfidf(documents: List[List[str]]) -> Tuple[List[Dict[str, float]], List[str]]:
    """Compute smoothed TF-IDF sparse vectors for a list of tokenised documents.

    Uses the formula ``tf * (log((N+1)/(df+1)) + 1)`` where *N* is the number
    of documents and *df* is the document frequency of a term — matching
    scikit-learn's ``TfidfTransformer(smooth_idf=True)`` behaviour.

    Parameters
    ----------
    documents:
        List of token lists, one per document.

    Returns
    -------
    vectors:
        Sparse TF-IDF vector per document (``Dict[term, weight]``).
    vocab:
        Sorted list of all terms in the corpus.
    """
    n = len(documents)
    if n == 0:
        return [], []

    vocab_set: set = set()
    for doc in documents:
        vocab_set.update(doc)
    vocab = sorted(vocab_set)
    term_idx = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)

    # Document frequency counts
    df = [0] * V
    for doc in documents:
        for t in set(doc):
            if t in term_idx:
                df[term_idx[t]] += 1

    vectors: List[Dict[str, float]] = []
    for doc in documents:
        tf: Dict[int, int] = {}
        for t in doc:
            if t in term_idx:
                idx = term_idx[t]
                tf[idx] = tf.get(idx, 0) + 1
        vec: Dict[str, float] = {}
        doc_len = len(doc) or 1
        for idx, count in tf.items():
            term = vocab[idx]
            tf_val = count / doc_len
            idf_val = math.log((n + 1) / (df[idx] + 1)) + 1.0
            vec[term] = tf_val * idf_val
        vectors.append(vec)
    return vectors, vocab


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Compute cosine similarity between two sparse TF-IDF vectors.

    Parameters
    ----------
    a, b:
        Sparse vectors represented as ``{term: weight}`` dicts.

    Returns
    -------
    float
        Cosine similarity in ``[0, 1]``; returns ``0.0`` for zero vectors.
    """
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
    """Lloyd's k-means clustering on sparse cosine-similarity vectors.

    Centroids are initialised by sampling *k* distinct documents at random
    (seeded for reproducibility). Empty clusters are reinitialised to a random
    document to prevent degenerate solutions.

    Parameters
    ----------
    vectors:
        Sparse TF-IDF vectors to cluster.
    k:
        Desired number of clusters; silently capped at ``len(vectors)``.
    max_iter:
        Maximum number of Lloyd iterations before early stopping.
    seed:
        Random seed for reproducible centroid initialisation.

    Returns
    -------
    List[int]
        Cluster label (0-indexed) for each input vector.
    """
    import random

    n = len(vectors)
    if n == 0:
        return []
    k = min(k, n)

    rng = random.Random(seed)
    centroid_indices = rng.sample(range(n), k)
    centroids = [dict(vectors[i]) for i in centroid_indices]
    labels: List[int] = [0] * n

    for _ in range(max_iter):
        new_labels = [
            max(range(k), key=lambda c: _cosine(vec, centroids[c]))
            for vec in vectors
        ]
        if new_labels == labels:
            break
        labels = new_labels

        for c in range(k):
            members = [vectors[i] for i, lbl in enumerate(labels) if lbl == c]
            if not members:
                # Reinitialise empty cluster to a random document
                centroids[c] = dict(vectors[rng.randint(0, n - 1)])
                continue
            new_centroid: Dict[str, float] = {}
            for vec in members:
                for t, v in vec.items():
                    new_centroid[t] = new_centroid.get(t, 0.0) + v
            m = len(members)
            centroids[c] = {t: v / m for t, v in new_centroid.items()}

    return labels


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    """Bibliographic metadata for a single paper.

    Parameters
    ----------
    paper_id:
        Unique identifier (BibTeX key, row index, etc.).
    title:
        Paper title.
    abstract:
        Abstract text used for clustering.
    authors:
        Author list as a single string (e.g. ``"Smith, J. and Doe, A."``).
    year:
        Publication year.
    venue:
        Journal or conference name.
    doi:
        Digital Object Identifier.
    keywords:
        Author-supplied keywords.
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
        """Concatenated title, abstract, and keywords used as clustering input."""
        return f"{self.title} {self.abstract} {self.keywords}"

    def to_dict(self) -> dict:
        """Return a plain-dict representation suitable for JSON serialisation."""
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
    """A single topic cluster produced by :class:`LitCluster`.

    Parameters
    ----------
    cluster_id:
        Zero-based cluster index.
    papers:
        Papers assigned to this cluster.
    top_terms:
        High-scoring TF-IDF terms that characterise the cluster topic.
    """

    cluster_id: int
    papers: List[Paper] = field(default_factory=list)
    top_terms: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Short human-readable label: ``"Cluster N: term1, term2, term3"``."""
        return f"Cluster {self.cluster_id}: {', '.join(self.top_terms[:3])}"

    def to_dict(self) -> dict:
        """Return a plain-dict representation suitable for JSON serialisation."""
        return {
            "cluster_id": self.cluster_id,
            "size": len(self.papers),
            "top_terms": self.top_terms,
            "label": self.label,
            "papers": [p.to_dict() for p in self.papers],
        }


# ---------------------------------------------------------------------------
# LitCluster
# ---------------------------------------------------------------------------

class LitCluster:
    """Cluster a collection of academic papers by topic using TF-IDF and k-means.

    The full pipeline is:

    1. Parse input (BibTeX / CSV / JSONL) into :class:`Paper` objects.
    2. Tokenise each paper's title + abstract + keywords.
    3. Optionally drop terms that appear in fewer than *min_term_freq* documents.
    4. Compute smoothed TF-IDF vectors.
    5. Run Lloyd's k-means with cosine similarity.
    6. Extract top TF-IDF terms per cluster as topic labels.

    Parameters
    ----------
    k:
        Number of clusters.
    max_iter:
        Maximum k-means iterations.
    seed:
        Random seed for reproducible centroid initialisation.
    min_term_freq:
        Minimum document frequency for a term to be included in the
        vocabulary.  Setting this to 1 disables filtering.

    Examples
    --------
    >>> from litcluster import LitCluster
    >>> lc = LitCluster.from_bibtex("refs.bib", k=6).fit()
    >>> print(lc.summary())
    >>> lc.export_csv("clusters.csv")
    """

    def __init__(
        self,
        k: int = 5,
        max_iter: int = 100,
        seed: int = 42,
        min_term_freq: int = 2,
    ) -> None:
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
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: Path, **kwargs) -> "LitCluster":
        """Load papers from a CSV file.

        Expected columns (any subset is accepted; missing columns default to
        empty strings): ``paper_id``, ``title``, ``abstract``, ``authors``,
        ``year``, ``venue``, ``doi``, ``keywords``.

        Parameters
        ----------
        path:
            Path to the ``.csv`` file.
        **kwargs:
            Forwarded to :class:`LitCluster` constructor (e.g. ``k=8``).
        """
        obj = cls(**kwargs)
        with Path(path).open(encoding="utf-8", errors="replace", newline="") as fh:
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

        Each line should be a JSON object with optional keys matching the
        :class:`Paper` dataclass fields.

        Parameters
        ----------
        path:
            Path to the ``.jsonl`` file.
        **kwargs:
            Forwarded to :class:`LitCluster` constructor.
        """
        obj = cls(**kwargs)
        with Path(path).open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
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
        """Load papers from a BibTeX file.

        Parses ``title``, ``abstract``, ``author``, ``year``,
        ``journal``/``booktitle``, ``doi``, and ``keywords`` fields.

        .. note::
            The parser uses regular expressions and handles most well-formed
            BibTeX files; highly nested or non-standard entries may be
            silently truncated.

        Parameters
        ----------
        path:
            Path to the ``.bib`` file.
        **kwargs:
            Forwarded to :class:`LitCluster` constructor.
        """
        obj = cls(**kwargs)
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        entries = re.split(r'(?=@\w+\s*\{)', text)
        for i, entry in enumerate(entries):
            if not entry.strip() or not entry.startswith('@'):
                continue

            def _get(f: str) -> str:
                m = re.search(
                    rf'{f}\s*=\s*[{{"](.*?)[}}"]([\s,}}]|$)',
                    entry, re.IGNORECASE | re.DOTALL,
                )
                return m.group(1).strip() if m else ""

            m_key = re.match(r'@\w+\s*\{\s*(\S+?)[,}]', entry)
            key = m_key.group(1) if m_key else str(i)
            obj.papers.append(Paper(
                paper_id=key,
                title=_get('title'),
                abstract=_get('abstract'),
                authors=_get('author'),
                year=_get('year'),
                venue=_get('journal') or _get('booktitle'),
                doi=_get('doi'),
                keywords=_get('keywords'),
            ))
        return obj

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self) -> "LitCluster":
        """Run the full TF-IDF + k-means clustering pipeline.

        Tokenises each paper's text, builds TF-IDF vectors (with optional
        rare-term filtering), clusters with Lloyd's algorithm, and populates
        :attr:`clusters`.

        Returns
        -------
        LitCluster
            *self*, to enable method chaining.
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

        self._vectors, self._vocab = _tfidf(tokens_list)
        self._labels = _kmeans(self._vectors, self.k, self.max_iter, self.seed)

        cluster_map: Dict[int, List[Paper]] = {}
        for paper, lbl in zip(self.papers, self._labels):
            cluster_map.setdefault(lbl, []).append(paper)

        self.clusters = []
        for cid in sorted(cluster_map):
            top_terms = self._top_terms_for_cluster(cid, n=10)
            self.clusters.append(
                Cluster(cluster_id=cid, papers=cluster_map[cid], top_terms=top_terms)
            )
        return self

    def _top_terms_for_cluster(self, cid: int, n: int = 10) -> List[str]:
        """Return the *n* highest aggregate TF-IDF terms for cluster *cid*."""
        member_vecs = [
            self._vectors[i] for i, lbl in enumerate(self._labels) if lbl == cid
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
        """Write clustering results to a CSV file.

        Each row corresponds to one paper. Columns: ``cluster_id``,
        ``cluster_label``, ``paper_id``, ``title``, ``authors``, ``year``,
        ``venue``, ``doi``.

        Parameters
        ----------
        path:
            Destination file path.
        """
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([
                "cluster_id", "cluster_label", "paper_id",
                "title", "authors", "year", "venue", "doi",
            ])
            for cluster in self.clusters:
                for p in cluster.papers:
                    w.writerow([
                        cluster.cluster_id, cluster.label, p.paper_id,
                        p.title, p.authors, p.year, p.venue, p.doi,
                    ])

    def export_json(self, path: Path) -> None:
        """Write clustering results to a JSON file.

        The output is a JSON array of cluster objects; each cluster object
        contains ``cluster_id``, ``size``, ``top_terms``, ``label``, and a
        ``papers`` array.

        Parameters
        ----------
        path:
            Destination file path.
        """
        with Path(path).open("w", encoding="utf-8") as fh:
            json.dump(
                [c.to_dict() for c in self.clusters],
                fh, indent=2, ensure_ascii=False,
            )

    def summary(self) -> str:
        """Return a plain-text summary of the clustering results.

        Returns
        -------
        str
            Multi-line string listing each cluster's ID, topic label, and
            paper count.
        """
        lines = [
            f"LitCluster: {len(self.papers)} papers in {len(self.clusters)} clusters",
            "",
        ]
        for c in self.clusters:
            lines.append(f"  [{c.cluster_id}] {c.label} ({len(c.papers)} papers)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    """Build and parse the command-line argument parser."""
    p = argparse.ArgumentParser(
        prog="litcluster",
        description=(
            "Cluster academic papers by topic using TF-IDF + k-means. "
            "Accepts BibTeX (.bib), CSV, or JSONL input."
        ),
    )
    p.add_argument("input", help="Input file: CSV, JSONL, or .bib")
    p.add_argument(
        "-k", "--clusters", type=int, default=5, dest="k",
        help="Number of clusters (default: 5)",
    )
    p.add_argument(
        "--format", choices=["csv", "json", "summary"], default="summary",
        help="Output format (default: summary)",
    )
    p.add_argument("--output", "-o", default=None, help="Output file (default: stdout/auto)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--max-iter", type=int, default=100, help="Max k-means iterations (default: 100)")
    p.add_argument(
        "--min-freq", type=int, default=2,
        help="Minimum document frequency for vocabulary terms (default: 2)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Entry point for the ``litcluster`` CLI command.

    Parameters
    ----------
    argv:
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 on success, 1 on error).
    """
    args = _parse_args(argv)
    path = Path(args.input)
    if not path.is_file():
        print(f"Error: {path} not found", file=sys.stderr)
        return 1

    suffix = path.suffix.lower()
    kwargs = dict(
        k=args.k, max_iter=args.max_iter,
        seed=args.seed, min_term_freq=args.min_freq,
    )
    if suffix == ".bib":
        lc = LitCluster.from_bibtex(path, **kwargs)
    elif suffix == ".jsonl":
        lc = LitCluster.from_jsonl(path, **kwargs)
    else:
        lc = LitCluster.from_csv(path, **kwargs)

    lc.fit()

    if args.format == "summary":
        output = lc.summary()
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
        else:
            print(output)
    elif args.format == "csv":
        out = Path(args.output) if args.output else path.with_suffix(".clusters.csv")
        lc.export_csv(out)
        print(f"Clusters written to {out}")
    elif args.format == "json":
        out = Path(args.output) if args.output else path.with_suffix(".clusters.json")
        lc.export_json(out)
        print(f"Clusters written to {out}")

    return 0


# Required by pyproject.toml entry point: litcluster = "litcluster:_cli"
_cli = main


if __name__ == "__main__":
    sys.exit(main())
