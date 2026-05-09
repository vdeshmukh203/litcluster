#!/usr/bin/env python3
"""
litcluster — Literature Clustering Tool
Clusters academic papers by topic using TF-IDF + k-means (pure stdlib).
Zero external dependencies. Python >= 3.8.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__version__ = "0.1.0"
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

_STOPWORDS = {
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
}


def _tokenise(text: str) -> List[str]:
    """Lower-case, extract alphabetic tokens of length >= 3, remove stopwords."""
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _tfidf(
    documents: List[List[str]],
) -> Tuple[List[Dict[str, float]], List[str]]:
    """Compute TF-IDF vectors for a list of token lists.

    Returns ``(vectors, vocab)`` where each vector is a sparse dict mapping
    term to TF-IDF weight.  IDF uses the smoothed formula
    ``log((N + 1) / (df + 1)) + 1``.
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
    """Cosine similarity between two sparse TF-IDF vectors."""
    dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in b)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _kmeans(
    vectors: List[Dict[str, float]],
    k: int,
    max_iter: int = 100,
    seed: int = 42,
) -> List[int]:
    """Lloyd's k-means on sparse TF-IDF vectors.

    Parameters
    ----------
    vectors:
        Sparse TF-IDF representations (one dict per document).
    k:
        Target number of clusters; silently capped at ``len(vectors)``.
    max_iter:
        Maximum iteration count.
    seed:
        Random seed for centroid initialisation.

    Returns
    -------
    List[int]
        Per-document cluster labels (length == ``len(vectors)``).
    """
    import random

    n = len(vectors)
    if n == 0:
        return []
    k = min(k, n)

    rng = random.Random(seed)
    centroid_indices = rng.sample(range(n), k)
    centroids = [dict(vectors[i]) for i in centroid_indices]
    labels = [0] * n

    for _ in range(max_iter):
        new_labels: List[int] = []
        for vec in vectors:
            sims = [_cosine(vec, centroids[c]) for c in range(k)]
            new_labels.append(sims.index(max(sims)))

        if new_labels == labels:
            break
        labels = new_labels

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
# BibTeX helper
# ---------------------------------------------------------------------------

def _bibtex_field(entry: str, field_name: str) -> str:
    """Extract the value of *field_name* from a single BibTeX entry string.

    Handles ``{...}`` and ``"..."`` delimiters and nested braces correctly.
    Returns an empty string when the field is absent.
    """
    pattern = re.compile(
        r'\b' + re.escape(field_name) + r'\s*=\s*',
        re.IGNORECASE,
    )
    m = pattern.search(entry)
    if not m:
        return ""
    start = m.end()
    if start >= len(entry):
        return ""

    delimiter = entry[start]
    if delimiter == '{':
        depth, i = 0, start
        while i < len(entry):
            if entry[i] == '{':
                depth += 1
            elif entry[i] == '}':
                depth -= 1
                if depth == 0:
                    return entry[start + 1:i].strip()
            i += 1
        return ""
    if delimiter == '"':
        i = start + 1
        while i < len(entry):
            if entry[i] == '"':
                return entry[start + 1:i].strip()
            i += 1
        return ""
    # Bare numeric value (e.g. year = 2024)
    numeric = re.match(r'(\d+)', entry[start:])
    return numeric.group(1) if numeric else ""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    """Represents a single scientific paper."""

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
        """Combined text used for vectorisation (title + abstract + keywords)."""
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

    def __repr__(self) -> str:
        return f"Paper(id={self.paper_id!r}, title={self.title[:50]!r})"


@dataclass
class Cluster:
    """A thematic cluster of papers with associated top terms."""

    cluster_id: int
    papers: List[Paper] = field(default_factory=list)
    top_terms: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Short human-readable label: 'Cluster N: term1, term2, term3'."""
        terms = ", ".join(self.top_terms[:3]) if self.top_terms else "—"
        return f"Cluster {self.cluster_id}: {terms}"

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "size": len(self.papers),
            "top_terms": self.top_terms,
            "label": self.label,
            "papers": [p.to_dict() for p in self.papers],
        }

    def __repr__(self) -> str:
        return (
            f"Cluster(id={self.cluster_id}, "
            f"size={len(self.papers)}, terms={self.top_terms[:3]})"
        )


# ---------------------------------------------------------------------------
# LitCluster
# ---------------------------------------------------------------------------

class LitCluster:
    """Cluster a collection of scientific papers using TF-IDF and k-means.

    Parameters
    ----------
    k : int
        Target number of clusters.  Silently capped at corpus size.
    max_iter : int
        Maximum k-means iterations.
    seed : int
        Random seed for reproducibility.
    min_term_freq : int
        Minimum document frequency for a term to enter the vocabulary.
        Terms appearing in fewer documents are discarded before vectorisation.

    Examples
    --------
    >>> lc = LitCluster(k=3, seed=0, min_term_freq=1)
    >>> lc.add_paper(Paper("1", "Deep Learning", "Neural network methods."))
    >>> lc.add_paper(Paper("2", "Clustering", "k-means algorithm for grouping."))
    >>> lc.fit()
    LitCluster(k=3, papers=2, clusters=2)
    """

    def __init__(
        self,
        k: int = 5,
        max_iter: int = 100,
        seed: int = 42,
        min_term_freq: int = 2,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if min_term_freq < 1:
            raise ValueError(f"min_term_freq must be >= 1, got {min_term_freq}")

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
    def from_csv(cls, path, **kwargs) -> "LitCluster":
        """Load papers from a CSV file.

        Expected columns: ``paper_id``, ``title``, ``abstract``, ``authors``,
        ``year``, ``venue``, ``doi``, ``keywords`` (all optional except ``title``).
        Accepts both ``Path`` objects and path strings.
        """
        obj = cls(**kwargs)
        path = Path(path)
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
    def from_jsonl(cls, path, **kwargs) -> "LitCluster":
        """Load papers from a JSONL file (one JSON object per line)."""
        obj = cls(**kwargs)
        path = Path(path)
        with path.open(encoding="utf-8") as fh:
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
    def from_bibtex(cls, path, **kwargs) -> "LitCluster":
        """Load papers from a BibTeX (.bib) file.

        Parses ``title``, ``abstract``, ``author``, ``year``, ``journal``/
        ``booktitle``, ``doi``, and ``keywords`` fields.  Uses brace-aware
        extraction so LaTeX-formatted values are handled correctly.
        """
        obj = cls(**kwargs)
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        entries = re.split(r'(?=@\w+\s*\{)', text)
        for i, entry in enumerate(entries):
            if not entry.strip() or not entry.startswith('@'):
                continue
            m_key = re.match(r'@\w+\s*\{\s*(\S+?)[,}]', entry)
            key = m_key.group(1) if m_key else str(i)
            obj.papers.append(Paper(
                paper_id=key,
                title=_bibtex_field(entry, 'title'),
                abstract=_bibtex_field(entry, 'abstract'),
                authors=_bibtex_field(entry, 'author'),
                year=_bibtex_field(entry, 'year'),
                venue=(
                    _bibtex_field(entry, 'journal')
                    or _bibtex_field(entry, 'booktitle')
                ),
                doi=_bibtex_field(entry, 'doi'),
                keywords=_bibtex_field(entry, 'keywords'),
            ))
        return obj

    def add_paper(self, paper: Paper) -> "LitCluster":
        """Append *paper* to the corpus.

        Returns *self* to support method chaining.  Call :meth:`fit` after
        all papers have been added.
        """
        self.papers.append(paper)
        return self

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self) -> "LitCluster":
        """Run TF-IDF vectorisation followed by k-means clustering.

        Populates :attr:`clusters` with :class:`Cluster` objects sorted by
        cluster ID.  Returns *self* so calls can be chained.
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

        empty_count = sum(1 for t in tokens_list if not t)
        if empty_count:
            warnings.warn(
                f"{empty_count} paper(s) produced empty token lists after "
                "frequency filtering.  Consider lowering --min-freq.",
                UserWarning,
                stacklevel=2,
            )

        self._vectors, self._vocab = _tfidf(tokens_list)
        self._labels = _kmeans(self._vectors, self.k, self.max_iter, self.seed)

        cluster_map: Dict[int, List[Paper]] = {}
        for paper, label in zip(self.papers, self._labels):
            cluster_map.setdefault(label, []).append(paper)

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
        """Return the *n* highest-scoring terms for cluster *cid*."""
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
    # Exports
    # ------------------------------------------------------------------

    def export_csv(self, path) -> None:
        """Write clustering results to *path* as a CSV file."""
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
                        p.paper_id, p.title, p.authors,
                        p.year, p.venue, p.doi,
                    ])

    def export_json(self, path) -> None:
        """Write clustering results to *path* as a JSON file."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                [c.to_dict() for c in self.clusters],
                fh,
                indent=2,
                ensure_ascii=False,
            )

    def summary(self) -> str:
        """Return a human-readable summary of clustering results."""
        if not self.clusters:
            return "No clusters — call .fit() first."
        lines = [
            f"LitCluster  v{__version__}",
            f"  Corpus  : {len(self.papers)} papers",
            f"  Clusters: {len(self.clusters)}",
            f"  Vocab   : {len(self._vocab)} terms",
            "",
        ]
        n_total = max(len(self.papers), 1)
        for c in self.clusters:
            pct = 100.0 * len(c.papers) / n_total
            lines.append(
                f"  [{c.cluster_id:>2d}]  {len(c.papers):>4d} papers"
                f"  ({pct:4.1f}%)  {', '.join(c.top_terms[:5])}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"LitCluster(k={self.k}, "
            f"papers={len(self.papers)}, "
            f"clusters={len(self.clusters)})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="litcluster",
        description="Cluster academic papers by topic using TF-IDF + k-means.",
    )
    p.add_argument("input", help="Input file: CSV, JSONL, or BibTeX (.bib)")
    p.add_argument(
        "-k", "--clusters", type=int, default=5, dest="k",
        help="Number of clusters (default: 5)",
    )
    p.add_argument(
        "--format", choices=["csv", "json", "summary"], default="summary",
        help="Output format (default: summary)",
    )
    p.add_argument("--output", "-o", default=None,
                   help="Output file path (default: stdout / auto-named)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--max-iter", type=int, default=100,
                   help="Maximum k-means iterations (default: 100)")
    p.add_argument(
        "--min-freq", type=int, default=2,
        help="Minimum term document frequency (default: 2)",
    )
    p.add_argument("--version", action="version", version=f"litcluster {__version__}")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """CLI entry point.  Returns an exit code (0 = success)."""
    args = _parse_args(argv)
    path = Path(args.input)
    if not path.is_file():
        print(f"Error: {path} not found", file=sys.stderr)
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
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error loading {path}: {exc}", file=sys.stderr)
        return 1

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


#: Alias kept for backwards-compatibility with older pyproject.toml entries.
_cli = main

if __name__ == "__main__":
    sys.exit(main())
