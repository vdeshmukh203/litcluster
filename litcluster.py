#!/usr/bin/env python3
"""
litcluster — Literature Clustering Tool

Clusters academic papers by topic using TF-IDF + k-means (stdlib only).
Supports BibTeX (.bib), CSV, and JSONL input formats.

Usage
-----
    litcluster papers.bib -k 6 --format json -o results.json
    litcluster papers.csv -k 5 --format summary
    python litcluster.py papers.jsonl -k 4
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
    """Lowercase, extract alpha tokens ≥3 chars, remove stopwords."""
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _tfidf(documents: List[List[str]]) -> Tuple[List[Dict[str, float]], List[str]]:
    """Compute TF-IDF vectors for tokenised documents.

    Uses smoothed IDF: idf(t) = log((N+1)/(df(t)+1)) + 1, which avoids
    zero weights for terms appearing in all documents.

    Returns
    -------
    vectors : sparse dicts mapping term → TF-IDF weight, one per document
    vocab   : sorted list of all unique terms
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
    dot = sum(a.get(t, 0.0) * v for t, v in b.items())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _kmeanspp_init(
    vectors: List[Dict[str, float]],
    k: int,
    rng,
) -> List[int]:
    """K-means++ centroid seeding for better initial spread.

    Selects k diverse starting centroids by choosing each successive
    centroid proportionally to its distance from the nearest already-chosen
    centroid (distance = 1 − cosine similarity).
    """
    n = len(vectors)
    chosen = [rng.randint(0, n - 1)]
    for _ in range(k - 1):
        dists = []
        for i, vec in enumerate(vectors):
            if i in chosen:
                dists.append(0.0)
            else:
                max_sim = max(_cosine(vec, vectors[c]) for c in chosen)
                dists.append(max(0.0, 1.0 - max_sim))
        total = sum(dists)
        if total == 0:
            chosen.append(rng.randint(0, n - 1))
            continue
        r = rng.random() * total
        cumulative = 0.0
        for i, d in enumerate(dists):
            cumulative += d
            if cumulative >= r:
                chosen.append(i)
                break
        else:
            chosen.append(n - 1)
    return chosen


def _kmeans(
    vectors: List[Dict[str, float]],
    k: int,
    max_iter: int = 100,
    seed: int = 42,
) -> List[int]:
    """Lloyd's k-means on sparse TF-IDF vectors using cosine similarity.

    Centres are initialised with k-means++ for better convergence.
    Empty clusters are re-seeded with a random point to avoid degeneracy.

    Returns
    -------
    labels : cluster index for each input vector (length == len(vectors))
    """
    import random

    n = len(vectors)
    if n == 0:
        return []
    k = min(k, n)
    if k == 0:
        return []

    rng = random.Random(seed)
    centroid_indices = _kmeanspp_init(vectors, k, rng)
    centroids = [dict(vectors[i]) for i in centroid_indices]
    labels: List[int] = [-1] * n

    for _ in range(max_iter):
        new_labels = [
            max(range(k), key=lambda c, v=vec: _cosine(v, centroids[c]))
            for vec in vectors
        ]
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
            m = len(members)
            centroids[c] = {t: v / m for t, v in new_centroid.items()}

    return labels


# ---------------------------------------------------------------------------
# BibTeX parser helpers
# ---------------------------------------------------------------------------

def _bibtex_field(entry: str, field_name: str) -> str:
    """Extract the value of a BibTeX field, handling nested braces.

    Supports both brace-delimited  ``field = {value}``  and
    quote-delimited  ``field = "value"``  forms, as well as bare numeric
    values such as  ``year = 2020``.
    """
    pattern = re.compile(
        rf'\b{re.escape(field_name)}\s*=\s*', re.IGNORECASE
    )
    m = pattern.search(entry)
    if not m:
        return ""
    pos = m.end()
    while pos < len(entry) and entry[pos] == ' ':
        pos += 1
    if pos >= len(entry):
        return ""

    start_char = entry[pos]
    if start_char == '{':
        depth = 0
        result: List[str] = []
        for ch in entry[pos:]:
            if ch == '{':
                depth += 1
                if depth > 1:
                    result.append(ch)
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    break
                result.append(ch)
            else:
                result.append(ch)
        return ''.join(result).strip()
    elif start_char == '"':
        end = entry.find('"', pos + 1)
        if end == -1:
            return ""
        return entry[pos + 1:end].strip()
    else:
        m2 = re.match(r'[\w\-.]+', entry[pos:])
        return m2.group(0) if m2 else ""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    """A single academic paper with bibliographic metadata.

    Parameters
    ----------
    paper_id : str
        Unique identifier (BibTeX key, row index, or explicit id).
    title : str
        Paper title (required for meaningful clustering).
    abstract : str
        Abstract text.
    authors : str
        Author list as a single string.
    year : str
        Publication year.
    venue : str
        Journal or conference name.
    doi : str
        Digital Object Identifier.
    keywords : str
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
        """Text fed to TF-IDF: title + abstract + keywords."""
        return f"{self.title} {self.abstract} {self.keywords}"

    def to_dict(self) -> dict:
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
    """A cluster of thematically related papers.

    Parameters
    ----------
    cluster_id : int
        Cluster index (0-based).
    papers : list of Paper
        Papers assigned to this cluster.
    top_terms : list of str
        Highest TF-IDF terms characterising this cluster.
    """

    cluster_id: int
    papers: List[Paper] = field(default_factory=list)
    top_terms: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Short descriptive label using the top-3 terms."""
        return f"Cluster {self.cluster_id}: {', '.join(self.top_terms[:3])}"

    def to_dict(self) -> dict:
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
    """Topic clustering of academic papers using TF-IDF and k-means.

    Ingests BibTeX, CSV, or JSONL files, builds sparse TF-IDF vectors from
    title + abstract + keywords, and partitions papers into *k* topical
    clusters using Lloyd's algorithm with k-means++ initialisation.

    Parameters
    ----------
    k : int
        Number of clusters (must be ≥ 1).
    max_iter : int
        Maximum k-means iterations (must be ≥ 1).
    seed : int
        Random seed for reproducibility.
    min_term_freq : int
        Minimum document frequency for a term to enter the vocabulary
        (must be ≥ 1).  Raising this value prunes very rare terms and
        can improve cluster coherence for larger corpora.

    Examples
    --------
    >>> lc = LitCluster.from_csv(Path("papers.csv"), k=4)
    >>> lc.fit()
    >>> print(lc.summary())
    """

    def __init__(
        self,
        k: int = 5,
        max_iter: int = 100,
        seed: int = 42,
        min_term_freq: int = 2,
    ):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k!r}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")
        if min_term_freq < 1:
            raise ValueError(f"min_term_freq must be >= 1, got {min_term_freq!r}")

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

        Expected columns (all optional except *title*): ``paper_id``,
        ``title``, ``abstract``, ``authors``, ``year``, ``venue``,
        ``doi``, ``keywords``.
        """
        obj = cls(**kwargs)
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
        """Load papers from a JSONL file (one JSON object per line).

        Expected keys match the CSV column names above.
        """
        obj = cls(**kwargs)
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
    def from_bibtex(cls, path: Path, **kwargs) -> "LitCluster":
        """Load papers from a BibTeX (.bib) file.

        Parses ``@article``, ``@inproceedings``, ``@misc``, etc. entries.
        Supports brace- and quote-delimited field values and handles
        nested braces (e.g. ``title = {The {ABC} Method}``).
        """
        obj = cls(**kwargs)
        text = path.read_text(encoding="utf-8", errors="replace")
        entries = re.split(r'(?=@\w+\s*\{)', text)
        for i, entry in enumerate(entries):
            if not entry.strip() or not entry.startswith('@'):
                continue
            m_key = re.match(r'@\w+\s*\{\s*([^\s,{]+)', entry)
            key = m_key.group(1) if m_key else str(i)
            venue = (_bibtex_field(entry, 'journal') or
                     _bibtex_field(entry, 'booktitle'))
            obj.papers.append(Paper(
                paper_id=key,
                title=_bibtex_field(entry, 'title'),
                abstract=_bibtex_field(entry, 'abstract'),
                authors=_bibtex_field(entry, 'author'),
                year=_bibtex_field(entry, 'year'),
                venue=venue,
                doi=_bibtex_field(entry, 'doi'),
                keywords=_bibtex_field(entry, 'keywords'),
            ))
        return obj

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def fit(self) -> "LitCluster":
        """Vectorise papers with TF-IDF and cluster with k-means.

        Returns
        -------
        self : LitCluster
            Allows method chaining: ``LitCluster.from_csv(p).fit()``.

        Notes
        -----
        Papers with an empty ``text`` property (no title, abstract, or
        keywords) produce zero vectors and are assigned to whatever
        cluster their zero vector maps to; they do not affect centroid
        computation.
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

        clusters_map: Dict[int, List[Paper]] = {}
        for paper, lbl in zip(self.papers, self._labels):
            clusters_map.setdefault(lbl, []).append(paper)

        self.clusters = [
            Cluster(
                cluster_id=cid,
                papers=clusters_map[cid],
                top_terms=self._top_terms_for_cluster(cid, n=10),
            )
            for cid in sorted(clusters_map)
        ]
        return self

    def _top_terms_for_cluster(self, cid: int, n: int = 10) -> List[str]:
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

        Columns: ``cluster_id``, ``cluster_label``, ``paper_id``,
        ``title``, ``authors``, ``year``, ``venue``, ``doi``.
        """
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["cluster_id", "cluster_label", "paper_id",
                        "title", "authors", "year", "venue", "doi"])
            for cluster in self.clusters:
                for p in cluster.papers:
                    w.writerow([
                        cluster.cluster_id, cluster.label, p.paper_id,
                        p.title, p.authors, p.year, p.venue, p.doi,
                    ])

    def export_json(self, path: Path) -> None:
        """Write clusters and their papers to a JSON file."""
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                [c.to_dict() for c in self.clusters],
                fh,
                indent=2,
                ensure_ascii=False,
            )

    def summary(self) -> str:
        """Return a plain-text summary of clustering results."""
        lines = [
            f"LitCluster v{__version__}: "
            f"{len(self.papers)} papers → {len(self.clusters)} clusters",
            "",
        ]
        for c in self.clusters:
            lines.append(f"  [{c.cluster_id}] {c.label}  ({len(c.papers)} papers)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="litcluster",
        description=(
            "Cluster academic papers by topic using TF-IDF + k-means.\n"
            "Input can be a BibTeX (.bib), CSV, or JSONL file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="Input file (.bib, .csv, or .jsonl)")
    p.add_argument(
        "-k", "--clusters", type=int, default=5, dest="k",
        help="Number of clusters (default: 5)",
    )
    p.add_argument(
        "--format", choices=["csv", "json", "summary"], default="summary",
        help="Output format (default: summary)",
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Output file path (default: stdout for summary, auto-named for csv/json)",
    )
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--max-iter", type=int, default=100,
                   help="Maximum k-means iterations (default: 100)")
    p.add_argument(
        "--min-freq", type=int, default=2,
        help="Minimum document frequency for vocabulary terms (default: 2)",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """CLI entry point. Returns 0 on success, 1 on error."""
    args = _parse_args(argv)

    path = Path(args.input)
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
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
            print(f"Error: no papers found in {path}", file=sys.stderr)
            return 1

        lc.fit()

    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.format == "summary":
        output = lc.summary()
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"Summary written to {args.output}")
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


_cli = main  # alias used by pyproject.toml entry point


if __name__ == "__main__":
    sys.exit(main())
