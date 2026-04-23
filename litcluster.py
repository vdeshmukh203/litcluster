#!/usr/bin/env python3
"""
litcluster.py — Literature Clustering Tool
Clusters academic papers by topic using TF-IDF + k-means (pure stdlib).
Stdlib-only. No external dependencies.
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


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

_STOPWORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'by','from','is','was','are','were','be','been','being','have','has',
    'had','do','does','did','will','would','could','should','may','might',
    'this','that','these','those','it','its','we','our','they','their',
    'as','if','not','no','nor','so','yet','both','either','whether',
    'each','few','more','most','other','some','such','than','too','very',
    'just','also','only','then','here','there','when','where','who','which',
    'how','all','any','can','into','through','during','before','after',
    'above','below','between','out','off','over','under','again','further',
    'once','i','my','me','he','she','his','her','him','you','your',
}


def _tokenise(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _tfidf(documents: List[List[str]]) -> Tuple[List[Dict[str,float]], List[str]]:
    """Compute TF-IDF vectors. Returns (vectors, vocab)."""
    n = len(documents)
    if n == 0:
        return [], []

    # Build vocab
    vocab_set: set = set()
    for doc in documents:
        vocab_set.update(doc)
    vocab = sorted(vocab_set)
    term_idx = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)

    # Document frequency
    df = [0] * V
    for doc in documents:
        doc_set = set(doc)
        for t in doc_set:
            if t in term_idx:
                df[term_idx[t]] += 1

    # TF-IDF
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
    dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in b)
    norm_a = math.sqrt(sum(v*v for v in a.values()))
    norm_b = math.sqrt(sum(v*v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _kmeans(
    vectors: List[Dict[str, float]],
    k: int,
    max_iter: int = 100,
    seed: int = 42,
) -> List[int]:
    """Lloyd's k-means on sparse TF-IDF vectors."""
    n = len(vectors)
    if n == 0:
        return []
    k = min(k, n)

    # Seeded centroid initialisation (evenly spaced)
    import random
    rng = random.Random(seed)
    centroid_indices = rng.sample(range(n), k)
    centroids = [dict(vectors[i]) for i in centroid_indices]
    labels = [0] * n

    for _ in range(max_iter):
        # Assignment
        new_labels = []
        for vec in vectors:
            best = max(range(k), key=lambda c: _cosine(vec, centroids[c]))
            new_labels.append(best)

        if new_labels == labels:
            break
        labels = new_labels

        # Update centroids
        for c in range(k):
            members = [vectors[i] for i, l in enumerate(labels) if l == c]
            if not members:
                centroids[c] = dict(vectors[rng.randint(0, n-1)])
                continue
            new_centroid: Dict[str, float] = {}
            for vec in members:
                for t, v in vec.items():
                    new_centroid[t] = new_centroid.get(t, 0.0) + v
            total = len(members)
            centroids[c] = {t: v/total for t, v in new_centroid.items()}

    return labels


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Paper:
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
        return f"{self.title} {self.abstract} {self.keywords}"

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id, "title": self.title, "abstract": self.abstract,
            "authors": self.authors, "year": self.year, "venue": self.venue,
            "doi": self.doi, "keywords": self.keywords,
        }


@dataclass
class Cluster:
    cluster_id: int
    papers: List[Paper] = field(default_factory=list)
    top_terms: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
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
    def __init__(self, k: int = 5, max_iter: int = 100, seed: int = 42,
                 min_term_freq: int = 2):
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.min_term_freq = min_term_freq
        self.papers: List[Paper] = []
        self.clusters: List[Cluster] = []
        self._labels: List[int] = []
        self._vectors: List[Dict[str, float]] = []
        self._vocab: List[str] = []

    @classmethod
    def from_csv(cls, path: Path, **kwargs) -> "LitCluster":
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
        """Parse a BibTeX file for titles, abstracts, keywords."""
        obj = cls(**kwargs)
        text = path.read_text(encoding="utf-8", errors="replace")
        entries = re.split(r'(?=@\w+\s*\{)', text)
        for i, entry in enumerate(entries):
            if not entry.strip() or not entry.startswith('@'):
                continue
            def _get(field):
                m = re.search(rf'{field}\s*=\s*[{{"](.*?)[}}"\,]', entry, re.IGNORECASE | re.DOTALL)
                return m.group(1).strip() if m else ""
            m_key = re.match(r'@\w+\s*\{\s*(\S+?)[,}]', entry)
            key = m_key.group(1) if m_key else str(i)
            obj.papers.append(Paper(
                paper_id=key, title=_get('title'), abstract=_get('abstract'),
                authors=_get('author'), year=_get('year'),
                venue=_get('journal') or _get('booktitle'),
                doi=_get('doi'), keywords=_get('keywords'),
            ))
        return obj

    def fit(self) -> "LitCluster":
        if not self.papers:
            return self
        tokens_list = [_tokenise(p.text) for p in self.papers]

        # Filter rare terms
        if self.min_term_freq > 1:
            freq: Dict[str, int] = {}
            for tokens in tokens_list:
                for t in set(tokens):
                    freq[t] = freq.get(t, 0) + 1
            tokens_list = [[t for t in tokens if freq.get(t, 0) >= self.min_term_freq]
                           for tokens in tokens_list]

        self._vectors, self._vocab = _tfidf(tokens_list)
        self._labels = _kmeans(self._vectors, self.k, self.max_iter, self.seed)

        # Build cluster objects
        clusters: Dict[int, List] = {}
        for paper, label in zip(self.papers, self._labels):
            clusters.setdefault(label, []).append(paper)

        self.clusters = []
        for cid in sorted(clusters):
            top_terms = self._top_terms_for_cluster(cid, n=10)
            self.clusters.append(Cluster(cluster_id=cid, papers=clusters[cid], top_terms=top_terms))
        return self

    def _top_terms_for_cluster(self, cid: int, n: int = 10) -> List[str]:
        member_vecs = [self._vectors[i] for i, l in enumerate(self._labels) if l == cid]
        if not member_vecs:
            return []
        scores: Dict[str, float] = {}
        for vec in member_vecs:
            for t, v in vec.items():
                scores[t] = scores.get(t, 0.0) + v
        return sorted(scores, key=lambda t: -scores[t])[:n]

    def export_csv(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["cluster_id","cluster_label","paper_id","title","authors","year","venue","doi"])
            for cluster in self.clusters:
                for p in cluster.papers:
                    w.writerow([cluster.cluster_id, cluster.label, p.paper_id,
                                p.title, p.authors, p.year, p.venue, p.doi])

    def export_json(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            json.dump([c.to_dict() for c in self.clusters], fh, indent=2, ensure_ascii=False)

    def summary(self) -> str:
        lines = [f"LitCluster: {len(self.papers)} papers in {len(self.clusters)} clusters", ""]
        for c in self.clusters:
            lines.append(f"  [{c.cluster_id}] {c.label} ({len(c.papers)} papers)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="litcluster",
                                description="Cluster academic papers by topic using TF-IDF + k-means.")
    p.add_argument("input", help="Input file: CSV, JSONL, or .bib")
    p.add_argument("-k", "--clusters", type=int, default=5, dest="k",
                   help="Number of clusters (default: 5)")
    p.add_argument("--format", choices=["csv","json","summary"], default="summary")
    p.add_argument("--output", "-o", default=None, help="Output file (default: stdout/auto)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--min-freq", type=int, default=2,
                   help="Minimum term frequency to include in vocabulary (default: 2)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    path = Path(args.input)
    if not path.is_file():
        print(f"Error: {path} not found", file=sys.stderr)
        return 1

    suffix = path.suffix.lower()
    kwargs = dict(k=args.k, max_iter=args.max_iter, seed=args.seed, min_term_freq=args.min_freq)
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


if __name__ == "__main__":
    sys.exit(main())
