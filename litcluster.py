#!/usr/bin/env python3
"""
litcluster — Literature Clustering Tool
========================================
Clusters academic papers by topic using TF-IDF + k-means (pure stdlib).

Usage
-----
  litcluster refs.bib -k 5 --format html -o report.html
  litcluster papers.csv -k 8 --format csv
  python -m litcluster refs.bib
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

__version__ = "0.1.0"
__all__ = ["LitCluster", "Paper", "Cluster"]


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

_STOPWORDS = {
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
    "you", "your",
}


def _tokenise(text: str) -> List[str]:
    """Tokenise *text* into lowercase alphabetic tokens, removing stopwords."""
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _tfidf(
    documents: List[List[str]],
    min_freq: int = 1,
) -> Tuple[List[Dict[str, float]], List[str]]:
    """Compute smoothed TF-IDF vectors.

    Parameters
    ----------
    documents:
        Token lists, one per document.
    min_freq:
        Minimum document-frequency for a term to be included in the vocabulary.

    Returns
    -------
    vectors:
        Sparse TF-IDF vector per document (dict mapping term to weight).
    vocab:
        Sorted list of vocabulary terms.
    """
    n = len(documents)
    if n == 0:
        return [], []

    df: Dict[str, int] = {}
    for doc in documents:
        for t in set(doc):
            df[t] = df.get(t, 0) + 1

    vocab = sorted(t for t, c in df.items() if c >= min_freq)
    if not vocab:
        return [{} for _ in documents], []
    term_idx = {t: i for i, t in enumerate(vocab)}

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
            idf_val = math.log((n + 1) / (df[term] + 1)) + 1.0
            vec[term] = tf_val * idf_val
        vectors.append(vec)
    return vectors, vocab


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
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
    """Lloyd's k-means clustering on sparse cosine-distance vectors.

    Parameters
    ----------
    vectors:
        Sparse TF-IDF vectors, one per document.
    k:
        Desired number of clusters (clamped to ``len(vectors)``).
    max_iter:
        Maximum number of Lloyd iterations.
    seed:
        Random seed for reproducible centroid initialisation.

    Returns
    -------
    labels : list[int]
        Cluster index for each document.
    """
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
            members = [vectors[i] for i, lb in enumerate(labels) if lb == c]
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


def _silhouette_score(
    vectors: List[Dict[str, float]], labels: List[int]
) -> float:
    """Mean silhouette coefficient for a clustering result.

    Computes pairwise cosine distances (O(n²)).  For corpora larger than a
    few thousand papers, consider sampling before calling this function.

    Returns a value in [-1, 1]; values close to +1 indicate well-separated
    clusters.  Returns 0.0 when fewer than two distinct clusters are present.
    """
    n = len(vectors)
    if n < 2:
        return 0.0
    cluster_ids = set(labels)
    if len(cluster_ids) < 2:
        return 0.0

    cluster_members: Dict[int, List[int]] = {c: [] for c in cluster_ids}
    for i, lb in enumerate(labels):
        cluster_members[lb].append(i)

    scores: List[float] = []
    for i in range(n):
        same = [j for j in cluster_members[labels[i]] if j != i]
        if not same:
            scores.append(0.0)
            continue
        a = sum(1.0 - _cosine(vectors[i], vectors[j]) for j in same) / len(same)
        other = [
            c for c in cluster_ids if c != labels[i] and cluster_members[c]
        ]
        if not other:
            scores.append(0.0)
            continue
        b = min(
            sum(1.0 - _cosine(vectors[i], vectors[j]) for j in cluster_members[c])
            / len(cluster_members[c])
            for c in other
        )
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0.0 else 0.0)

    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# BibTeX field extractor
# ---------------------------------------------------------------------------

def _bibtex_field(entry: str, field_name: str) -> str:
    """Extract a field value from a BibTeX entry.

    Handles ``field = {braced value}``, ``field = "quoted value"``, and bare
    numeric values.  Correctly handles nested braces inside ``{…}`` values.
    """
    pat = re.compile(rf"\b{re.escape(field_name)}\s*=\s*", re.IGNORECASE)
    m = pat.search(entry)
    if not m:
        return ""
    rest = entry[m.end():].lstrip()
    if rest.startswith("{"):
        depth = 0
        for i, ch in enumerate(rest):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return rest[1:i].strip()
        return ""
    if rest.startswith('"'):
        end = rest.find('"', 1)
        return rest[1:end].strip() if end != -1 else ""
    m2 = re.match(r"(\w+)", rest)
    return m2.group(1) if m2 else ""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    """A single scientific paper with bibliographic metadata."""

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
    """A thematic cluster of papers."""

    cluster_id: int
    papers: List[Paper] = field(default_factory=list)
    top_terms: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Short human-readable label derived from top terms."""
        terms = ", ".join(self.top_terms[:3]) if self.top_terms else "uncategorised"
        return f"Cluster {self.cluster_id}: {terms}"

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
    """Topic-based clustering of scientific papers.

    Parameters
    ----------
    k:
        Number of clusters.  Must be ≥ 1.
    max_iter:
        Maximum k-means iterations.
    seed:
        Random seed for reproducible results.
    min_term_freq:
        Minimum number of documents a term must appear in to be included in
        the vocabulary.  Setting this to 1 includes all terms.

    Examples
    --------
    >>> lc = LitCluster.from_bibtex("refs.bib", k=5)
    >>> lc.fit()
    >>> print(lc.summary())
    >>> lc.export_html("report.html")
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
    def from_csv(cls, path, **kwargs) -> "LitCluster":
        """Load papers from a CSV file.

        The CSV must have a header row.  Recognised columns: ``paper_id``,
        ``title``, ``abstract``, ``authors``, ``year``, ``venue``, ``doi``,
        ``keywords``.  Missing columns default to empty strings.
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
    def from_jsonl(cls, path, **kwargs) -> "LitCluster":
        """Load papers from a JSON-Lines file (one JSON object per line)."""
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
    def from_bibtex(cls, path, **kwargs) -> "LitCluster":
        """Load papers from a BibTeX (.bib) file.

        Parses ``title``, ``abstract``, ``author``, ``year``,
        ``journal``/``booktitle``, ``doi``, and ``keywords`` fields.
        Correctly handles nested braces and multi-line field values.
        """
        obj = cls(**kwargs)
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        entries = re.split(r"(?=@\w+\s*\{)", text)
        for i, entry in enumerate(entries):
            entry = entry.strip()
            if not entry or not entry.startswith("@"):
                continue
            m_key = re.match(r"@\w+\s*\{\s*(\S+?)\s*[,}]", entry)
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
    # Fit
    # ------------------------------------------------------------------

    def fit(self) -> "LitCluster":
        """Vectorise papers and assign them to clusters.

        Applies TF-IDF vectorisation followed by Lloyd's k-means on
        cosine distance to group papers by topic.

        Returns
        -------
        self : LitCluster
            Enables method chaining.
        """
        if not self.papers:
            return self
        tokens_list = [_tokenise(p.text) for p in self.papers]
        self._vectors, self._vocab = _tfidf(tokens_list, min_freq=self.min_term_freq)
        self._labels = _kmeans(self._vectors, self.k, self.max_iter, self.seed)

        clusters_map: Dict[int, List[Paper]] = {}
        for paper, label in zip(self.papers, self._labels):
            clusters_map.setdefault(label, []).append(paper)

        self.clusters = []
        for cid in sorted(clusters_map):
            top_terms = self._top_terms_for_cluster(cid, n=10)
            self.clusters.append(
                Cluster(cluster_id=cid, papers=clusters_map[cid], top_terms=top_terms)
            )
        return self

    def _top_terms_for_cluster(self, cid: int, n: int = 10) -> List[str]:
        member_vecs = [self._vectors[i] for i, lb in enumerate(self._labels) if lb == cid]
        if not member_vecs:
            return []
        scores: Dict[str, float] = {}
        for vec in member_vecs:
            for t, v in vec.items():
                scores[t] = scores.get(t, 0.0) + v
        return sorted(scores, key=lambda t: -scores[t])[:n]

    # ------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------

    def silhouette(self) -> float:
        """Mean silhouette coefficient of the current clustering.

        Requires :meth:`fit` to have been called first.
        Returns a value in [-1, 1]; values close to +1 indicate
        well-separated clusters.  Returns 0.0 when fewer than two
        distinct clusters exist.
        """
        return _silhouette_score(self._vectors, self._labels)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, path) -> None:
        """Write cluster assignments to a CSV file."""
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

    def export_json(self, path) -> None:
        """Write full cluster data (including abstracts) to a JSON file."""
        with Path(path).open("w", encoding="utf-8") as fh:
            json.dump(
                [c.to_dict() for c in self.clusters],
                fh, indent=2, ensure_ascii=False,
            )

    def export_html(self, path) -> None:
        """Write a self-contained interactive HTML report to *path*.

        Each cluster is presented as a collapsible section listing its top
        discriminative terms and a table of the constituent papers with
        clickable DOI links.
        """
        rows = []
        for c in self.clusters:
            paper_rows = "".join(_paper_row(p) for p in c.papers)
            rows.append(f"""
  <details open>
    <summary><strong>{_he(c.label)}</strong> &nbsp;({len(c.papers)} papers)</summary>
    <p><em>Top terms:</em> {_he(", ".join(c.top_terms))}</p>
    <table>
      <thead>
        <tr><th>Title</th><th>Authors</th><th>Year</th><th>Venue</th><th>DOI</th></tr>
      </thead>
      <tbody>{paper_rows}</tbody>
    </table>
  </details>""")

        title = f"litcluster — {len(self.papers)} papers, {len(self.clusters)} clusters"
        Path(path).write_text(
            _HTML_TEMPLATE.format(title=_he(title), body="\n".join(rows)),
            encoding="utf-8",
        )

    def summary(self) -> str:
        """Return a human-readable plain-text summary of the clustering."""
        if not self.clusters:
            return "No clusters found. Did you call fit()?"
        lines = [
            f"litcluster v{__version__}",
            f"{len(self.papers)} papers → {len(self.clusters)} clusters",
            "",
        ]
        for c in self.clusters:
            lines.append(f"  [{c.cluster_id:2d}] {c.label}  ({len(c.papers)} papers)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _he(text: str) -> str:
    """Minimal HTML entity escaping."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


def _paper_row(p: Paper) -> str:
    doi_cell = (
        f'<a href="https://doi.org/{_he(p.doi)}">{_he(p.doi)}</a>'
        if p.doi else ""
    )
    return (
        f"<tr>"
        f"<td>{_he(p.title)}</td>"
        f"<td>{_he(p.authors)}</td>"
        f"<td>{_he(p.year)}</td>"
        f"<td>{_he(p.venue)}</td>"
        f"<td>{doi_cell}</td>"
        f"</tr>"
    )


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, sans-serif;
      max-width: 1100px; margin: auto; padding: 1.5rem;
      color: #222;
    }}
    h1 {{ font-size: 1.35rem; margin-bottom: .25rem; }}
    p.meta {{ color: #666; font-size: .9rem; margin-top: 0; }}
    details {{
      margin-bottom: 1rem; border: 1px solid #ddd;
      padding: .6rem 1rem; border-radius: 6px;
    }}
    summary {{ cursor: pointer; font-size: 1.05rem; user-select: none; }}
    summary:hover {{ color: #0055cc; }}
    table {{
      border-collapse: collapse; width: 100%;
      margin-top: .6rem; font-size: .875rem;
    }}
    th, td {{ border: 1px solid #ccc; padding: .35rem .6rem; text-align: left; }}
    th {{ background: #f5f5f5; font-weight: 600; }}
    tr:hover td {{ background: #fafafa; }}
    a {{ color: #0066cc; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="meta">
    Generated by <a href="https://github.com/vdeshmukh203/litcluster">litcluster</a>.
  </p>
{body}
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="litcluster",
        description=(
            "Cluster academic papers by topic using TF-IDF + k-means.\n"
            "Accepts BibTeX (.bib), CSV, or JSON-Lines (.jsonl) input."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="Input file (.bib, .csv, or .jsonl)")
    p.add_argument(
        "-k", "--clusters", type=int, default=5, dest="k",
        help="Number of clusters (default: 5)",
    )
    p.add_argument(
        "--format",
        choices=["csv", "json", "html", "summary"],
        default="summary",
        help="Output format (default: summary)",
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Output file path (default: auto-named next to input file)",
    )
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--max-iter", type=int, default=100,
                   help="Maximum k-means iterations (default: 100)")
    p.add_argument(
        "--min-freq", type=int, default=2,
        help="Minimum document frequency for vocabulary inclusion (default: 2)",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """CLI entry point.  Returns 0 on success, 1 on error."""
    args = _parse_args(argv)
    path = Path(args.input)
    if not path.is_file():
        print(f"Error: '{path}' not found.", file=sys.stderr)
        return 1

    kwargs = dict(
        k=args.k,
        max_iter=args.max_iter,
        seed=args.seed,
        min_term_freq=args.min_freq,
    )
    suffix = path.suffix.lower()
    if suffix == ".bib":
        lc = LitCluster.from_bibtex(path, **kwargs)
    elif suffix == ".jsonl":
        lc = LitCluster.from_jsonl(path, **kwargs)
    else:
        lc = LitCluster.from_csv(path, **kwargs)

    if not lc.papers:
        print("Error: no papers found in input file.", file=sys.stderr)
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
    elif args.format == "html":
        out = Path(args.output) if args.output else path.with_suffix(".clusters.html")
        lc.export_html(out)
        print(f"HTML report written to {out}")

    return 0


# Alias kept for backwards compatibility with any ``pyproject.toml`` that
# still references ``litcluster:_cli``.
_cli = main


if __name__ == "__main__":
    sys.exit(main())
