#!/usr/bin/env python3
"""
litcluster — Literature Clustering Tool
Clusters academic papers by topic using TF-IDF + k-means (pure stdlib).
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
from typing import Callable, Dict, List, Optional, Tuple

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
    'you', 'your', 'use', 'used', 'using', 'show', 'shows', 'shown',
    'present', 'propose', 'paper', 'study', 'results', 'based', 'approach',
    'method', 'methods', 'new', 'two', 'three', 'one', 'first', 'second',
}


def _tokenise(text: str) -> List[str]:
    """Lowercase, strip stopwords, keep alphabetic tokens >= 3 chars."""
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _tfidf(
    documents: List[List[str]],
) -> Tuple[List[Dict[str, float]], List[str]]:
    """Compute smoothed TF-IDF vectors (sparse dicts). Returns (vectors, vocab)."""
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
    """Lloyd's k-means on sparse TF-IDF vectors (cosine similarity)."""
    n = len(vectors)
    if n == 0:
        return []
    k = min(k, n)

    import random
    rng = random.Random(seed)
    centroid_indices = rng.sample(range(n), k)
    centroids = [dict(vectors[i]) for i in centroid_indices]
    labels: List[int] = [-1] * n  # sentinel so first iteration always runs

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
# BibTeX field extraction
# ---------------------------------------------------------------------------

_BIBTEX_NON_PAPER_TYPES = frozenset({"comment", "preamble", "string"})


def _parse_bibtex_field(entry: str, field_name: str) -> str:
    """Extract a BibTeX field value, correctly handling nested braces and quotes."""
    pattern = re.compile(rf"(?i)\b{re.escape(field_name)}\s*=\s*")
    m = pattern.search(entry)
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
        return rest[1:].strip()
    if rest.startswith('"'):
        end = rest.find('"', 1)
        return rest[1:end].strip() if end != -1 else rest[1:].strip()
    m2 = re.match(r"[\w./-]+", rest)
    return m2.group() if m2 else ""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    """Metadata for a single academic paper."""

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
        """Concatenated text used for vectorisation."""
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

    def __repr__(self) -> str:
        return f"Paper(id={self.paper_id!r}, title={self.title[:50]!r})"


@dataclass
class Cluster:
    """A topic cluster containing a group of papers."""

    cluster_id: int
    papers: List[Paper] = field(default_factory=list)
    top_terms: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        terms = ", ".join(self.top_terms[:3]) if self.top_terms else "—"
        return f"Cluster {self.cluster_id}: {terms}"

    def to_dict(self) -> dict:
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
            f"size={len(self.papers)}, "
            f"terms={self.top_terms[:3]})"
        )


# ---------------------------------------------------------------------------
# LitCluster
# ---------------------------------------------------------------------------

class LitCluster:
    """Cluster academic papers by topic using TF-IDF + k-means.

    Parameters
    ----------
    k:
        Number of clusters.
    max_iter:
        Maximum k-means iterations.
    seed:
        Random seed for reproducible centroid initialisation.
    min_term_freq:
        Minimum document frequency for a term to be included in the vocabulary.
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
    # Paper management
    # ------------------------------------------------------------------

    def clear(self) -> "LitCluster":
        """Remove all loaded papers and reset cluster state."""
        self.papers.clear()
        self.clusters.clear()
        self._labels.clear()
        self._vectors.clear()
        self._vocab.clear()
        return self

    def add_paper(self, paper: Paper) -> "LitCluster":
        """Add a single :class:`Paper` and invalidate prior cluster state."""
        self.papers.append(paper)
        self.clusters.clear()
        return self

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: Path, **kwargs) -> "LitCluster":
        """Load papers from a CSV file (header row required).

        Expected columns: ``paper_id``, ``title``, ``abstract``, ``authors``,
        ``year``, ``venue``, ``doi``, ``keywords``.  Unknown columns are ignored;
        missing columns default to empty strings.
        """
        obj = cls(**kwargs)
        with path.open(encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                obj.papers.append(Paper(
                    paper_id=row.get("paper_id") or str(i),
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
        """Load papers from a JSON Lines file (one JSON object per line)."""
        obj = cls(**kwargs)
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                obj.papers.append(Paper(
                    paper_id=row.get("paper_id") or str(i),
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
        """Parse a BibTeX file and load paper entries.

        Handles ``{brace}`` and ``"quote"`` field delimiters as well as nested
        braces (common in titles with protected case).  Skips ``@comment``,
        ``@preamble``, and ``@string`` meta-entries.
        """
        obj = cls(**kwargs)
        text = path.read_text(encoding="utf-8", errors="replace")
        entries = re.split(r"(?=@\w+\s*\{)", text)
        for i, entry in enumerate(entries):
            entry = entry.strip()
            if not entry or not entry.startswith("@"):
                continue
            m_type = re.match(r"@(\w+)", entry)
            if m_type and m_type.group(1).lower() in _BIBTEX_NON_PAPER_TYPES:
                continue
            m_key = re.match(r"@\w+\s*\{\s*(\S+?)\s*[,}]", entry)
            key = m_key.group(1) if m_key else str(i)
            obj.papers.append(Paper(
                paper_id=key,
                title=_parse_bibtex_field(entry, "title"),
                abstract=_parse_bibtex_field(entry, "abstract"),
                authors=_parse_bibtex_field(entry, "author"),
                year=_parse_bibtex_field(entry, "year"),
                venue=(
                    _parse_bibtex_field(entry, "journal")
                    or _parse_bibtex_field(entry, "booktitle")
                ),
                doi=_parse_bibtex_field(entry, "doi"),
                keywords=_parse_bibtex_field(entry, "keywords"),
            ))
        return obj

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        progress: Optional[Callable[[str], None]] = None,
    ) -> "LitCluster":
        """Tokenise, vectorise, and cluster the loaded papers.

        Parameters
        ----------
        progress:
            Optional callback that receives human-readable status messages
            during fitting (useful for GUI progress indicators).

        Raises
        ------
        ValueError
            If no papers have been loaded.
        """
        if not self.papers:
            raise ValueError(
                "No papers loaded. Call from_csv(), from_jsonl(), or "
                "from_bibtex() before fit()."
            )

        def _msg(s: str) -> None:
            if progress:
                progress(s)

        _msg(f"Tokenising {len(self.papers)} papers…")
        tokens_list = [_tokenise(p.text) for p in self.papers]

        if self.min_term_freq > 1:
            _msg("Filtering rare terms…")
            freq: Dict[str, int] = {}
            for tokens in tokens_list:
                for t in set(tokens):
                    freq[t] = freq.get(t, 0) + 1
            tokens_list = [
                [t for t in toks if freq.get(t, 0) >= self.min_term_freq]
                for toks in tokens_list
            ]

        _msg("Computing TF-IDF vectors…")
        self._vectors, self._vocab = _tfidf(tokens_list)

        effective_k = min(self.k, len(self.papers))
        if effective_k < self.k:
            _msg(
                f"Warning: k capped at {effective_k} "
                f"(fewer papers than requested clusters)."
            )

        _msg(f"Running k-means (k={effective_k}, max_iter={self.max_iter})…")
        self._labels = _kmeans(
            self._vectors, effective_k, self.max_iter, self.seed
        )

        cluster_map: Dict[int, List[Paper]] = {}
        for paper, lbl in zip(self.papers, self._labels):
            cluster_map.setdefault(lbl, []).append(paper)

        self.clusters = [
            Cluster(
                cluster_id=cid,
                papers=cluster_map[cid],
                top_terms=self._top_terms_for_cluster(cid),
            )
            for cid in sorted(cluster_map)
        ]
        _msg(f"Done — {len(self.clusters)} clusters formed.")
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
    # Exporters
    # ------------------------------------------------------------------

    def export_csv(self, path: Path) -> None:
        """Write cluster assignments to a CSV file."""
        with path.open("w", newline="", encoding="utf-8") as fh:
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
        """Write clusters and paper metadata to a JSON file."""
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                [c.to_dict() for c in self.clusters],
                fh,
                indent=2,
                ensure_ascii=False,
            )

    def export_html(self, path: Path) -> None:
        """Generate a self-contained HTML report of cluster results."""
        _COLOURS = [
            "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
            "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
        ]
        cluster_html = ""
        for c in self.clusters:
            colour = _COLOURS[c.cluster_id % len(_COLOURS)]
            terms_html = " ".join(
                f'<span class="tag">{t}</span>' for t in c.top_terms
            )
            paper_rows = ""
            for p in c.papers:
                doi_link = (
                    f'<a href="https://doi.org/{p.doi}" target="_blank">'
                    f"{p.doi}</a>"
                    if p.doi
                    else "—"
                )
                paper_rows += (
                    f"<tr>"
                    f"<td>{p.paper_id}</td>"
                    f"<td>{p.title}</td>"
                    f"<td>{p.authors}</td>"
                    f"<td>{p.year}</td>"
                    f"<td>{p.venue}</td>"
                    f"<td>{doi_link}</td>"
                    f"</tr>\n"
                )
            cluster_html += f"""
<details open>
  <summary style="background:{colour}">
    <strong>Cluster {c.cluster_id}</strong>
    &mdash; {len(c.papers)} papers
    &mdash; {terms_html}
  </summary>
  <table>
    <thead><tr>
      <th>ID</th><th>Title</th><th>Authors</th>
      <th>Year</th><th>Venue</th><th>DOI</th>
    </tr></thead>
    <tbody>{paper_rows}</tbody>
  </table>
</details>
"""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>litcluster Report</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 1200px; margin: 2rem auto; padding: 0 1.5rem;
    color: #222; background: #fff;
  }}
  h1 {{ color: #333; margin-bottom: .25rem; }}
  .meta {{ color: #666; margin-bottom: 1.5rem; }}
  details {{ margin-bottom: 1rem; border: 1px solid #ddd; border-radius: 6px;
             overflow: hidden; }}
  summary {{
    list-style: none; padding: 10px 14px; cursor: pointer;
    color: #fff; user-select: none;
  }}
  summary::-webkit-details-marker {{ display: none; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{
    text-align: left; padding: 7px 12px;
    border-bottom: 1px solid #eee; font-size: .88em;
  }}
  th {{ background: #f8f8f8; font-weight: 600; }}
  tr:last-child td {{ border-bottom: none; }}
  .tag {{
    display: inline-block; background: rgba(255,255,255,.25);
    border-radius: 3px; padding: 1px 6px; font-size: .82em;
    margin-right: 3px;
  }}
  a {{ color: #4e79a7; }}
</style>
</head>
<body>
<h1>litcluster Report</h1>
<p class="meta">
  {len(self.papers)} papers &middot; {len(self.clusters)} clusters
  &middot; generated by litcluster {__version__}
</p>
{cluster_html}
</body>
</html>
"""
        path.write_text(html, encoding="utf-8")

    def summary(self) -> str:
        """Return a human-readable text summary of the clustering."""
        lines = [
            f"LitCluster: {len(self.papers)} papers "
            f"→ {len(self.clusters)} clusters",
            "",
        ]
        for c in self.clusters:
            lines.append(
                f"  [{c.cluster_id:2d}]  {c.label}  ({len(c.papers)} papers)"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="litcluster",
        description="Cluster academic papers by topic using TF-IDF + k-means.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Input file (.bib, .csv, or .jsonl)")
    p.add_argument(
        "-k", "--clusters",
        type=int, default=5, dest="k",
        help="Number of clusters",
    )
    p.add_argument(
        "--format",
        choices=["csv", "json", "html", "summary"],
        default="summary",
        help="Output format",
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Output file path (auto-named next to input if omitted)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--max-iter", type=int, default=100, help="K-means iteration cap")
    p.add_argument(
        "--min-freq", type=int, default=2,
        help="Minimum document frequency for vocabulary inclusion",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    path = Path(args.input)
    if not path.is_file():
        print(f"Error: '{path}' not found.", file=sys.stderr)
        return 1

    suffix = path.suffix.lower()
    kwargs = dict(
        k=args.k,
        max_iter=args.max_iter,
        seed=args.seed,
        min_term_freq=args.min_freq,
    )

    if suffix == ".bib":
        lc = LitCluster.from_bibtex(path, **kwargs)
    elif suffix == ".jsonl":
        lc = LitCluster.from_jsonl(path, **kwargs)
    else:
        lc = LitCluster.from_csv(path, **kwargs)

    lc.fit(progress=lambda s: print(s, file=sys.stderr))

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
        print(f"Report written to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
