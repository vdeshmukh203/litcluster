#!/usr/bin/env python3
"""
litcluster — Literature Clustering Tool

Clusters academic papers by topic using TF-IDF + k-means.
Pure standard-library implementation; zero external dependencies.

Typical usage::

    lc = LitCluster.from_bibtex(Path("refs.bib"), k=6)
    lc.fit()
    lc.export_html(Path("clusters.html"))
    print(lc.summary())
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
__author__ = "Vaibhav Deshmukh"
__license__ = "MIT"


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

_STOPWORDS = {
    # Articles / conjunctions / prepositions
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
    # Common academic filler words (suppress them so clusters contain
    # domain-meaningful terms only)
    'paper', 'study', 'work', 'approach', 'method', 'methods', 'technique',
    'techniques', 'propose', 'proposed', 'present', 'show', 'shows',
    'shown', 'demonstrate', 'demonstrates', 'find', 'found', 'results',
    'result', 'data', 'new', 'use', 'used', 'using', 'based', 'two',
    'three', 'four', 'five', 'one', 'first', 'second', 'third', 'also',
    'thus', 'however', 'therefore', 'while', 'whereas', 'since', 'due',
    'via', 'within', 'without', 'across', 'per', 'well', 'large', 'high',
    'low', 'good', 'different', 'various', 'several', 'many',
}


def _tokenise(text: str) -> List[str]:
    """Lowercase and tokenise *text*, removing stop-words and short tokens.

    Only alphabetic tokens of three or more characters are kept.
    """
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _tfidf(
    documents: List[List[str]],
) -> Tuple[List[Dict[str, float]], List[str]]:
    """Compute TF-IDF vectors for a list of tokenised documents.

    Uses the *sklearn*-style smoothed IDF formula
    ``idf = log((N+1)/(df+1)) + 1`` so that terms appearing in every
    document still receive a non-zero weight.

    Parameters
    ----------
    documents:
        List of token lists, one per document.

    Returns
    -------
    vectors:
        Sparse ``{term: weight}`` dicts, one per document.
    vocab:
        Sorted list of all unique terms across the corpus.
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
    """Cosine similarity between two sparse TF-IDF vectors.

    Returns 0.0 when either vector is empty or has zero norm.
    """
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
    """Lloyd's k-means clustering on sparse cosine-similarity vectors.

    Parameters
    ----------
    vectors:
        Sparse TF-IDF vectors as ``{term: weight}`` dicts.
    k:
        Target number of clusters (clamped to ``len(vectors)``).
    max_iter:
        Maximum Lloyd iterations before forced convergence.
    seed:
        Random seed for reproducible centroid initialisation.

    Returns
    -------
    List[int]
        Integer cluster label (0-indexed) for every input vector.
    """
    import random

    n = len(vectors)
    if n == 0:
        return []
    k = min(k, n)

    rng = random.Random(seed)
    centroid_indices = rng.sample(range(n), k)
    centroids = [dict(vectors[i]) for i in centroid_indices]
    labels: List[int] = [-1] * n

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
                # Reinitialise empty cluster from a random document
                centroids[c] = dict(vectors[rng.randint(0, n - 1)])
                continue
            new_centroid: Dict[str, float] = {}
            for vec in members:
                for t, v in vec.items():
                    new_centroid[t] = new_centroid.get(t, 0.0) + v
            m = len(members)
            centroids[c] = {t: v / m for t, v in new_centroid.items()}

    return labels


def _bibtex_field(entry: str, field_name: str) -> str:
    """Extract the value of *field_name* from a single BibTeX entry string.

    Correctly handles both ``{...}`` and ``"..."`` delimiters and tracks
    nested curly braces so titles like ``{Advances in {Machine Learning}}``
    are extracted in full.
    """
    pattern = re.compile(
        r'(?i)\b' + re.escape(field_name) + r'\s*=\s*'
    )
    m = pattern.search(entry)
    if not m:
        return ""
    rest = entry[m.end():].lstrip()
    if rest.startswith('{'):
        depth = 0
        for i, ch in enumerate(rest):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return rest[1:i].strip()
        return ""
    if rest.startswith('"'):
        end = rest.find('"', 1)
        return rest[1:end].strip() if end >= 0 else ""
    # Numeric / unquoted value
    m2 = re.match(r'[^\s,}\n]+', rest)
    return m2.group(0) if m2 else ""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    """A single academic paper with standard bibliographic metadata.

    Attributes
    ----------
    paper_id:
        Unique identifier (BibTeX cite-key, row index, or DOI).
    title:
        Paper title.
    abstract:
        Full abstract text.
    authors:
        Author string (free-form, e.g. ``"Smith, J. and Doe, A."``).
    year:
        Publication year as a string.
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
        """Concatenated text used for vectorisation (title + abstract + keywords)."""
        return " ".join(filter(None, [self.title, self.abstract, self.keywords]))

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
    """A group of thematically related papers.

    Attributes
    ----------
    cluster_id:
        Zero-based integer identifier assigned by k-means.
    papers:
        All :class:`Paper` objects assigned to this cluster.
    top_terms:
        Top-N TF-IDF terms that characterise the cluster, ranked by
        aggregate weight.
    """

    cluster_id: int
    papers: List[Paper] = field(default_factory=list)
    top_terms: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Short human-readable label built from the top three terms."""
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


# ---------------------------------------------------------------------------
# LitCluster
# ---------------------------------------------------------------------------

class LitCluster:
    """Orchestrates loading, clustering, and exporting paper collections.

    Parameters
    ----------
    k:
        Number of clusters (default 5).
    max_iter:
        Maximum k-means iterations (default 100).
    seed:
        Random seed for reproducible results (default 42).
    min_term_freq:
        Minimum document-frequency (number of papers a term must appear
        in) to be included in the vocabulary.  Setting this higher speeds
        up computation on large corpora and removes rare noise terms
        (default 2).

    Examples
    --------
    >>> from pathlib import Path
    >>> lc = LitCluster.from_bibtex(Path("refs.bib"), k=5)
    >>> lc.fit()
    >>> print(lc.summary())
    >>> lc.export_html(Path("report.html"))
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
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: Path, **kwargs) -> "LitCluster":
        """Load papers from a CSV file.

        Expected columns (all optional except ``title``):
        ``paper_id``, ``title``, ``abstract``, ``authors``, ``year``,
        ``venue``, ``doi``, ``keywords``.

        Extra columns are silently ignored.
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

        The same field names as :meth:`from_csv` are recognised.
        Blank lines are skipped.
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
        """Load papers from a BibTeX (``.bib``) file.

        Extracts ``title``, ``abstract``, ``author``, ``year``,
        ``journal``/``booktitle``, ``doi``, and ``keywords`` fields.
        Nested curly braces in field values are handled correctly.
        Entries without a ``@`` marker are skipped.
        """
        obj = cls(**kwargs)
        text = path.read_text(encoding="utf-8", errors="replace")
        entries = re.split(r'(?=@\w+\s*\{)', text)
        for i, entry in enumerate(entries):
            entry = entry.strip()
            if not entry or not entry.startswith('@'):
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

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def fit(self) -> "LitCluster":
        """Run the TF-IDF + k-means clustering pipeline.

        Tokenises each paper's text (title + abstract + keywords),
        builds a TF-IDF matrix, applies k-means, and populates
        :attr:`clusters`.

        Returns *self* for method chaining.
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
        """Return the top-*n* terms for cluster *cid* by aggregate TF-IDF weight."""
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
        """Write cluster assignments to *path* as a flat CSV file.

        Columns: ``cluster_id``, ``cluster_label``, ``paper_id``,
        ``title``, ``authors``, ``year``, ``venue``, ``doi``.
        """
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([
                "cluster_id", "cluster_label", "paper_id",
                "title", "authors", "year", "venue", "doi",
            ])
            for cluster in self.clusters:
                for p in cluster.papers:
                    w.writerow([
                        cluster.cluster_id, cluster.label,
                        p.paper_id, p.title, p.authors,
                        p.year, p.venue, p.doi,
                    ])

    def export_json(self, path: Path) -> None:
        """Write full cluster data (including abstracts) to *path* as JSON."""
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                [c.to_dict() for c in self.clusters],
                fh,
                indent=2,
                ensure_ascii=False,
            )

    def export_html(self, path: Path) -> None:
        """Write a self-contained interactive HTML report to *path*.

        The report lists every cluster as a collapsible card showing its
        top terms and a table of member papers.  Clicking a paper row
        reveals its abstract.  A search box filters papers in real-time.
        """
        path.write_text(_render_html(self), encoding="utf-8")

    def summary(self) -> str:
        """Return a human-readable plain-text overview of the clusters.

        Includes the top three terms per cluster and a sample of up to
        three paper titles.  If :meth:`fit` has not been called, returns
        a reminder.
        """
        if not self.clusters:
            return (
                f"LitCluster: {len(self.papers)} papers loaded — "
                "call fit() to run clustering."
            )
        lines = [
            f"LitCluster: {len(self.papers)} papers in {len(self.clusters)} clusters",
            "",
        ]
        for c in self.clusters:
            lines.append(f"  [{c.cluster_id}] {c.label}  ({len(c.papers)} papers)")
            for p in c.papers[:3]:
                snippet = (p.title or "(no title)")[:72]
                lines.append(f"      • {snippet}")
            if len(c.papers) > 3:
                lines.append(f"        … and {len(c.papers) - 3} more")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

_CLUSTER_COLOURS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


def _render_html(lc: LitCluster) -> str:
    """Build and return a self-contained interactive HTML string for *lc*."""
    clusters_json = json.dumps(
        [c.to_dict() for c in lc.clusters], ensure_ascii=False
    )
    n_papers = len(lc.papers)
    n_clusters = len(lc.clusters)

    cards_html_parts: List[str] = []
    for idx, cluster in enumerate(lc.clusters):
        colour = _CLUSTER_COLOURS[idx % len(_CLUSTER_COLOURS)]
        terms_html = " ".join(
            f'<span class="term">{_html_escape(t)}</span>'
            for t in cluster.top_terms[:8]
        )
        paper_rows: List[str] = []
        for p in cluster.papers:
            title_esc = _html_escape(p.title or "(no title)")
            authors_esc = _html_escape(p.authors or "")
            year = _html_escape(p.year or "")
            doi_link = (
                f'<a href="https://doi.org/{_html_escape(p.doi)}" '
                f'target="_blank">{_html_escape(p.doi)}</a>'
                if p.doi else ""
            )
            abstract_text = p.abstract or ""
            abstract_short = _html_escape(abstract_text[:300])
            if len(abstract_text) > 300:
                abstract_short += "…"
            paper_rows.append(
                f'<tr class="paper-row">'
                f'<td class="paper-title">{title_esc}</td>'
                f'<td>{authors_esc}</td>'
                f'<td class="paper-year">{year}</td>'
                f'<td>{doi_link}</td>'
                f'</tr>'
                f'<tr class="abstract-row" style="display:none">'
                f'<td colspan="4" class="abstract-cell">{abstract_short}</td>'
                f'</tr>'
            )

        papers_html = "\n".join(paper_rows)
        cards_html_parts.append(f"""
  <div class="cluster-card" data-cluster="{cluster.cluster_id}" style="border-left:5px solid {colour}">
    <div class="cluster-header" onclick="toggleCluster(this)">
      <span class="cluster-badge" style="background:{colour}">{cluster.cluster_id}</span>
      <span class="cluster-title">{_html_escape(cluster.label)}</span>
      <span class="cluster-count">{len(cluster.papers)} paper{"s" if len(cluster.papers) != 1 else ""}</span>
      <span class="toggle-icon">&#9660;</span>
    </div>
    <div class="cluster-terms">{terms_html}</div>
    <div class="cluster-body" style="display:none">
      <table class="paper-table">
        <thead><tr><th>Title</th><th>Authors</th><th>Year</th><th>DOI</th></tr></thead>
        <tbody>{papers_html}</tbody>
      </table>
    </div>
  </div>""")

    cards_html = "\n".join(cards_html_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>litcluster &mdash; {n_papers} papers, {n_clusters} clusters</title>
<style>
*,*::before,*::after{{box-sizing:border-box}}
body{{font-family:system-ui,sans-serif;margin:0;background:#f5f5f5;color:#333}}
header{{background:#2c3e50;color:#fff;padding:.9rem 2rem;display:flex;align-items:center;gap:1rem}}
header h1{{margin:0;font-size:1.4rem}}
.meta{{font-size:.85rem;opacity:.75}}
.controls{{padding:.75rem 2rem;background:#fff;border-bottom:1px solid #ddd;display:flex;gap:.6rem;flex-wrap:wrap;align-items:center}}
.controls input{{flex:1;min-width:200px;padding:.4rem .7rem;border:1px solid #ccc;border-radius:4px;font-size:.9rem}}
.controls button{{padding:.4rem .9rem;border:none;border-radius:4px;cursor:pointer;font-size:.85rem}}
.btn-expand{{background:#4e79a7;color:#fff}}
.btn-collapse{{background:#e0e0e0;color:#333}}
main{{max-width:1100px;margin:1.2rem auto;padding:0 1rem}}
.cluster-card{{background:#fff;border-radius:6px;margin-bottom:.9rem;box-shadow:0 1px 3px rgba(0,0,0,.1);overflow:hidden}}
.cluster-header{{display:flex;align-items:center;gap:.7rem;padding:.7rem 1rem;cursor:pointer;user-select:none}}
.cluster-header:hover{{background:#f9f9f9}}
.cluster-badge{{width:26px;height:26px;border-radius:50%;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:.8rem;flex-shrink:0}}
.cluster-title{{flex:1;font-weight:600}}
.cluster-count{{font-size:.8rem;color:#777}}
.toggle-icon{{font-size:.75rem;color:#aaa;transition:transform .2s}}
.cluster-terms{{padding:.2rem 1rem .5rem 3.5rem;display:flex;flex-wrap:wrap;gap:.3rem}}
.term{{background:#eef2ff;color:#3b4cb8;padding:.12rem .45rem;border-radius:999px;font-size:.78rem}}
.cluster-body{{padding:0 1rem 1rem}}
.paper-table{{width:100%;border-collapse:collapse;font-size:.85rem}}
.paper-table th{{text-align:left;padding:.35rem .5rem;border-bottom:2px solid #eee;color:#555;font-weight:600}}
.paper-row td{{padding:.35rem .5rem;border-bottom:1px solid #f0f0f0;vertical-align:top;cursor:pointer}}
.paper-row:hover td{{background:#fafafa}}
.paper-title{{font-weight:500}}
.paper-year{{width:50px;text-align:center}}
.abstract-cell{{padding:.5rem 1rem .7rem;font-size:.82rem;color:#555;background:#fafbff}}
.no-results{{text-align:center;color:#999;padding:2rem}}
footer{{text-align:center;padding:1.5rem;font-size:.8rem;color:#aaa}}
</style>
</head>
<body>
<header>
  <h1>&#128218; litcluster</h1>
  <div class="meta">{n_papers} papers &nbsp;&middot;&nbsp; {n_clusters} clusters</div>
</header>
<div class="controls">
  <input type="text" id="search" placeholder="Filter by title, author or year&hellip;"
         oninput="filterPapers(this.value)">
  <button class="btn-expand" onclick="expandAll()">Expand all</button>
  <button class="btn-collapse" onclick="collapseAll()">Collapse all</button>
</div>
<main id="main">
{cards_html}
  <p class="no-results" id="no-results" style="display:none">No papers match your filter.</p>
</main>
<footer>Generated by <a href="https://github.com/vdeshmukh203/litcluster">litcluster</a> {__version__}</footer>
<script>
function toggleCluster(hdr){{
  var card=hdr.closest('.cluster-card');
  var body=card.querySelector('.cluster-body');
  var icon=hdr.querySelector('.toggle-icon');
  var open=body.style.display!=='none';
  body.style.display=open?'none':'block';
  icon.style.transform=open?'':'rotate(180deg)';
}}
function expandAll(){{
  document.querySelectorAll('.cluster-body').forEach(function(b){{b.style.display='block';}});
  document.querySelectorAll('.toggle-icon').forEach(function(i){{i.style.transform='rotate(180deg)';}});
}}
function collapseAll(){{
  document.querySelectorAll('.cluster-body').forEach(function(b){{b.style.display='none';}});
  document.querySelectorAll('.toggle-icon').forEach(function(i){{i.style.transform='';}});
}}
document.querySelectorAll('.paper-row').forEach(function(row){{
  row.addEventListener('click',function(){{
    var next=row.nextElementSibling;
    if(next&&next.classList.contains('abstract-row')){{
      next.style.display=next.style.display==='none'?'table-row':'none';
    }}
  }});
}});
function filterPapers(q){{
  q=q.toLowerCase();
  var anyVisible=false;
  document.querySelectorAll('.cluster-card').forEach(function(card){{
    var rows=card.querySelectorAll('.paper-row');
    var cardHit=false;
    rows.forEach(function(row){{
      var match=!q||row.textContent.toLowerCase().includes(q);
      row.style.display=match?'':'none';
      var abs=row.nextElementSibling;
      if(abs&&abs.classList.contains('abstract-row')&&!match)abs.style.display='none';
      if(match)cardHit=true;
    }});
    card.style.display=cardHit?'':'none';
    if(cardHit)anyVisible=true;
    if(cardHit&&q){{
      card.querySelector('.cluster-body').style.display='block';
      card.querySelector('.toggle-icon').style.transform='rotate(180deg)';
    }}
  }});
  document.getElementById('no-results').style.display=anyVisible?'none':'block';
}}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="litcluster",
        description="Cluster academic papers by topic using TF-IDF + k-means.",
    )
    p.add_argument("input", help="Input file (.csv, .jsonl, or .bib)")
    p.add_argument(
        "-k", "--clusters", type=int, default=5, dest="k",
        metavar="K",
        help="Number of clusters (default: 5)",
    )
    p.add_argument(
        "--format", choices=["csv", "json", "html", "summary"],
        default="summary",
        help="Output format (default: summary)",
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Output file path (default: derived from input filename)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--max-iter", type=int, default=100,
                   help="Max k-means iterations (default: 100)")
    p.add_argument(
        "--min-freq", type=int, default=2,
        help="Minimum document-frequency for vocabulary terms (default: 2)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Command-line entry point for the ``litcluster`` command."""
    args = _parse_args(argv)
    path = Path(args.input)
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
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

    lc.fit()

    fmt = args.format
    if fmt == "summary":
        output = lc.summary()
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
        else:
            print(output)
    elif fmt == "csv":
        out = Path(args.output) if args.output else path.with_suffix(".clusters.csv")
        lc.export_csv(out)
        print(f"Clusters written to {out}")
    elif fmt == "json":
        out = Path(args.output) if args.output else path.with_suffix(".clusters.json")
        lc.export_json(out)
        print(f"Clusters written to {out}")
    elif fmt == "html":
        out = Path(args.output) if args.output else path.with_suffix(".clusters.html")
        lc.export_html(out)
        print(f"Interactive HTML written to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
