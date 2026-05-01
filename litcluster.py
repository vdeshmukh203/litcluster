#!/usr/bin/env python3
"""
litcluster — Literature Clustering Tool

Clusters academic papers by topic using TF-IDF vectorisation and k-means
(Lloyd's algorithm with k-means++ initialisation). Supports BibTeX, CSV,
and JSONL input formats, and exports results as CSV, JSON, or a
self-contained interactive HTML report.

All functionality uses the Python standard library only; no external
packages are required.
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
from typing import Dict, List, Optional, Tuple


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
    'present', 'propose', 'proposed', 'paper', 'study', 'work', 'based',
    'approach', 'method', 'results', 'data', 'new', 'two', 'first',
}


def _tokenise(text: str) -> List[str]:
    """Extract lowercase alphabetic tokens of 3+ characters, excluding stopwords."""
    if not text:
        return []
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _tfidf(
    documents: List[List[str]],
    min_freq: int = 1,
) -> Tuple[List[Dict[str, float]], List[str]]:
    """Compute TF-IDF sparse vectors for a corpus.

    Parameters
    ----------
    documents:
        Each element is a list of tokens for one document.
    min_freq:
        Terms appearing in fewer than this many documents are excluded.

    Returns
    -------
    vectors:
        One sparse TF-IDF dict per document.
    vocab:
        Sorted list of terms in the vocabulary.
    """
    n = len(documents)
    if n == 0:
        return [], []

    # Document frequency
    df: Dict[str, int] = {}
    for doc in documents:
        for t in set(doc):
            df[t] = df.get(t, 0) + 1

    vocab = sorted(t for t, count in df.items() if count >= min_freq)
    if not vocab:
        return [{} for _ in documents], []

    # TF-IDF vectors
    vectors: List[Dict[str, float]] = []
    for doc in documents:
        doc_len = len(doc) or 1
        tf: Dict[str, int] = {}
        for t in doc:
            if t in df and df[t] >= min_freq:
                tf[t] = tf.get(t, 0) + 1
        vec: Dict[str, float] = {}
        for t, count in tf.items():
            idf = math.log((n + 1) / (df[t] + 1)) + 1.0
            vec[t] = (count / doc_len) * idf
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


def _kmeans_plusplus_init(
    vectors: List[Dict[str, float]],
    k: int,
    rng: random.Random,
) -> List[Dict[str, float]]:
    """k-means++ centroid initialisation for better cluster quality."""
    n = len(vectors)
    centroids: List[Dict[str, float]] = [dict(vectors[rng.randint(0, n - 1)])]

    for _ in range(k - 1):
        # Distance from each point to its nearest current centroid
        dists = [
            1.0 - max(_cosine(vec, c) for c in centroids)
            for vec in vectors
        ]
        total = sum(d * d for d in dists)
        if total == 0.0:
            centroids.append(dict(vectors[rng.randint(0, n - 1)]))
            continue
        # Weighted random selection proportional to squared distance
        threshold = rng.random() * total
        cumul = 0.0
        chosen = n - 1
        for j, d in enumerate(dists):
            cumul += d * d
            if cumul >= threshold:
                chosen = j
                break
        centroids.append(dict(vectors[chosen]))

    return centroids


def _kmeans(
    vectors: List[Dict[str, float]],
    k: int,
    max_iter: int = 100,
    seed: int = 42,
) -> List[int]:
    """Lloyd's k-means on sparse cosine-similarity space with k-means++ init.

    Parameters
    ----------
    vectors:
        Sparse TF-IDF document vectors.
    k:
        Target number of clusters.
    max_iter:
        Maximum number of Lloyd iterations.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    labels:
        Integer cluster assignment for each document.
    """
    n = len(vectors)
    if n == 0:
        return []
    k = min(k, n)
    rng = random.Random(seed)

    centroids = _kmeans_plusplus_init(vectors, k, rng)
    labels = [0] * n

    for _ in range(max_iter):
        new_labels = [
            max(range(k), key=lambda c: _cosine(vec, centroids[c]))
            for vec in vectors
        ]
        if new_labels == labels:
            break
        labels = new_labels

        # Recompute centroids as mean of assigned vectors
        for c in range(k):
            members = [vectors[i] for i, lbl in enumerate(labels) if lbl == c]
            if not members:
                # Reinitialise empty cluster to a random point
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
    """Metadata record for a single academic paper."""

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
        """Combined text used for TF-IDF vectorisation."""
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

    @classmethod
    def from_dict(cls, row: dict, index: int = 0) -> "Paper":
        return cls(
            paper_id=str(row.get("paper_id", index)),
            title=str(row.get("title", "")),
            abstract=str(row.get("abstract", "")),
            authors=str(row.get("authors", "")),
            year=str(row.get("year", "")),
            venue=str(row.get("venue", "")),
            doi=str(row.get("doi", "")),
            keywords=str(row.get("keywords", "")),
        )


@dataclass
class Cluster:
    """A group of thematically related papers."""

    cluster_id: int
    papers: List[Paper] = field(default_factory=list)
    top_terms: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        terms = ", ".join(self.top_terms[:3]) if self.top_terms else "unlabelled"
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
# HTML export
# ---------------------------------------------------------------------------

# NOTE: braces in the CSS/JS blocks are NOT Python format specifiers;
# the placeholder __LITCLUSTER_DATA__ is substituted via str.replace().
_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__LITCLUSTER_TITLE__</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:#f0f4f8;color:#1a202c;line-height:1.5}
header{background:#1e40af;color:#fff;padding:1.5rem 2rem}
.logo{font-size:1.25rem;font-weight:700;letter-spacing:.02em;opacity:.9}
h1{font-size:1.5rem;font-weight:700;margin:.25rem 0}
.hstats{display:flex;gap:1rem;margin-top:.5rem;font-size:.82rem;flex-wrap:wrap}
.hstat{background:rgba(255,255,255,.15);padding:.2rem .65rem;border-radius:9999px}
.controls{background:#fff;border-bottom:1px solid #e2e8f0;padding:.65rem 2rem;
  display:flex;gap:.65rem;align-items:center;position:sticky;top:0;z-index:50;
  box-shadow:0 1px 3px rgba(0,0,0,.08)}
#search{flex:1;padding:.4rem .75rem;border:1px solid #cbd5e0;border-radius:.375rem;
  font-size:.9rem;outline:none}
#search:focus{border-color:#3b82f6;box-shadow:0 0 0 2px rgba(59,130,246,.2)}
#cfilter{padding:.4rem .75rem;border:1px solid #cbd5e0;border-radius:.375rem;
  font-size:.9rem;background:#fff;cursor:pointer}
.overview{padding:1.25rem 2rem .5rem}
.ov-title{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.07em;
  color:#718096;margin-bottom:.6rem}
.pills{display:flex;gap:.4rem;flex-wrap:wrap}
.pill{padding:.3rem .8rem;border-radius:9999px;font-size:.8rem;cursor:pointer;
  transition:opacity .15s;display:flex;align-items:center;gap:.35rem;color:#fff;
  border:none;font-weight:500;font-family:inherit}
.pill:hover{opacity:.85}
.pill.dim{opacity:.35}
.main{padding:1rem 2rem 3rem}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:.5rem;margin-bottom:1rem;
  overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.card-hdr{display:flex;align-items:center;gap:.65rem;padding:.85rem 1.25rem;
  cursor:pointer;user-select:none;transition:background .12s}
.card-hdr:hover{background:#f7fafc}
.cbadge{width:2rem;height:2rem;border-radius:50%;display:flex;align-items:center;
  justify-content:center;color:#fff;font-weight:700;font-size:.78rem;flex-shrink:0}
.ctitle{font-weight:600;flex:1;font-size:.93rem}
.csize{font-size:.8rem;color:#718096}
.chev{color:#a0aec0;font-size:.72rem;transition:transform .2s;margin-left:.25rem}
.card.open .chev{transform:rotate(90deg)}
.card-body{display:none;border-top:1px solid #e2e8f0}
.card.open .card-body{display:block}
.terms{padding:.55rem 1.25rem;display:flex;flex-wrap:wrap;gap:.3rem;
  background:#f7fafc;border-bottom:1px solid #e2e8f0}
.term{background:#fff;border:1px solid #e2e8f0;padding:.1rem .45rem;border-radius:.25rem;
  font-size:.76rem;color:#4a5568}
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.84rem}
thead th{background:#f7fafc;padding:.45rem .75rem;text-align:left;font-size:.72rem;
  font-weight:600;color:#718096;text-transform:uppercase;letter-spacing:.05em;
  border-bottom:1px solid #e2e8f0;white-space:nowrap}
tbody td{padding:.45rem .75rem;border-bottom:1px solid #f0f4f8;vertical-align:top}
tbody tr:last-child td{border-bottom:none}
tbody tr.hidden{display:none}
tbody tr:hover td{background:#fafbfc}
.tc-title{max-width:320px;font-weight:500}
.tc-author{max-width:200px;color:#4a5568;font-size:.8rem}
.tc-year{white-space:nowrap;color:#718096}
.tc-venue{max-width:180px;color:#4a5568;font-size:.8rem}
.doi-link{color:#3b82f6;text-decoration:none;font-size:.76rem}
.doi-link:hover{text-decoration:underline}
.no-match{padding:2rem;text-align:center;color:#a0aec0;font-size:.88rem}
footer{text-align:center;padding:2rem;font-size:.76rem;color:#a0aec0;
  border-top:1px solid #e2e8f0}
footer a{color:#718096;text-decoration:none}
@media(max-width:600px){
  header,.controls,.overview,.main{padding-left:1rem;padding-right:1rem}
  .hstats{flex-wrap:wrap}
}
</style>
</head>
<body>
<header>
  <div class="logo">litcluster</div>
  <h1>__LITCLUSTER_TITLE__</h1>
  <div class="hstats" id="hstats"></div>
</header>
<div class="controls">
  <input type="search" id="search" placeholder="Search titles and authors…">
  <select id="cfilter" onchange="filterBySelect(this.value)">
    <option value="">All clusters</option>
  </select>
</div>
<div class="overview">
  <div class="ov-title">Cluster overview</div>
  <div class="pills" id="pills"></div>
</div>
<div class="main" id="main"></div>
<footer>Generated by <a href="https://github.com/vdeshmukh203/litcluster">litcluster</a> v0.1.0</footer>
<script>
(function(){
var COLORS=['#2563eb','#16a34a','#dc2626','#d97706','#7c3aed',
            '#0891b2','#be185d','#059669','#ea580c','#4338ca'];
function clr(i){return COLORS[i%COLORS.length];}
function esc(s){
  if(!s)return'';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
var DATA=__LITCLUSTER_DATA__;
var activeCluster=null;
function init(){
  // Header stats
  var total=DATA.clusters.reduce(function(a,c){return a+c.size;},0);
  document.getElementById('hstats').innerHTML=
    '<span class="hstat">'+total+' papers</span>'+
    '<span class="hstat">'+DATA.clusters.length+' clusters</span>';
  // Cluster filter dropdown
  var sel=document.getElementById('cfilter');
  DATA.clusters.forEach(function(c,i){
    var o=document.createElement('option');
    o.value=c.cluster_id;
    o.textContent='Cluster '+c.cluster_id+' ('+c.size+')';
    sel.appendChild(o);
  });
  // Pills
  var pillsEl=document.getElementById('pills');
  DATA.clusters.forEach(function(c,i){
    var b=document.createElement('button');
    b.className='pill';
    b.id='pill-'+c.cluster_id;
    b.style.background=clr(i);
    b.innerHTML='C'+c.cluster_id+' <strong>'+c.size+'</strong>';
    b.onclick=function(){toggleCluster(c.cluster_id);};
    pillsEl.appendChild(b);
  });
  // Cards
  var mainEl=document.getElementById('main');
  DATA.clusters.forEach(function(c,i){
    var rows=c.papers.map(function(p){
      var doi=p.doi
        ?'<a class="doi-link" href="https://doi.org/'+esc(p.doi)+'" target="_blank" rel="noopener">'+esc(p.doi)+'</a>'
        :'—';
      return '<tr data-search="'+esc((p.title||'')+' '+(p.authors||''))+'">'+
        '<td class="tc-title">'+esc(p.title||'—')+'</td>'+
        '<td class="tc-author">'+esc(p.authors||'—')+'</td>'+
        '<td class="tc-year">'+esc(p.year||'—')+'</td>'+
        '<td class="tc-venue">'+esc(p.venue||'—')+'</td>'+
        '<td>'+doi+'</td></tr>';
    }).join('');
    var terms=c.top_terms.map(function(t){
      return '<span class="term">'+esc(t)+'</span>';
    }).join('');
    var div=document.createElement('div');
    div.className='card';
    div.id='card-'+c.cluster_id;
    div.innerHTML=
      '<div class="card-hdr" onclick="toggleCard('+c.cluster_id+')">'+
        '<div class="cbadge" style="background:'+clr(i)+'">'+c.cluster_id+'</div>'+
        '<div class="ctitle">'+esc(c.label)+'</div>'+
        '<div class="csize" id="csz-'+c.cluster_id+'">'+c.size+' paper'+(c.size!==1?'s':'')+'</div>'+
        '<div class="chev">&#9654;</div>'+
      '</div>'+
      '<div class="card-body">'+
        '<div class="terms">'+terms+'</div>'+
        '<div class="tbl-wrap"><table>'+
          '<thead><tr><th>Title</th><th>Authors</th><th>Year</th><th>Venue</th><th>DOI</th></tr></thead>'+
          '<tbody id="tbody-'+c.cluster_id+'">'+rows+'</tbody>'+
        '</table></div>'+
      '</div>';
    mainEl.appendChild(div);
  });
  document.getElementById('search').addEventListener('input',applySearch);
}
function toggleCard(cid){
  document.getElementById('card-'+cid).classList.toggle('open');
}
function toggleCluster(cid){
  activeCluster=(activeCluster===cid)?null:cid;
  applyClusterFilter();
}
function filterBySelect(val){
  activeCluster=(val==='')?null:parseInt(val,10);
  applyClusterFilter();
}
function applyClusterFilter(){
  document.querySelectorAll('.card').forEach(function(card){
    var cid=parseInt(card.id.replace('card-',''),10);
    card.style.display=(activeCluster===null||cid===activeCluster)?'':'none';
  });
  document.querySelectorAll('.pill').forEach(function(pill){
    var cid=parseInt(pill.id.replace('pill-',''),10);
    pill.classList.toggle('dim',activeCluster!==null&&cid!==activeCluster);
  });
  applySearch();
}
function applySearch(){
  var q=document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('tbody tr').forEach(function(row){
    var match=!q||(row.dataset.search||'').toLowerCase().indexOf(q)!==-1;
    row.classList.toggle('hidden',!match);
  });
  DATA.clusters.forEach(function(c){
    var card=document.getElementById('card-'+c.cluster_id);
    if(!card||card.style.display==='none')return;
    var tbody=document.getElementById('tbody-'+c.cluster_id);
    if(!tbody)return;
    var visible=tbody.querySelectorAll('tr:not(.hidden)').length;
    var sz=document.getElementById('csz-'+c.cluster_id);
    if(sz)sz.textContent=visible+' paper'+(visible!==1?'s':'');
  });
}
window.addEventListener('DOMContentLoaded',init);
})();
</script>
</body>
</html>
"""


def _build_html(clusters: List[Cluster], title: str = "Literature Cluster Report") -> str:
    """Render a self-contained interactive HTML report for *clusters*."""
    data = {
        "clusters": [c.to_dict() for c in clusters],
    }
    html = _HTML_TEMPLATE
    html = html.replace("__LITCLUSTER_TITLE__", title)
    html = html.replace("__LITCLUSTER_DATA__", json.dumps(data, ensure_ascii=False))
    return html


# ---------------------------------------------------------------------------
# LitCluster
# ---------------------------------------------------------------------------

class LitCluster:
    """Cluster a collection of academic papers by topic using TF-IDF + k-means.

    Parameters
    ----------
    k:
        Number of clusters.
    max_iter:
        Maximum k-means iterations.
    seed:
        Random seed for reproducibility.
    min_term_freq:
        Discard terms that appear in fewer than this many documents.
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
    # Input parsers
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: Path, **kwargs) -> "LitCluster":
        """Load papers from a CSV file.

        Expected columns (all optional except at least *title* or *abstract*):
        ``paper_id``, ``title``, ``abstract``, ``authors``, ``year``,
        ``venue``, ``doi``, ``keywords``.
        """
        obj = cls(**kwargs)
        with Path(path).open(encoding="utf-8", errors="replace", newline="") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                obj.papers.append(Paper.from_dict(row, index=i))
        return obj

    @classmethod
    def from_jsonl(cls, path: Path, **kwargs) -> "LitCluster":
        """Load papers from a JSONL file (one JSON object per line)."""
        obj = cls(**kwargs)
        with Path(path).open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                obj.papers.append(Paper.from_dict(json.loads(line), index=i))
        return obj

    @classmethod
    def from_bibtex(cls, path: Path, **kwargs) -> "LitCluster":
        """Load papers from a BibTeX (.bib) file.

        Extracts ``title``, ``abstract``, ``author``, ``year``,
        ``journal``/``booktitle``, ``doi``, and ``keywords`` fields.
        Handles multi-line values and nested braces correctly.
        """
        obj = cls(**kwargs)
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        entries = re.split(r'(?=@\w+\s*\{)', text)
        for i, entry in enumerate(entries):
            if not entry.strip() or not entry.startswith('@'):
                continue
            m_key = re.match(r'@\w+\s*\{\s*([^\s,}]+)', entry)
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

    @classmethod
    def from_list(cls, papers: List[dict], **kwargs) -> "LitCluster":
        """Load papers from a list of dicts (programmatic API).

        Each dict may contain any subset of: ``paper_id``, ``title``,
        ``abstract``, ``authors``, ``year``, ``venue``, ``doi``,
        ``keywords``.
        """
        obj = cls(**kwargs)
        for i, row in enumerate(papers):
            obj.papers.append(Paper.from_dict(row, index=i))
        return obj

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def fit(self) -> "LitCluster":
        """Tokenise papers, build TF-IDF vectors, and run k-means clustering.

        Returns *self* so calls can be chained::

            clusters = LitCluster.from_csv(path).fit().clusters
        """
        if not self.papers:
            return self

        tokens_list = [_tokenise(p.text) for p in self.papers]
        self._vectors, self._vocab = _tfidf(tokens_list, min_freq=self.min_term_freq)

        # Fall back to unfiltered tokens if min_freq discards everything
        if not self._vocab:
            self._vectors, self._vocab = _tfidf(tokens_list, min_freq=1)

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
        member_vecs = [
            self._vectors[i]
            for i, lbl in enumerate(self._labels)
            if lbl == cid
        ]
        if not member_vecs:
            return []
        scores: Dict[str, float] = {}
        size = len(member_vecs)
        for vec in member_vecs:
            for t, v in vec.items():
                scores[t] = scores.get(t, 0.0) + v / size
        return sorted(scores, key=lambda t: -scores[t])[:n]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, path: Path) -> None:
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

    def export_json(self, path: Path) -> None:
        """Write clusters (with all paper metadata) to a JSON file."""
        with Path(path).open("w", encoding="utf-8") as fh:
            json.dump(
                [c.to_dict() for c in self.clusters],
                fh,
                indent=2,
                ensure_ascii=False,
            )

    def export_html(
        self,
        path: Path,
        title: str = "Literature Cluster Report",
    ) -> None:
        """Write a self-contained interactive HTML report.

        The output file requires no internet connection and has no external
        dependencies.
        """
        Path(path).write_text(
            _build_html(self.clusters, title=title),
            encoding="utf-8",
        )

    def summary(self) -> str:
        """Return a Markdown-formatted summary table of clustering results."""
        lines = [
            "# LitCluster Report",
            "",
            f"**{len(self.papers)} papers** — **{len(self.clusters)} clusters**",
            "",
            "| # | Label | Papers | Top Terms |",
            "|---|-------|-------:|-----------|",
        ]
        for c in self.clusters:
            terms = ", ".join(c.top_terms[:5]) if c.top_terms else "—"
            lines.append(
                f"| {c.cluster_id} | {c.label} | {len(c.papers)} | {terms} |"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BibTeX field extraction
# ---------------------------------------------------------------------------

def _bibtex_field(entry: str, field: str) -> str:
    """Extract the value of *field* from a single BibTeX entry string.

    Handles brace-delimited values (including nested braces), quote-delimited
    values, and bare numeric values (e.g. ``year = 2023``).
    """
    pattern = re.compile(
        r'\b' + re.escape(field) + r'\s*=\s*',
        re.IGNORECASE,
    )
    m = pattern.search(entry)
    if not m:
        return ""
    pos = m.end()
    if pos >= len(entry):
        return ""

    # Skip leading whitespace
    while pos < len(entry) and entry[pos] in ' \t':
        pos += 1

    if pos >= len(entry):
        return ""

    opener = entry[pos]

    if opener == '{':
        # Brace-delimited: count depth to find matching close
        depth = 1
        start = pos + 1
        pos = start
        while pos < len(entry) and depth > 0:
            if entry[pos] == '{':
                depth += 1
            elif entry[pos] == '}':
                depth -= 1
            pos += 1
        value = entry[start:pos - 1]
    elif opener == '"':
        # Quote-delimited
        start = pos + 1
        end = entry.find('"', start)
        if end == -1:
            return ""
        value = entry[start:end]
    else:
        # Bare value (e.g. year = 2023)
        end = len(entry)
        for ch in (',', '}', '\n'):
            idx = entry.find(ch, pos)
            if idx != -1 and idx < end:
                end = idx
        value = entry[pos:end]

    return re.sub(r'\s+', ' ', value).strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="litcluster",
        description=(
            "Cluster academic papers by topic using TF-IDF + k-means. "
            "Input may be a BibTeX (.bib), CSV, or JSONL file."
        ),
    )
    p.add_argument("input", help="Input file (.bib, .csv, or .jsonl)")
    p.add_argument(
        "-k", "--clusters",
        type=int, default=5, dest="k",
        help="Number of clusters (default: 5)",
    )
    p.add_argument(
        "--format",
        choices=["summary", "csv", "json", "html"],
        default="summary",
        help="Output format (default: summary)",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: auto-named beside input)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--max-iter",
        type=int, default=100,
        help="Maximum k-means iterations (default: 100)",
    )
    p.add_argument(
        "--min-freq",
        type=int, default=2,
        help="Minimum document frequency for vocabulary terms (default: 2)",
    )
    p.add_argument(
        "--title",
        default="Literature Cluster Report",
        help="Report title used in HTML export",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
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
    try:
        if suffix == ".bib":
            lc = LitCluster.from_bibtex(path, **kwargs)
        elif suffix == ".jsonl":
            lc = LitCluster.from_jsonl(path, **kwargs)
        else:
            lc = LitCluster.from_csv(path, **kwargs)
    except Exception as exc:
        print(f"Error reading '{path}': {exc}", file=sys.stderr)
        return 1

    if not lc.papers:
        print("No papers found in input file.", file=sys.stderr)
        return 1

    lc.fit()

    fmt = args.format
    if fmt == "summary":
        output = lc.summary()
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"Summary written to {args.output}")
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
        lc.export_html(out, title=args.title)
        print(f"HTML report written to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
