"""
litcluster: Topic-based clustering of scientific literature using TF-IDF and k-means.

Ingests BibTeX files or JSONL records of papers (title + abstract), vectorises them
with TF-IDF, clusters with k-means (or hierarchical), and produces per-cluster keyword
summaries, intra-cluster similarity scores, and a CSV export for downstream analysis.
"""
from __future__ import annotations
import csv, json, math, re, random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Simple BibTeX title/abstract extractor
# ---------------------------------------------------------------------------

def _extract_bibtex_papers(text: str) -> List[Dict[str, str]]:
    papers = []
    entries = re.split(r"@\w+\s*\{", text)[1:]
    for entry in entries:
        title_m = re.search(r"title\s*=\s*\{(.+?)\}", entry, re.DOTALL | re.IGNORECASE)
        abs_m   = re.search(r"abstract\s*=\s*\{(.+?)\}", entry, re.DOTALL | re.IGNORECASE)
        key_m   = re.match(r"([^,]+),", entry)
        if title_m:
            papers.append({
                "key":      key_m.group(1).strip() if key_m else "",
                "title":    re.sub(r"\s+", " ", title_m.group(1)).strip(),
                "abstract": re.sub(r"\s+", " ", abs_m.group(1)).strip() if abs_m else "",
            })
    return papers


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset("""
a about above after again against all also am an and any are aren't as at
be because been before being below between both but by can can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for from
further get got had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over own
same shan't she she'd she'll she's should shouldn't so some such than that
that's the their theirs them themselves then there there's these they they'd
they'll they're they've this those through to too under until up very was wasn't
we we'd we'll we're we've were weren't what what's when when's where where's
which while who who's whom why why's will with won't would wouldn't you you'd
you'll you're you've your yours yourself yourselves
""".split())


def _tokenise(text: str) -> List[str]:
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]


def _compute_tfidf(documents: List[List[str]]) -> List[Dict[str, float]]:
    """Return per-document TF-IDF dicts."""
    n = len(documents)
    # IDF
    df: Counter = Counter()
    for doc in documents:
        for term in set(doc):
            df[term] += 1
    idf: Dict[str, float] = {
        term: math.log((n + 1) / (count + 1)) + 1
        for term, count in df.items()
    }
    vectors = []
    for doc in documents:
        tf: Counter = Counter(doc)
        total = max(len(doc), 1)
        vec = {term: (count / total) * idf[term] for term, count in tf.items()}
        # L2-normalise
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({k: v / norm for k, v in vec.items()})
    return vectors


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    shared = set(a) & set(b)
    return sum(a[k] * b[k] for k in shared)


# ---------------------------------------------------------------------------
# k-means
# ---------------------------------------------------------------------------

def _kmeans(
    vectors: List[Dict[str, float]],
    k: int,
    max_iter: int = 100,
    seed: int = 42,
) -> List[int]:
    """Return cluster assignments (list of int, length == len(vectors))."""
    rng = random.Random(seed)
    # k-means++ initialisation
    centroids: List[Dict[str, float]] = [dict(rng.choice(vectors))]
    while len(centroids) < k:
        dists = []
        for vec in vectors:
            d = min(1.0 - _cosine(vec, c) for c in centroids)
            dists.append(max(d, 0.0))
        total = sum(dists) or 1.0
        probs = [d / total for d in dists]
        cumulative = 0.0
        r = rng.random()
        chosen = len(vectors) - 1
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                chosen = i
                break
        centroids.append(dict(vectors[chosen]))

    assignments = [0] * len(vectors)
    for _ in range(max_iter):
        new_assign = []
        for vec in vectors:
            sims = [_cosine(vec, c) for c in centroids]
            new_assign.append(sims.index(max(sims)))
        if new_assign == assignments:
            break
        assignments = new_assign
        # Recompute centroids
        for ci in range(k):
            members = [vectors[i] for i, a in enumerate(assignments) if a == ci]
            if not members:
                centroids[ci] = dict(rng.choice(vectors))
                continue
            new_c: Dict[str, float] = {}
            for vec in members:
                for term, val in vec.items():
                    new_c[term] = new_c.get(term, 0.0) + val / len(members)
            norm = math.sqrt(sum(v * v for v in new_c.values())) or 1.0
            centroids[ci] = {k: v / norm for k, v in new_c.items()}

    return assignments


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    key: str
    title: str
    abstract: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return (self.title + " " + self.abstract).strip()


@dataclass
class Cluster:
    cluster_id: int
    papers: List[Paper] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    cohesion: float = 0.0          # mean intra-cluster cosine similarity

    @property
    def size(self) -> int:
        return len(self.papers)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

class LitCluster:
    """
    Cluster a collection of scientific papers by topic.

    Parameters
    ----------
    k : int
        Number of clusters.
    top_keywords : int
        Keywords to surface per cluster.
    max_iter : int
        Maximum k-means iterations.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        k: int = 5,
        top_keywords: int = 10,
        max_iter: int = 100,
        seed: int = 42,
    ) -> None:
        self.k = k
        self.top_keywords = top_keywords
        self.max_iter = max_iter
        self.seed = seed
        self._papers: List[Paper] = []
        self._clusters: List[Cluster] = []
        self._vectors: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    def add_papers(self, papers: List[Paper]) -> None:
        self._papers.extend(papers)

    def load_bibtex(self, path: str) -> int:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        raw = _extract_bibtex_papers(text)
        papers = [Paper(key=r["key"], title=r["title"], abstract=r["abstract"]) for r in raw]
        self._papers.extend(papers)
        return len(papers)

    def load_jsonl(self, path: str) -> int:
        count = 0
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self._papers.append(Paper(
                    key=d.get("key", d.get("id", str(count))),
                    title=d.get("title", ""),
                    abstract=d.get("abstract", ""),
                    metadata={k: v for k, v in d.items()
                               if k not in ("key", "id", "title", "abstract")},
                ))
                count += 1
        return count

    # ------------------------------------------------------------------
    def fit(self) -> List[Cluster]:
        """Vectorise papers and run k-means clustering. Returns cluster list."""
        if len(self._papers) < self.k:
            raise ValueError(
                "Number of papers (" + str(len(self._papers)) + ") < k (" + str(self.k) + ")."
            )
        tokenised = [_tokenise(p.text) for p in self._papers]
        self._vectors = _compute_tfidf(tokenised)
        assignments = _kmeans(self._vectors, k=self.k,
                               max_iter=self.max_iter, seed=self.seed)

        # Build clusters
        clusters = [Cluster(cluster_id=i) for i in range(self.k)]
        for i, (paper, assignment) in enumerate(zip(self._papers, assignments)):
            clusters[assignment].papers.append(paper)

        # Compute keywords and cohesion per cluster
        for cluster in clusters:
            if not cluster.papers:
                continue
            member_indices = [j for j, a in enumerate(assignments) if a == cluster.cluster_id]
            member_vecs = [self._vectors[j] for j in member_indices]
            # Keywords: top terms by mean TF-IDF across cluster members
            term_scores: Dict[str, float] = {}
            for vec in member_vecs:
                for term, score in vec.items():
                    term_scores[term] = term_scores.get(term, 0.0) + score
            sorted_terms = sorted(term_scores.items(), key=lambda x: -x[1])
            cluster.keywords = [t for t, _ in sorted_terms[:self.top_keywords]]
            # Cohesion: mean pairwise cosine
            if len(member_vecs) > 1:
                total_sim = 0.0
                count = 0
                for a_idx in range(len(member_vecs)):
                    for b_idx in range(a_idx + 1, len(member_vecs)):
                        total_sim += _cosine(member_vecs[a_idx], member_vecs[b_idx])
                        count += 1
                cluster.cohesion = round(total_sim / count, 4) if count else 0.0
            else:
                cluster.cohesion = 1.0

        self._clusters = sorted(clusters, key=lambda c: -c.size)
        return self._clusters

    # ------------------------------------------------------------------
    def to_markdown(self) -> str:
        lines = [
            "# LitCluster Report",
            "",
            "Total papers: " + str(len(self._papers)) + " | Clusters: " + str(self.k),
            "",
        ]
        for cluster in self._clusters:
            lines += [
                "## Cluster " + str(cluster.cluster_id) + " (" + str(cluster.size) + " papers)",
                "",
                "**Keywords**: " + ", ".join(cluster.keywords),
                "",
                "**Cohesion**: " + str(cluster.cohesion),
                "",
                "**Papers**:",
                "",
            ]
            for p in cluster.papers[:20]:
                lines.append("- " + p.title[:120])
            if cluster.size > 20:
                lines.append("- _(and " + str(cluster.size - 20) + " more)_")
            lines.append("")
        return "\n".join(lines)

    def to_csv(self, path: str) -> Path:
        out = Path(path)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["cluster_id", "key", "title", "abstract", "cohesion", "keywords"])
            for cluster in self._clusters:
                kw = "; ".join(cluster.keywords)
                for paper in cluster.papers:
                    writer.writerow([cluster.cluster_id, paper.key,
                                     paper.title, paper.abstract,
                                     cluster.cohesion, kw])
        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        prog="litcluster",
        description="Cluster scientific literature by topic using TF-IDF and k-means.",
    )
    parser.add_argument("input", help="BibTeX (.bib) or JSONL (.jsonl) file.")
    parser.add_argument("-k", "--clusters", type=int, default=5, dest="k")
    parser.add_argument("--keywords", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output-csv", default=None)
    parser.add_argument("--report", default=None, help="Write Markdown report to path.")
    args = parser.parse_args()

    lc = LitCluster(k=args.k, top_keywords=args.keywords, seed=args.seed)
    path = Path(args.input)
    if path.suffix.lower() == ".bib":
        n = lc.load_bibtex(str(path))
    else:
        n = lc.load_jsonl(str(path))
    print("Loaded " + str(n) + " papers.")

    clusters = lc.fit()
    for cluster in clusters:
        print("Cluster " + str(cluster.cluster_id) + " (" + str(cluster.size) + " papers): " + ", ".join(cluster.keywords[:5]))

    if args.output_csv:
        lc.to_csv(args.output_csv)
        print("CSV written to " + args.output_csv)
    if args.report:
        Path(args.report).write_text(lc.to_markdown(), encoding="utf-8")
        print("Report written to " + args.report)


if __name__ == "__main__":
    _cli()
