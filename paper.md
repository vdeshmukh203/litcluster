---
title: 'litcluster: Topic-based clustering of scientific literature using TF-IDF and k-means'
tags:
  - Python
  - NLP
  - topic-modelling
  - clustering
  - literature-review
  - bibliometrics
authors:
  - name: Vaibhav Deshmukh
    orcid: 0000-0001-6745-7062
    affiliation: 1
affiliations:
  - name: Independent Researcher, Nagpur, India
    index: 1
date: 23 April 2026
bibliography: paper.bib
---

# Summary

`litcluster` is a pure-Python command-line tool and library for automatic
topic clustering of scientific literature collections.  Given a BibTeX file,
a CSV, or a JSON Lines file containing paper titles, abstracts, and keywords,
`litcluster` (1) tokenises and filters the text, (2) computes smoothed
TF-IDF (term-frequency–inverse-document-frequency) vectors [@salton1988term]
for each paper, and (3) partitions the papers into $k$ topic clusters using
Lloyd's k-means algorithm with cosine similarity [@lloyd1982least].  The tool
outputs a plain-text summary, structured CSV or JSON exports, a self-contained
interactive HTML report, and a Tkinter graphical interface for exploratory
use.  No external dependencies are required: the entire implementation relies
exclusively on the Python standard library, making it trivially portable and
reproducible.

# Statement of Need

Systematic and scoping literature reviews require researchers to organise
large collections of papers into thematic groups — a task typically performed
manually or with expensive proprietary software.  Free alternatives often
require complex dependency stacks (PyTorch, sentence-transformers, UMAP,
HDBSCAN) whose installation and version management create reproducibility
barriers [@gundersen2018state; @pineau2021improving].

`litcluster` is designed for researchers who need a *lightweight, zero-dependency
entry point* into computational literature organisation.  The entire pipeline
runs with `pip install litcluster` and a single command:

```bash
litcluster refs.bib -k 8 --format html -o report.html
```

Unlike black-box neural embedding approaches, TF-IDF vectors are fully
interpretable: each dimension corresponds to a specific term, and per-cluster
top terms immediately explain the thematic content of each group.  This
transparency supports the methodological accountability expected in systematic
reviews [@page2021prisma].

`litcluster` accepts BibTeX files exported directly from Zotero, Mendeley,
JabRef, or any reference manager, requiring no additional data preparation.
It also accepts CSV and JSON Lines formats to support programmatic pipelines.

# Implementation

## Text representation

Each paper is represented as the concatenation of its title, abstract, and
author-supplied keywords.  Tokenisation applies a regular-expression-based
word boundary split, discards English stopwords and tokens shorter than three
characters, and yields a bag-of-words token list.  A vocabulary-level minimum
document frequency filter (configurable, default 2) removes rare terms that
would add noise without discriminative value.

TF-IDF weighting uses the smoothed formulation:

$$\text{TF-IDF}(t, d) = \frac{f_{t,d}}{|d|} \cdot \left(\log\frac{N+1}{\mathrm{df}(t)+1} + 1\right)$$

where $f_{t,d}$ is the raw term frequency in document $d$, $|d|$ is the
document length in tokens, $N$ is the total number of documents, and
$\mathrm{df}(t)$ is the number of documents containing term $t$.  Vectors
are stored as sparse Python dictionaries to keep memory usage proportional to
the non-zero entries.

## Clustering

Lloyd's k-means algorithm is applied with cosine similarity as the distance
metric, which is standard practice for sparse TF-IDF vectors [@manning2008introduction].
Centroid initialisation uses a seeded random sample; the random seed is fully
configurable, guaranteeing reproducible results across runs.  If the
requested $k$ exceeds the number of papers, it is silently capped.

## Outputs

The `summary` format prints a human-readable cluster table to standard output.
The `csv` format writes one row per paper with cluster assignment.
The `json` format produces a nested structure suitable for downstream
programmatic processing.
The `html` format generates a self-contained, zero-dependency HTML report
with collapsible cluster sections, colour-coded by cluster index, and DOI
hyperlinks for each paper.  The graphical interface (`litcluster-gui`)
provides interactive file loading, parameter adjustment, cluster browsing,
and one-click export.

# Acknowledgements

The author used Claude (Anthropic) for drafting portions of this manuscript.
All scientific claims and design decisions are the author's own.

# References
