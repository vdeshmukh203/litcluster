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
date: 29 April 2026
bibliography: paper.bib
---

# Summary

`litcluster` is a pure-Python tool for clustering collections of academic
papers by topic. Given a BibTeX file, a CSV export, or a JSON Lines file,
`litcluster` tokenises each paper's title, abstract, and keywords; constructs
smoothed TF-IDF vectors; and partitions the papers into *k* thematic groups
using Lloyd's k-means algorithm with cosine similarity. Each cluster is
labelled automatically with its highest-scoring TF-IDF terms. Results can be
exported as a plain-text summary, a flat CSV table, or a structured JSON file
for further downstream processing. A Tkinter graphical interface is also
provided for users who prefer a point-and-click workflow.

The tool is implemented entirely in the Python standard library (Python ≥ 3.8)
and therefore requires no package installation beyond `pip install litcluster`
itself, making it straightforward to use in air-gapped or restricted
computational environments.

# Statement of Need

Systematic and scoping reviews require researchers to organise large
collections of papers into thematic groups — a task typically performed
manually or with expensive proprietary software. Existing open-source
tools either require heavy dependencies (scikit-learn, sentence-transformers,
UMAP-learn) that are difficult to install in constrained environments, or
offer only keyword-search rather than text-similarity-based grouping
[@wei2022chain].

`litcluster` fills the gap for researchers who need a lightweight,
reproducible clustering tool they can run immediately from a standard Python
installation. The TF-IDF + cosine k-means approach has well-understood
properties [@salton1988term], is fast enough for collections of several
thousand papers on a laptop, and produces interpretable cluster labels that
directly reflect the vocabulary of the literature. The seeded random
initialisation and deterministic algorithm ensure that results are fully
reproducible across platforms.

# Implementation

## Input parsing

`litcluster` accepts three input formats:

- **BibTeX** (`.bib`) — parsed with a regular-expression scanner that
  extracts `title`, `abstract`, `author`, `year`, `journal`/`booktitle`,
  `doi`, and `keywords` fields.
- **CSV** — standard comma-separated values with an optional header row.
- **JSON Lines** (`.jsonl`) — one JSON object per line.

All formats map to the same internal `Paper` dataclass.

## Text vectorisation

Each paper's title, abstract, and keywords are concatenated and tokenised:
alphabetic tokens of length ≥ 3 are lowercased and filtered against a 70-word
English stopword list. Terms appearing in fewer than `min_term_freq` documents
(default: 2) are discarded to reduce noise from idiosyncratic spellings.

TF-IDF weights are computed as

$$w_{t,d} = \text{tf}_{t,d} \times \left(\log\frac{N+1}{\text{df}_t+1} + 1\right)$$

where $N$ is the number of documents and $\text{df}_t$ is the document
frequency of term $t$. This matches the `smooth_idf=True` convention of
scikit-learn's `TfidfTransformer` [@pedregosa2011scikit].

## Clustering

Lloyd's k-means algorithm [@lloyd1982least] is applied in the TF-IDF vector
space using cosine similarity as the proximity measure. Centroids are
initialised by sampling *k* distinct documents uniformly at random (seeded for
reproducibility). Empty clusters that arise during iteration are reinitialised
to a randomly chosen document to prevent degenerate solutions.

## Topic labelling

Each cluster is characterised by the terms with the highest aggregate TF-IDF
score across its member documents, providing an interpretable keyword label for
the discovered topic.

## Interfaces

`litcluster` exposes three interfaces:

1. **Python API** — `LitCluster.from_bibtex()`, `.from_csv()`, `.from_jsonl()`,
   `.fit()`, `.export_csv()`, `.export_json()`, `.summary()`.
2. **CLI** — `litcluster <file> -k <n> [--format csv|json|summary]`.
3. **GUI** — a Tkinter window (`litcluster-gui`) for file selection, parameter
   tuning, result inspection, and export.

# Acknowledgements

The author used Claude (Anthropic) for drafting portions of this manuscript.
All scientific claims and design decisions are the author's own.

# References
