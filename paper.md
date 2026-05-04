---
title: 'litcluster: Topic-based clustering of scientific literature using TF-IDF and k-means'
tags:
  - Python
  - NLP
  - topic-modelling
  - clustering
  - literature-review
  - bibliometrics
  - systematic-review
authors:
  - name: Vaibhav Deshmukh
    orcid: 0000-0001-6745-7062
    affiliation: 1
affiliations:
  - name: Independent Researcher, Nagpur, India
    index: 1
date: 4 May 2026
bibliography: paper.bib
---

# Summary

`litcluster` is a dependency-free Python tool for topic-based clustering and
organisation of scientific literature.  Given a BibTeX file, a CSV spreadsheet,
or a JSON Lines file of paper metadata, `litcluster` tokenises title, abstract,
and keyword text, builds TF-IDF (Term Frequency–Inverse Document Frequency)
vectors for each document, and groups the collection into *k* thematic clusters
using Lloyd's k-means algorithm with cosine similarity.  Each cluster is
automatically labelled with its most discriminative TF-IDF terms.  Results can
be exported as a human-readable summary, a flat CSV, or a structured JSON file.
An optional desktop GUI (built on the standard-library `tkinter` toolkit)
provides point-and-click access to the full pipeline without requiring any
command-line interaction.

`litcluster` uses only Python's standard library; it has no third-party
dependencies and installs instantly in any Python 3.8+ environment.

# Statement of Need

Systematic and scoping reviews require researchers to organise potentially
hundreds of papers into thematic groups — a task that is time-consuming when
performed manually and typically requires either expensive proprietary software
or complex NLP toolchains with heavy dependencies.

`litcluster` fills the gap between rudimentary keyword search and heavyweight
machine-learning pipelines.  It requires only a BibTeX export from any
reference manager (Zotero, Mendeley, EndNote) or a simple CSV spreadsheet, and
produces cluster assignments and topic keywords in seconds with a single
command.  Because the tool relies on no external packages, it can be deployed in
restricted computing environments (e.g., institutional HPC clusters, air-gapped
systems) without additional installation steps.

The TF-IDF + k-means approach is well understood, deterministic given a fixed
random seed, and fast enough to process several thousand papers on a standard
laptop.  The explainable top-term labels make it straightforward for researchers
to interpret and validate cluster content — a property that is important for
reproducibility in systematic reviews [@kitchenham2007guidelines].

# Algorithm

## Text representation

Each paper is represented by the concatenation of its title, abstract, and
author-supplied keywords.  The text is lowercased, split on word boundaries
(3+ alphabetic characters), and filtered against an extended stopword list that
covers both common English function words and high-frequency academic phrases
(e.g., *results*, *proposed*, *method*) that carry no discriminating information
across disciplines.

Term frequencies are normalised by document length (raw TF) and combined with a
smoothed inverse document frequency:

$$\text{IDF}(t) = \log\!\left(\frac{n+1}{df(t)+1}\right) + 1$$

where $n$ is the number of documents and $df(t)$ is the number of documents
containing term $t$.  The additive smoothing prevents zero-IDF for terms that
appear in every document [@pedregosa2011scikit].

## Clustering

Lloyd's k-means algorithm [@lloyd1982least] is applied to the sparse TF-IDF
vectors using cosine similarity as the distance metric.  Cosine similarity is
preferable to Euclidean distance for text vectors because it is invariant to
document length.  Centroids are initialised by random sampling without
replacement, with the RNG seeded for full reproducibility.  Empty clusters
(which can arise if an initialised centroid has no nearest neighbours) are
re-seeded from a random existing document rather than being left degenerate.

The algorithm converges when cluster assignments are unchanged between
iterations, or after a user-specified maximum number of iterations (default 100).

## Cluster labelling

Each cluster is labelled with its top-10 terms by aggregate TF-IDF score across
member documents, providing a human-readable summary of each topic.

# Features

- **Input formats**: BibTeX (`.bib`), CSV, and JSON Lines (`.jsonl`).
- **Output formats**: plain-text summary, flat CSV of assignments, and
  structured JSON with full paper metadata.
- **CLI**: `litcluster refs.bib -k 8 --format json -o clusters.json`
- **Python API**: `LitCluster.from_bibtex("refs.bib", k=5).fit()`
- **GUI**: `python litcluster_gui.py` — file browser, parameter spinboxes,
  interactive cluster/paper explorer, and CSV/JSON export.
- **Reproducibility**: deterministic output via `--seed`.
- **Zero dependencies**: Python 3.8+ standard library only.

# Acknowledgements

The author used Claude (Anthropic) for drafting portions of this manuscript.
All scientific claims and design decisions are the author's own.

# References
