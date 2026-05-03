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
date: 23 April 2026
bibliography: paper.bib
---

# Summary

`litcluster` is a lightweight Python tool for automated topic clustering of
scientific literature.  Given a BibTeX file, a CSV, or a JSONL collection of
paper metadata, `litcluster` tokenises title, abstract, and keyword text;
builds sparse TF-IDF vectors [@salton1988]; and partitions papers into
thematic groups using Lloyd's k-means algorithm [@lloyd1982] with cosine
similarity.  For each cluster, the tool extracts the top discriminative terms
by aggregate TF-IDF score, producing interpretable cluster labels without
manual annotation.  Results can be exported as structured CSV or JSON and
inspected interactively through a built-in graphical interface.

`litcluster` requires only the Python standard library (Python >= 3.8) and
introduces no external dependencies.  It is reproducible by design: the k-means
initialisation is seeded so that repeated runs on the same input always return
the same partition.

# Statement of Need

Systematic and scoping reviews require researchers to organise large collections
of papers into thematic groups — a task typically done manually or with
proprietary commercial tools [@beller2018].  Automated approaches that depend on
large pre-trained language models impose significant computational costs and
introduce versioning complexity, making them impractical for researchers without
access to GPU hardware or stable internet connectivity.

`litcluster` addresses this gap by providing a zero-dependency, reproducible,
and computationally inexpensive baseline for literature triage.  TF-IDF
vectorisation [@salton1988] captures term-frequency patterns that correlate
strongly with topical similarity, and cosine-based k-means produces compact,
well-separated clusters for collections of dozens to several hundred papers
[@macqueen1967].  The `min_term_freq` parameter filters rare terms that often
reflect typographical variation rather than genuine topics, improving cluster
coherence.  The seeded initialisation and deterministic tokenisation pipeline
ensure that results can be fully reproduced from the same input file
[@gundersen2018state; @stodden2016enhancing].

The tool is aimed at researchers who need a quick, auditable first-pass
organisation of a literature search result — for example, to identify core
topic areas before undertaking manual full-text screening — and who prefer a
tool they can inspect and modify rather than a black-box web service.

# Implementation

`litcluster` is implemented as a single Python module (`litcluster.py`) with no
third-party dependencies.  The public API consists of three classes — `Paper`,
`Cluster`, and `LitCluster` — and four pure functions (`_tokenise`, `_tfidf`,
`_cosine`, `_kmeans`) that are independently testable and reusable.

The processing pipeline is as follows:

1. **Ingestion.**  `LitCluster.from_bibtex()`, `from_csv()`, and `from_jsonl()`
   parse input files into a list of `Paper` dataclass instances.  The BibTeX
   parser handles nested braces and multi-line field values.
2. **Tokenisation.**  `_tokenise()` lower-cases input text, extracts alphabetic
   tokens of length >= 3, and removes a curated English stopword list.
3. **Vocabulary filtering.**  Terms appearing in fewer than `min_term_freq`
   documents are removed before vectorisation.
4. **TF-IDF vectorisation.**  `_tfidf()` computes smoothed TF-IDF weights
   (`idf(t) = log((N+1)/(df(t)+1)) + 1`) and returns sparse vectors as Python
   dicts, avoiding the memory overhead of dense matrices for large vocabularies.
5. **Clustering.**  `_kmeans()` implements Lloyd's algorithm [@lloyd1982] with
   random-seed initialisation; empty clusters are re-seeded to a random document
   rather than left empty.
6. **Labelling.**  For each cluster, the top-10 terms by summed TF-IDF score
   across member documents are returned as the cluster label.

A graphical user interface built with Python's built-in `tkinter` library
(`litcluster_gui.py`) provides file-browser, parameter controls, a hierarchical
cluster/paper tree view, a detail panel, and one-click CSV/JSON export — all
without additional installation steps.

# Testing

The test suite (`tests/test_litcluster.py`) covers all four core functions and
the three I/O loaders with 62 unit and integration tests using `pytest`
[@pytest2024].  Tests include edge cases such as empty input, single-document
collections, deeply nested BibTeX braces, and round-trip export verification.

# Acknowledgements

The author used Claude (Anthropic) for drafting portions of this manuscript and
for code assistance.  All scientific claims and design decisions are the
author's own.

# References
