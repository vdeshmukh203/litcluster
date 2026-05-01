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

`litcluster` is a dependency-free Python tool for topic-based clustering of scientific literature. Given a BibTeX file, a CSV spreadsheet, or a JSONL file of paper records, `litcluster` tokenises each paper's title, abstract, and keywords, builds a sparse term frequency–inverse document frequency (TF-IDF) vector for each paper [@salton1988term], and groups papers into *k* thematic clusters using Lloyd's k-means algorithm with k-means++ initialisation [@lloyd1982least; @arthur2007kmeans]. The tool outputs a Markdown summary table, a CSV or JSON cluster assignment file, and a fully self-contained interactive HTML report with collapsible cluster cards, top-term badges, and a real-time paper search panel. A Tkinter graphical user interface is included for researchers who prefer not to use the command line. All functionality uses only the Python standard library; no external packages are required.

# Statement of Need

Systematic and scoping literature reviews require researchers to organise large collections of papers — often hundreds or thousands of items — into coherent thematic groups. In practice, this grouping is performed manually, which is time-consuming and introduces inter-reviewer variability, or by proprietary bibliometric software that may not be freely available to all researchers [@kitchenham2004procedures; @moher2009preferred].

Open-source alternatives exist but typically impose heavy dependency stacks (PyTorch, sentence-transformers, UMAP, HDBSCAN) that create installation barriers, particularly on shared computing environments or institutional machines with restricted package management. `litcluster` addresses this gap by delivering a useful baseline clustering capability — TF-IDF vectorisation and k-means — that installs in seconds and runs on any Python 3.8+ interpreter without compiling C extensions or downloading pre-trained model weights.

The k-means++ initialisation strategy [@arthur2007kmeans] reduces sensitivity to the random seed and consistently produces tighter, more interpretable clusters than naive random initialisation, while the cosine-similarity distance metric is well-suited to sparse bag-of-words representations [@manning2008introduction]. Researchers with small to medium literature collections (tens to a few thousand papers) who need a fast, reproducible first-pass grouping will find `litcluster` immediately useful. Those who later require embedding-based clustering can export the cluster labels as CSV and use them as priors or evaluation baselines for more sophisticated pipelines.

# Functionality

`litcluster` exposes three interfaces that share the same underlying algorithm:

**Command-line interface.** A `litcluster` script accepts an input file, the desired number of clusters, and optional parameters, and writes the chosen output format to disk or stdout.

**Python API.** The `LitCluster` class provides class methods `from_bibtex`, `from_csv`, `from_jsonl`, and `from_list` for loading paper collections, a `fit()` method that executes the clustering pipeline, and `export_csv`, `export_json`, and `export_html` methods for writing results.

**Graphical user interface.** The `litcluster-gui` script launches a Tkinter window in which users can browse for an input file, adjust all clustering parameters with spin-boxes, run clustering in a background thread, inspect results in a tabbed panel (Summary / Clusters / Papers), and export in any supported format via a file-save dialog.

The interactive HTML export is a single, standalone file requiring no internet connection. It embeds cluster data as an inline JSON object and uses vanilla JavaScript for search filtering, cluster visibility toggling, and collapsible paper tables.

# Acknowledgements

The author used Claude (Anthropic) for drafting portions of this manuscript and for code review. All scientific claims and design decisions are the author's own.

# References
