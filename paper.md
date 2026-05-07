---
title: 'litcluster: Topic-based clustering of scientific literature from BibTeX or CSV'
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

`litcluster` is a pure-Python tool for automated topic-based clustering of
scientific literature.  Given a BibTeX file, a CSV spreadsheet, or a
JSON-Lines collection of paper abstracts, `litcluster` vectorises the text
using smoothed TF-IDF, groups papers into thematic clusters with Lloyd's
k-means algorithm on cosine distance, and reports a silhouette-score quality
estimate for the chosen number of clusters.  Results can be exported as a
plain-text summary, a CSV assignment table, a structured JSON file, or a
self-contained interactive HTML report.  A companion Tkinter graphical user
interface (GUI) makes the tool accessible to researchers who prefer not to
use the command line.  `litcluster` has no external dependencies beyond the
Python standard library and runs on any platform where Python 3.8 or later is
installed.

# Statement of Need

Systematic and scoping reviews require researchers to organise large
collections of papers into thematic groups — a task typically done manually
or with expensive proprietary tools such as Covidence or Rayyan.
`litcluster` automates thematic grouping using TF-IDF vectorisation and
k-means clustering, requiring only a BibTeX file as input.  Unlike fully
manual approaches, `litcluster` provides a reproducible, quantitative
starting point for thematic organisation and reports the most discriminative
vocabulary terms for each cluster, helping researchers label and interpret
the groups.  The silhouette-score quality metric guides the choice of the
number of clusters.  The interactive HTML report and Tkinter GUI lower the
barrier for researchers with limited programming experience.  Because the
tool is a single Python file with no external dependencies, it can be
deployed in any Python environment without complex installation steps,
making it suitable for use in institutional computing environments with
restricted package access.

# Acknowledgements

The author used Claude (Anthropic) for drafting portions of this manuscript.
All scientific claims and design decisions are the author's own.

# References
