---
title: 'litcluster: Topic modelling and semantic clustering of scientific literature from BibTeX or DOI lists'
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

`litcluster` is a Python tool for semantic clustering and topic modelling of scientific literature. Given a BibTeX file, a list of DOIs, or a directory of PDF abstracts, `litcluster` fetches or extracts abstract text, embeds papers using pre-trained sentence transformers, applies dimensionality reduction (UMAP), and clusters the resulting embedding space (HDBSCAN). It outputs an interactive HTML visualisation of the literature landscape and a Markdown summary table labelling each cluster with automatically extracted topic keywords. `litcluster` is designed for researchers beginning a new domain survey or tracking how a field has evolved over time.

# Statement of Need

Systematic and scoping reviews require researchers to organise large collections of papers into thematic groups — a task typically done manually or with expensive proprietary tools. `litcluster` automates this using modern sentence embeddings and density-based clustering, requiring only a BibTeX file as input. Unlike keyword-based approaches, semantic clustering groups papers by meaning rather than surface vocabulary, surfacing connections that term-based methods miss [@wei2022chain]. The interactive output helps researchers quickly identify core topics, niche sub-areas, and potential gaps in a literature collection, accelerating the early stages of a systematic review.

# Acknowledgements

The author used Claude (Anthropic) for drafting portions of this manuscript. All scientific claims and design decisions are the author's own.

# References
