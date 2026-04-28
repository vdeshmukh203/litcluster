#!/usr/bin/env python3
"""
litcluster_gui — Graphical interface for LitCluster.

Launch with:
    python litcluster_gui.py
    litcluster-gui          # after pip install
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
import tkinter.ttk as ttk

# Support running directly from the repo without installation
sys.path.insert(0, str(Path(__file__).parent))
from litcluster import LitCluster, __version__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _natural_sort_key(text: str):
    """Sort key that handles embedded integers naturally."""
    import re
    parts = re.split(r'(\d+)', text)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class LitClusterApp:
    """Tkinter front-end for LitCluster."""

    _FORMATS = ("summary", "csv", "json")
    _PAD = {"padx": 6, "pady": 4}

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"LitCluster {__version__} — Literature Clustering")
        self.root.minsize(740, 560)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self._lc: LitCluster | None = None
        self._input_path: Path | None = None

        self._build_toolbar()
        self._build_results()
        self._build_statusbar()
        self._set_status("Ready. Load a BibTeX (.bib), CSV, or JSONL file to begin.")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(1, weight=1)

        # ---- Row 0: file selector ----
        ttk.Label(toolbar, text="Input file:").grid(row=0, column=0, sticky="w", **self._PAD)

        self._path_var = tk.StringVar()
        path_entry = ttk.Entry(toolbar, textvariable=self._path_var, state="readonly")
        path_entry.grid(row=0, column=1, sticky="ew", **self._PAD)

        ttk.Button(toolbar, text="Browse…", command=self._browse).grid(
            row=0, column=2, sticky="e", **self._PAD)

        # ---- Row 1: parameters ----
        params = ttk.LabelFrame(toolbar, text="Parameters", padding=4)
        params.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        labels_vars = [
            ("Clusters (k):", "k_var", 5, 1, 99),
            ("Seed:", "seed_var", 42, 0, 9999),
            ("Max iterations:", "iter_var", 100, 1, 9999),
            ("Min term freq:", "freq_var", 2, 1, 999),
        ]
        self.k_var = tk.IntVar(value=5)
        self.seed_var = tk.IntVar(value=42)
        self.iter_var = tk.IntVar(value=100)
        self.freq_var = tk.IntVar(value=2)

        spin_cfg = [
            ("Clusters (k):", self.k_var, 1, 99),
            ("Seed:", self.seed_var, 0, 9999),
            ("Max iterations:", self.iter_var, 1, 9999),
            ("Min term freq:", self.freq_var, 1, 999),
        ]
        for col, (lbl, var, lo, hi) in enumerate(spin_cfg):
            ttk.Label(params, text=lbl).grid(row=0, column=col * 2, sticky="e", padx=(8, 2))
            sb = ttk.Spinbox(params, textvariable=var, from_=lo, to=hi, width=6)
            sb.grid(row=0, column=col * 2 + 1, sticky="w", padx=(0, 8))

        # ---- Row 2: run + export ----
        actions = ttk.Frame(toolbar)
        actions.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0))

        self._run_btn = ttk.Button(
            actions, text="▶  Run Clustering", command=self._run_async,
        )
        self._run_btn.pack(side="left", padx=(0, 12))

        ttk.Separator(actions, orient="vertical").pack(side="left", fill="y", padx=6)

        ttk.Label(actions, text="Export:").pack(side="left", padx=(0, 4))
        ttk.Button(actions, text="CSV", command=lambda: self._export("csv")).pack(
            side="left", padx=2)
        ttk.Button(actions, text="JSON", command=lambda: self._export("json")).pack(
            side="left", padx=2)
        ttk.Button(actions, text="Summary TXT", command=lambda: self._export("summary")).pack(
            side="left", padx=2)

    def _build_results(self) -> None:
        nb = ttk.Notebook(self.root)
        nb.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        self._notebook = nb

        # ---- Summary tab ----
        summary_frame = ttk.Frame(nb, padding=4)
        nb.add(summary_frame, text="Summary")
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)

        self._summary_text = tk.Text(
            summary_frame, wrap="word", font=("Courier", 10),
            state="disabled", relief="flat",
        )
        sb1 = ttk.Scrollbar(summary_frame, command=self._summary_text.yview)
        self._summary_text.configure(yscrollcommand=sb1.set)
        self._summary_text.grid(row=0, column=0, sticky="nsew")
        sb1.grid(row=0, column=1, sticky="ns")

        # ---- Clusters tab ----
        cluster_frame = ttk.Frame(nb, padding=4)
        nb.add(cluster_frame, text="Clusters")
        cluster_frame.columnconfigure(0, weight=1)
        cluster_frame.rowconfigure(0, weight=1)

        cols = ("cluster", "size", "top_terms")
        self._cluster_tree = ttk.Treeview(
            cluster_frame, columns=cols, show="headings", height=10,
        )
        self._cluster_tree.heading("cluster", text="Cluster")
        self._cluster_tree.heading("size", text="Papers")
        self._cluster_tree.heading("top_terms", text="Top Terms")
        self._cluster_tree.column("cluster", width=130, anchor="w")
        self._cluster_tree.column("size", width=60, anchor="center")
        self._cluster_tree.column("top_terms", width=420, anchor="w")
        sb2 = ttk.Scrollbar(cluster_frame, command=self._cluster_tree.yview)
        self._cluster_tree.configure(yscrollcommand=sb2.set)
        self._cluster_tree.grid(row=0, column=0, sticky="nsew")
        sb2.grid(row=0, column=1, sticky="ns")
        self._cluster_tree.bind("<<TreeviewSelect>>", self._on_cluster_select)

        # ---- Papers tab ----
        papers_frame = ttk.Frame(nb, padding=4)
        nb.add(papers_frame, text="Papers")
        papers_frame.columnconfigure(0, weight=1)
        papers_frame.rowconfigure(0, weight=1)

        pcols = ("cluster", "paper_id", "title", "year", "authors")
        self._papers_tree = ttk.Treeview(
            papers_frame, columns=pcols, show="headings", height=18,
        )
        for col, w in zip(pcols, (100, 100, 320, 55, 160)):
            self._papers_tree.heading(col, text=col.replace("_", " ").title())
            self._papers_tree.column(col, width=w, anchor="w")
        sb3 = ttk.Scrollbar(papers_frame, command=self._papers_tree.yview)
        self._papers_tree.configure(yscrollcommand=sb3.set)
        self._papers_tree.grid(row=0, column=0, sticky="nsew")
        sb3.grid(row=0, column=1, sticky="ns")

    def _build_statusbar(self) -> None:
        bar = ttk.Frame(self.root, relief="sunken", padding=(6, 2))
        bar.grid(row=2, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)

        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(bar, textvariable=self._status_var, anchor="w").grid(
            row=0, column=0, sticky="ew")

        self._progress = ttk.Progressbar(bar, mode="indeterminate", length=120)
        self._progress.grid(row=0, column=1, sticky="e", padx=(8, 0))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        filetypes = [
            ("Supported files", "*.bib *.csv *.jsonl"),
            ("BibTeX files", "*.bib"),
            ("CSV files", "*.csv"),
            ("JSONL files", "*.jsonl"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(
            title="Select input file", filetypes=filetypes,
        )
        if path:
            self._path_var.set(path)
            self._input_path = Path(path)
            self._set_status(f"Loaded: {path}")

    def _run_async(self) -> None:
        if not self._input_path or not self._input_path.is_file():
            messagebox.showerror("No file", "Please select a valid input file first.")
            return
        try:
            k = self.k_var.get()
            seed = self.seed_var.get()
            max_iter = self.iter_var.get()
            min_freq = self.freq_var.get()
        except tk.TclError:
            messagebox.showerror("Invalid parameters", "All parameters must be integers.")
            return

        self._run_btn.configure(state="disabled")
        self._progress.start(10)
        self._set_status("Clustering…")
        threading.Thread(
            target=self._run_clustering,
            args=(k, seed, max_iter, min_freq),
            daemon=True,
        ).start()

    def _run_clustering(self, k: int, seed: int, max_iter: int, min_freq: int) -> None:
        try:
            path = self._input_path
            kwargs = dict(k=k, seed=seed, max_iter=max_iter, min_term_freq=min_freq)
            suffix = path.suffix.lower()
            if suffix == ".bib":
                lc = LitCluster.from_bibtex(path, **kwargs)
            elif suffix == ".jsonl":
                lc = LitCluster.from_jsonl(path, **kwargs)
            else:
                lc = LitCluster.from_csv(path, **kwargs)

            if not lc.papers:
                raise ValueError("No papers found in the input file.")

            lc.fit()
            self._lc = lc
            self.root.after(0, self._display_results)
        except Exception as exc:
            self.root.after(0, lambda: self._on_error(str(exc)))

    def _on_cluster_select(self, _event=None) -> None:
        """When a cluster row is selected, jump to Papers tab filtered by it."""
        selection = self._cluster_tree.selection()
        if not selection or self._lc is None:
            return
        item = self._cluster_tree.item(selection[0])
        cid_str = item["values"][0]
        # cid_str is like "Cluster 2: term1, term2, term3" — parse the integer
        import re
        m = re.match(r"Cluster\s+(\d+)", str(cid_str))
        if not m:
            return
        cid = int(m.group(1))
        self._populate_papers(filter_cid=cid)
        self._notebook.select(2)  # switch to Papers tab

    # ------------------------------------------------------------------
    # Result population
    # ------------------------------------------------------------------

    def _display_results(self) -> None:
        self._progress.stop()
        self._run_btn.configure(state="normal")
        lc = self._lc
        n_papers = len(lc.papers)
        n_clusters = len(lc.clusters)
        self._set_status(
            f"Done: {n_papers} papers → {n_clusters} clusters  "
            f"(k={lc.k}, seed={lc.seed})"
        )

        # Summary tab
        self._summary_text.configure(state="normal")
        self._summary_text.delete("1.0", "end")
        self._summary_text.insert("end", lc.summary())
        self._summary_text.configure(state="disabled")

        # Clusters tab
        for row in self._cluster_tree.get_children():
            self._cluster_tree.delete(row)
        for c in lc.clusters:
            self._cluster_tree.insert(
                "", "end",
                values=(c.label, len(c.papers), ", ".join(c.top_terms)),
            )

        # Papers tab (all)
        self._populate_papers()
        self._notebook.select(0)

    def _populate_papers(self, filter_cid: int | None = None) -> None:
        for row in self._papers_tree.get_children():
            self._papers_tree.delete(row)
        if self._lc is None:
            return
        for c in self._lc.clusters:
            if filter_cid is not None and c.cluster_id != filter_cid:
                continue
            tag = f"c{c.cluster_id % 2}"
            for p in c.papers:
                self._papers_tree.insert(
                    "", "end", tags=(tag,),
                    values=(c.cluster_id, p.paper_id, p.title, p.year, p.authors),
                )
        self._papers_tree.tag_configure("c0", background="#f0f4ff")
        self._papers_tree.tag_configure("c1", background="#ffffff")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(self, fmt: str) -> None:
        if self._lc is None:
            messagebox.showinfo("No results", "Run clustering first.")
            return
        if fmt == "csv":
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export CSV",
            )
            if path:
                self._lc.export_csv(Path(path))
                self._set_status(f"Exported CSV → {path}")
        elif fmt == "json":
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Export JSON",
            )
            if path:
                self._lc.export_json(Path(path))
                self._set_status(f"Exported JSON → {path}")
        elif fmt == "summary":
            path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Export Summary",
            )
            if path:
                Path(path).write_text(self._lc.summary(), encoding="utf-8")
                self._set_status(f"Exported summary → {path}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _on_error(self, msg: str) -> None:
        self._progress.stop()
        self._run_btn.configure(state="normal")
        self._set_status(f"Error: {msg}")
        messagebox.showerror("Clustering error", msg)

    def _set_status(self, msg: str) -> None:
        self._status_var.set(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    app = LitClusterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
