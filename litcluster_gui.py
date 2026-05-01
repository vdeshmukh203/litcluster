#!/usr/bin/env python3
"""
litcluster_gui.py — Desktop GUI for litcluster

A Tkinter-based graphical interface that lets researchers load a BibTeX,
CSV, or JSONL file, configure clustering parameters, run the algorithm,
browse results by cluster, and export to CSV, JSON, or HTML.

No external packages are required beyond the Python standard library.
"""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

# Resolve litcluster from the same directory as this script
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import litcluster as lc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_APP_TITLE = "litcluster — Literature Clustering Tool"
_CLUSTER_COLORS = [
    "#2563eb", "#16a34a", "#dc2626", "#d97706", "#7c3aed",
    "#0891b2", "#be185d", "#059669", "#ea580c", "#4338ca",
]


def _clr(i: int) -> str:
    return _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class LitClusterApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(_APP_TITLE)
        self.minsize(780, 560)
        self.resizable(True, True)
        self._lc: lc.LitCluster | None = None
        self._input_path: Path | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ---- top bar ---------------------------------------------------
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Input file:").grid(row=0, column=0, sticky=tk.W)
        self._path_var = tk.StringVar()
        ttk.Entry(top, textvariable=self._path_var, width=52).grid(
            row=0, column=1, padx=4
        )
        ttk.Button(top, text="Browse…", command=self._browse).grid(row=0, column=2)

        # ---- parameters ------------------------------------------------
        params = ttk.LabelFrame(self, text="Parameters", padding=8)
        params.pack(fill=tk.X, padx=8, pady=4)

        def _param(parent, label, default, col):
            ttk.Label(parent, text=label).grid(row=0, column=col, sticky=tk.W, padx=(8 if col > 0 else 0, 2))
            var = tk.IntVar(value=default)
            ttk.Spinbox(parent, textvariable=var, from_=1, to=9999, width=7).grid(row=0, column=col + 1)
            return var

        self._k_var = _param(params, "Clusters (k):", 5, 0)
        self._seed_var = _param(params, "Seed:", 42, 2)
        self._max_iter_var = _param(params, "Max iter:", 100, 4)
        self._min_freq_var = _param(params, "Min term freq:", 2, 6)

        # ---- action row ------------------------------------------------
        action = ttk.Frame(self, padding=4)
        action.pack(fill=tk.X, padx=8)

        self._run_btn = ttk.Button(
            action, text="Run Clustering", command=self._run, style="Accent.TButton"
        )
        self._run_btn.pack(side=tk.LEFT)
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(action, textvariable=self._status_var, foreground="#555").pack(
            side=tk.LEFT, padx=12
        )
        self._progress = ttk.Progressbar(action, mode="indeterminate", length=120)
        self._progress.pack(side=tk.LEFT)

        # ---- notebook (results) ----------------------------------------
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self._tab_summary = scrolledtext.ScrolledText(
            self._nb, wrap=tk.WORD, state=tk.DISABLED, font=("Courier", 10)
        )
        self._nb.add(self._tab_summary, text="Summary")

        self._tab_clusters = ttk.Frame(self._nb)
        self._nb.add(self._tab_clusters, text="Clusters")
        self._build_cluster_tab()

        self._tab_papers = ttk.Frame(self._nb)
        self._nb.add(self._tab_papers, text="Papers")
        self._build_papers_tab()

        # ---- export bar ------------------------------------------------
        export = ttk.Frame(self, padding=6)
        export.pack(fill=tk.X, padx=8, pady=(0, 6))

        ttk.Label(export, text="Export:").pack(side=tk.LEFT)
        ttk.Button(export, text="CSV", command=lambda: self._export("csv")).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(export, text="JSON", command=lambda: self._export("json")).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(export, text="HTML report", command=lambda: self._export("html")).pack(
            side=tk.LEFT, padx=4
        )

    def _build_cluster_tab(self) -> None:
        frame = self._tab_clusters

        # Left: cluster list
        left = ttk.Frame(frame)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 0), pady=4)

        ttk.Label(left, text="Clusters", font=("", 9, "bold")).pack(anchor=tk.W)
        self._cluster_listbox = tk.Listbox(left, width=30, selectmode=tk.SINGLE)
        self._cluster_listbox.pack(fill=tk.Y, expand=True)
        self._cluster_listbox.bind("<<ListboxSelect>>", self._on_cluster_select)

        sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self._cluster_listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._cluster_listbox.configure(yscrollcommand=sb.set)

        # Right: cluster details
        right = ttk.Frame(frame)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        ttk.Label(right, text="Top terms", font=("", 9, "bold")).pack(anchor=tk.W)
        self._terms_text = scrolledtext.ScrolledText(
            right, height=4, wrap=tk.WORD, state=tk.DISABLED, font=("Courier", 10)
        )
        self._terms_text.pack(fill=tk.X)

        ttk.Label(right, text="Papers", font=("", 9, "bold")).pack(
            anchor=tk.W, pady=(6, 0)
        )
        cols = ("title", "authors", "year", "venue")
        self._cluster_tree = ttk.Treeview(
            right, columns=cols, show="headings", selectmode="browse"
        )
        for col, w in zip(cols, (300, 180, 60, 160)):
            self._cluster_tree.heading(col, text=col.capitalize())
            self._cluster_tree.column(col, width=w, minwidth=40)
        self._cluster_tree.pack(fill=tk.BOTH, expand=True)

        tsb = ttk.Scrollbar(
            right, orient=tk.VERTICAL, command=self._cluster_tree.yview
        )
        tsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._cluster_tree.configure(yscrollcommand=tsb.set)

    def _build_papers_tab(self) -> None:
        frame = self._tab_papers

        # Search bar
        sbar = ttk.Frame(frame)
        sbar.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(sbar, text="Search:").pack(side=tk.LEFT)
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._filter_papers())
        ttk.Entry(sbar, textvariable=self._search_var, width=40).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(sbar, text="Clear", command=lambda: self._search_var.set("")).pack(
            side=tk.LEFT
        )

        # Papers treeview
        cols = ("cluster", "title", "authors", "year", "venue", "doi")
        self._papers_tree = ttk.Treeview(
            frame, columns=cols, show="headings", selectmode="browse"
        )
        widths = {"cluster": 60, "title": 280, "authors": 160, "year": 55, "venue": 140, "doi": 140}
        for col in cols:
            self._papers_tree.heading(col, text=col.capitalize())
            self._papers_tree.column(col, width=widths[col], minwidth=40)
        self._papers_tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        psb = ttk.Scrollbar(
            frame, orient=tk.VERTICAL, command=self._papers_tree.yview
        )
        psb.pack(side=tk.RIGHT, fill=tk.Y)
        self._papers_tree.configure(yscrollcommand=psb.set)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select input file",
            filetypes=[
                ("Supported files", "*.bib *.csv *.jsonl"),
                ("BibTeX", "*.bib"),
                ("CSV", "*.csv"),
                ("JSONL", "*.jsonl"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._path_var.set(path)
            self._input_path = Path(path)

    def _run(self) -> None:
        path_str = self._path_var.get().strip()
        if not path_str:
            messagebox.showwarning("No file", "Please select an input file first.")
            return
        self._input_path = Path(path_str)
        if not self._input_path.is_file():
            messagebox.showerror("File not found", f"Cannot find:\n{self._input_path}")
            return

        self._run_btn.configure(state=tk.DISABLED)
        self._status_var.set("Running…")
        self._progress.start(12)

        thread = threading.Thread(target=self._cluster_thread, daemon=True)
        thread.start()

    def _cluster_thread(self) -> None:
        try:
            kwargs = dict(
                k=self._k_var.get(),
                max_iter=self._max_iter_var.get(),
                seed=self._seed_var.get(),
                min_term_freq=self._min_freq_var.get(),
            )
            suffix = self._input_path.suffix.lower()
            if suffix == ".bib":
                obj = lc.LitCluster.from_bibtex(self._input_path, **kwargs)
            elif suffix == ".jsonl":
                obj = lc.LitCluster.from_jsonl(self._input_path, **kwargs)
            else:
                obj = lc.LitCluster.from_csv(self._input_path, **kwargs)

            if not obj.papers:
                raise ValueError("No papers found in the input file.")

            obj.fit()
            self._lc = obj
            self.after(0, self._on_success)

        except Exception as exc:
            self.after(0, lambda: self._on_error(str(exc)))

    def _on_success(self) -> None:
        self._progress.stop()
        self._run_btn.configure(state=tk.NORMAL)
        n = len(self._lc.papers)
        k = len(self._lc.clusters)
        self._status_var.set(f"Done — {n} papers in {k} clusters.")
        self._populate_results()

    def _on_error(self, msg: str) -> None:
        self._progress.stop()
        self._run_btn.configure(state=tk.NORMAL)
        self._status_var.set("Error.")
        messagebox.showerror("Clustering failed", msg)

    def _on_cluster_select(self, _event=None) -> None:
        sel = self._cluster_listbox.curselection()
        if not sel or self._lc is None:
            return
        idx = sel[0]
        cluster = self._lc.clusters[idx]

        # Update top-terms panel
        self._terms_text.configure(state=tk.NORMAL)
        self._terms_text.delete("1.0", tk.END)
        self._terms_text.insert(tk.END, "  ".join(cluster.top_terms))
        self._terms_text.configure(state=tk.DISABLED)

        # Update papers treeview
        for row in self._cluster_tree.get_children():
            self._cluster_tree.delete(row)
        for p in cluster.papers:
            self._cluster_tree.insert(
                "", tk.END,
                values=(p.title, p.authors, p.year, p.venue),
            )

    def _filter_papers(self) -> None:
        query = self._search_var.get().lower()
        for iid in self._papers_tree.get_children():
            vals = self._papers_tree.item(iid, "values")
            match = not query or any(query in str(v).lower() for v in vals)
            # Tkinter Treeview doesn't support hiding rows natively;
            # we detach/reattach to simulate visibility filtering.
            if not match:
                self._papers_tree.detach(iid)
            else:
                # Reattach if not already present
                try:
                    self._papers_tree.reattach(iid, "", tk.END)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Populate results
    # ------------------------------------------------------------------

    def _populate_results(self) -> None:
        if self._lc is None:
            return

        # Summary tab
        self._tab_summary.configure(state=tk.NORMAL)
        self._tab_summary.delete("1.0", tk.END)
        self._tab_summary.insert(tk.END, self._lc.summary())
        self._tab_summary.configure(state=tk.DISABLED)

        # Cluster listbox
        self._cluster_listbox.delete(0, tk.END)
        for c in self._lc.clusters:
            self._cluster_listbox.insert(
                tk.END,
                f"[{c.cluster_id}] {', '.join(c.top_terms[:2])} ({len(c.papers)})",
            )

        # Papers treeview (store all items including detached)
        self._all_paper_iids: list[str] = []
        for row in self._papers_tree.get_children():
            self._papers_tree.delete(row)
        for c in self._lc.clusters:
            for p in c.papers:
                iid = self._papers_tree.insert(
                    "", tk.END,
                    values=(c.cluster_id, p.title, p.authors, p.year, p.venue, p.doi),
                )
                self._all_paper_iids.append(iid)

        # Switch to Summary tab
        self._nb.select(0)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(self, fmt: str) -> None:
        if self._lc is None:
            messagebox.showinfo("No results", "Run clustering first.")
            return

        ext_map = {"csv": ".csv", "json": ".json", "html": ".html"}
        filetypes_map = {
            "csv": [("CSV files", "*.csv")],
            "json": [("JSON files", "*.json")],
            "html": [("HTML files", "*.html")],
        }
        default_name = (self._input_path.stem + ".clusters" + ext_map[fmt]) \
            if self._input_path else ("clusters" + ext_map[fmt])

        out = filedialog.asksaveasfilename(
            title=f"Export as {fmt.upper()}",
            defaultextension=ext_map[fmt],
            filetypes=filetypes_map[fmt],
            initialfile=default_name,
        )
        if not out:
            return
        out_path = Path(out)
        try:
            if fmt == "csv":
                self._lc.export_csv(out_path)
            elif fmt == "json":
                self._lc.export_json(out_path)
            elif fmt == "html":
                title = (
                    f"Literature Cluster Report — {self._input_path.name}"
                    if self._input_path
                    else "Literature Cluster Report"
                )
                self._lc.export_html(out_path, title=title)
            messagebox.showinfo("Export complete", f"Saved to:\n{out_path}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = LitClusterApp()
    # Centre window on screen
    app.update_idletasks()
    w, h = app.winfo_width(), app.winfo_height()
    sw, sh = app.winfo_screenwidth(), app.winfo_screenheight()
    app.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")
    app.mainloop()


if __name__ == "__main__":
    main()
