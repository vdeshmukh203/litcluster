#!/usr/bin/env python3
"""
litcluster_gui.py — Graphical interface for LitCluster.

Launch with:
    python litcluster_gui.py
Or, after installation:
    litcluster-gui

Requires Python's built-in tkinter (install via your OS package manager if
missing, e.g. ``apt install python3-tk`` on Debian/Ubuntu).
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

# Allow running directly from the repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
except ImportError:
    print(
        "tkinter is not available.  Install it via your system package manager:\n"
        "  Debian/Ubuntu : sudo apt install python3-tk\n"
        "  Fedora/RHEL   : sudo dnf install python3-tkinter\n"
        "  macOS (brew)  : brew install python-tk",
        file=sys.stderr,
    )
    sys.exit(1)

from litcluster import Cluster, LitCluster, Paper, __version__


# ---------------------------------------------------------------------------
# Cluster detail pop-up
# ---------------------------------------------------------------------------

class _ClusterDetail(tk.Toplevel):
    """Modal-ish window that lists all papers in one cluster."""

    def __init__(self, parent: tk.Misc, cluster: Cluster) -> None:
        super().__init__(parent)
        self.title(cluster.label)
        self.geometry("820x460")
        self.resizable(True, True)

        # Top bar: top-terms summary
        top = ttk.Frame(self, padding=(8, 6, 8, 4))
        top.pack(fill="x")
        ttk.Label(
            top,
            text=f"Top terms:  {', '.join(cluster.top_terms)}",
            font=("TkDefaultFont", 10, "italic"),
        ).pack(anchor="w")
        ttk.Separator(self).pack(fill="x", padx=8)

        # Papers treeview
        cols = ("paper_id", "title", "authors", "year", "venue")
        tree = ttk.Treeview(self, columns=cols, show="headings", selectmode="browse")
        for col, heading, width in [
            ("paper_id", "ID",      90),
            ("title",    "Title",  340),
            ("authors",  "Authors",190),
            ("year",     "Year",    55),
            ("venue",    "Venue",  150),
        ]:
            tree.heading(col, text=heading)
            tree.column(col, width=width, minwidth=40, anchor="w")

        for p in cluster.papers:
            tree.insert("", "end", values=(
                p.paper_id, p.title, p.authors, p.year, p.venue,
            ))

        vsb = ttk.Scrollbar(self, orient="vertical",   command=tree.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        hsb.pack(side="bottom", fill="x")
        vsb.pack(side="right",  fill="y")
        tree.pack(side="left",  fill="both", expand=True)

        # Focus this window
        self.transient(parent)
        self.lift()


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class LitClusterGUI:
    """Main application window for LitCluster."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"LitCluster {__version__}  —  Literature Clustering Tool")
        self.root.geometry("1060x740")
        self.root.minsize(820, 580)

        self._lc: LitCluster | None = None
        self._running = False
        self._sort_col = ""
        self._sort_reverse = False

        self._build_menu()
        self._build_controls()
        self._build_notebook()
        self._build_statusbar()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open file…",   accelerator="Ctrl+O", command=self._browse)
        file_menu.add_separator()
        file_menu.add_command(label="Export CSV…",  command=self._export_csv)
        file_menu.add_command(label="Export JSON…", command=self._export_json)
        file_menu.add_separator()
        file_menu.add_command(label="Exit",         command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)
        self.root.bind("<Control-o>", lambda _e: self._browse())

    def _build_controls(self) -> None:
        ctrl = ttk.Frame(self.root, padding=(10, 6, 10, 2))
        ctrl.pack(fill="x")

        # File row
        file_frame = ttk.LabelFrame(ctrl, text="Input file", padding=6)
        file_frame.pack(fill="x", pady=(0, 6))

        self._file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self._file_var, width=72).pack(
            side="left", fill="x", expand=True,
        )
        ttk.Button(file_frame, text="Browse…", command=self._browse).pack(
            side="left", padx=(6, 0),
        )

        # Parameters row
        params_frame = ttk.LabelFrame(ctrl, text="Clustering parameters", padding=6)
        params_frame.pack(fill="x", pady=(0, 6))

        self._k_var    = tk.IntVar(value=5)
        self._iter_var = tk.IntVar(value=100)
        self._seed_var = tk.IntVar(value=42)
        self._freq_var = tk.IntVar(value=2)

        for col, label, var, lo, hi in [
            (0, "Clusters (k)",    self._k_var,    1,     100),
            (1, "Max iterations",  self._iter_var, 10,   2000),
            (2, "Random seed",     self._seed_var,  0,  99999),
            (3, "Min term freq",   self._freq_var,  1,     50),
        ]:
            cell = ttk.Frame(params_frame)
            cell.grid(row=0, column=col, padx=16, sticky="w")
            ttk.Label(cell, text=label).pack(anchor="w")
            ttk.Spinbox(cell, from_=lo, to=hi, textvariable=var, width=9).pack()

        # Action row
        btn_row = ttk.Frame(ctrl)
        btn_row.pack(fill="x", pady=(0, 4))

        self._run_btn = ttk.Button(
            btn_row, text="▶  Run clustering", command=self._run,
        )
        self._run_btn.pack(side="left", padx=(0, 10))

        self._progress = ttk.Progressbar(btn_row, mode="indeterminate", length=180)
        self._progress.pack(side="left")

        ttk.Button(btn_row, text="Export JSON…", command=self._export_json).pack(
            side="right", padx=(4, 0),
        )
        ttk.Button(btn_row, text="Export CSV…",  command=self._export_csv).pack(
            side="right",
        )

    def _build_notebook(self) -> None:
        self._nb = ttk.Notebook(self.root)
        self._nb.pack(fill="both", expand=True, padx=10, pady=(0, 4))

        # --- Summary tab ---
        self._summary_text = scrolledtext.ScrolledText(
            self._nb,
            wrap=tk.WORD,
            font=("Courier", 10),
            state="disabled",
            background="#f8f8f8",
        )
        self._nb.add(self._summary_text, text="  Summary  ")

        # --- Clusters tab ---
        self._build_clusters_tab()

        # --- All papers tab ---
        self._build_papers_tab()

    def _build_clusters_tab(self) -> None:
        frame = ttk.Frame(self._nb)
        self._nb.add(frame, text="  Clusters  ")

        cols = ("id", "size", "pct", "top_terms")
        self._clusters_tree = ttk.Treeview(
            frame, columns=cols, show="headings", selectmode="browse",
        )
        for col, heading, width, anchor in [
            ("id",       "ID",        55,  "center"),
            ("size",     "Papers",    72,  "center"),
            ("pct",      "%",         62,  "center"),
            ("top_terms","Top terms", 800, "w"),
        ]:
            self._clusters_tree.heading(col, text=heading)
            self._clusters_tree.column(col, width=width, anchor=anchor, minwidth=40)

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self._clusters_tree.yview)
        self._clusters_tree.configure(yscrollcommand=vsb.set)

        self._clusters_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._clusters_tree.bind("<Double-1>", self._on_cluster_double_click)
        self._clusters_tree.bind("<Return>",   self._on_cluster_double_click)

        hint = ttk.Label(
            frame,
            text="Double-click or press Enter on a cluster to see its papers.",
            foreground="gray",
            font=("TkDefaultFont", 9, "italic"),
        )
        hint.place(relx=0.5, rely=0.995, anchor="s")

    def _build_papers_tab(self) -> None:
        frame = ttk.Frame(self._nb)
        self._nb.add(frame, text="  All papers  ")

        cols = ("cluster", "paper_id", "title", "authors", "year", "venue")
        self._papers_tree = ttk.Treeview(
            frame, columns=cols, show="headings", selectmode="browse",
        )
        for col, heading, width in [
            ("cluster",  "Cluster",  72),
            ("paper_id", "ID",       90),
            ("title",    "Title",   360),
            ("authors",  "Authors", 200),
            ("year",     "Year",     55),
            ("venue",    "Venue",   180),
        ]:
            self._papers_tree.heading(
                col, text=heading,
                command=lambda c=col: self._sort_papers(c),
            )
            self._papers_tree.column(col, width=width, minwidth=40, anchor="w")

        vsb = ttk.Scrollbar(frame, orient="vertical",   command=self._papers_tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self._papers_tree.xview)
        self._papers_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        hsb.pack(side="bottom", fill="x")
        vsb.pack(side="right",  fill="y")
        self._papers_tree.pack(side="left", fill="both", expand=True)

    def _build_statusbar(self) -> None:
        self._status_var = tk.StringVar(value="Ready — open a file and run clustering.")
        ttk.Label(
            self.root,
            textvariable=self._status_var,
            relief="sunken",
            anchor="w",
            padding=(6, 2),
        ).pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select input file",
            filetypes=[
                ("BibTeX",     "*.bib"),
                ("CSV",        "*.csv"),
                ("JSONL",      "*.jsonl"),
                ("All files",  "*.*"),
            ],
        )
        if path:
            self._file_var.set(path)
            self._set_status(f"File selected: {path}")

    def _run(self) -> None:
        if self._running:
            return
        path_str = self._file_var.get().strip()
        if not path_str:
            messagebox.showwarning("No file", "Please select an input file first.")
            return
        path = Path(path_str)
        if not path.is_file():
            messagebox.showerror("File not found", f"Cannot find:\n{path}")
            return

        self._running = True
        self._run_btn.config(state="disabled")
        self._progress.start(12)
        self._set_status("Running clustering…")

        def _worker() -> None:
            try:
                kwargs = dict(
                    k=self._k_var.get(),
                    max_iter=self._iter_var.get(),
                    seed=self._seed_var.get(),
                    min_term_freq=self._freq_var.get(),
                )
                suffix = path.suffix.lower()
                if suffix == ".bib":
                    obj = LitCluster.from_bibtex(path, **kwargs)
                elif suffix == ".jsonl":
                    obj = LitCluster.from_jsonl(path, **kwargs)
                else:
                    obj = LitCluster.from_csv(path, **kwargs)
                obj.fit()
                self.root.after(0, lambda: self._on_done(obj))
            except Exception as exc:  # noqa: BLE001
                self.root.after(0, lambda: self._on_error(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_done(self, obj: LitCluster) -> None:
        self._lc = obj
        self._progress.stop()
        self._run_btn.config(state="normal")
        self._running = False
        self._populate_results()
        n = len(obj.papers)
        k = len(obj.clusters)
        self._set_status(
            f"Done — {n} paper{'s' if n != 1 else ''} clustered into "
            f"{k} group{'s' if k != 1 else ''}.  "
            "Double-click a cluster to inspect its papers."
        )
        self._nb.select(0)

    def _on_error(self, exc: Exception) -> None:
        self._progress.stop()
        self._run_btn.config(state="normal")
        self._running = False
        self._set_status(f"Error: {exc}")
        messagebox.showerror("Clustering failed", str(exc))

    # ------------------------------------------------------------------
    # Populate results
    # ------------------------------------------------------------------

    def _populate_results(self) -> None:
        obj = self._lc
        assert obj is not None

        # Summary tab
        self._summary_text.config(state="normal")
        self._summary_text.delete("1.0", tk.END)
        self._summary_text.insert("1.0", obj.summary())
        self._summary_text.config(state="disabled")

        # Clusters tab
        self._clusters_tree.delete(*self._clusters_tree.get_children())
        n_total = max(len(obj.papers), 1)
        for c in obj.clusters:
            pct = f"{100 * len(c.papers) / n_total:.1f}"
            self._clusters_tree.insert(
                "", "end",
                iid=str(c.cluster_id),
                values=(c.cluster_id, len(c.papers), pct, ", ".join(c.top_terms)),
            )

        # Papers tab — color-code alternate clusters for readability
        self._papers_tree.delete(*self._papers_tree.get_children())
        for c in obj.clusters:
            tag = "even" if c.cluster_id % 2 == 0 else "odd"
            for p in c.papers:
                self._papers_tree.insert(
                    "", "end", tags=(tag,),
                    values=(c.cluster_id, p.paper_id, p.title,
                            p.authors, p.year, p.venue),
                )
        self._papers_tree.tag_configure("even", background="#f0f4ff")
        self._papers_tree.tag_configure("odd",  background="#ffffff")

        # Reset sort state
        self._sort_col = ""
        self._sort_reverse = False

    # ------------------------------------------------------------------
    # Cluster detail pop-up
    # ------------------------------------------------------------------

    def _on_cluster_double_click(self, _event=None) -> None:
        if not self._lc:
            return
        sel = self._clusters_tree.selection()
        if not sel:
            return
        cid = int(sel[0])
        cluster = next((c for c in self._lc.clusters if c.cluster_id == cid), None)
        if cluster:
            _ClusterDetail(self.root, cluster)

    # ------------------------------------------------------------------
    # Sortable papers table
    # ------------------------------------------------------------------

    def _sort_papers(self, col: str) -> None:
        rows = [
            (self._papers_tree.set(k, col), k)
            for k in self._papers_tree.get_children()
        ]
        reverse = self._sort_col == col and not self._sort_reverse
        rows.sort(key=lambda x: x[0].lower(), reverse=reverse)
        for idx, (_, k) in enumerate(rows):
            self._papers_tree.move(k, "", idx)
        self._sort_col = col
        self._sort_reverse = reverse
        # Update heading to show sort direction
        arrow = " ▲" if not reverse else " ▼"
        for c in ("cluster", "paper_id", "title", "authors", "year", "venue"):
            heading = c.replace("_", " ").title()
            self._papers_tree.heading(
                c, text=heading + (arrow if c == col else ""),
            )

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    def _export_csv(self) -> None:
        if not self._lc or not self._lc.clusters:
            messagebox.showwarning("No results", "Run clustering first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self._lc.export_csv(Path(path))
            self._set_status(f"Exported CSV → {path}")

    def _export_json(self) -> None:
        if not self._lc or not self._lc.clusters:
            messagebox.showwarning("No results", "Run clustering first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            self._lc.export_json(Path(path))
            self._set_status(f"Exported JSON → {path}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self._status_var.set(msg)

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About LitCluster",
            f"LitCluster  v{__version__}\n\n"
            "Semantic clustering and topic modelling\n"
            "of scientific literature.\n\n"
            "Supported formats: BibTeX (.bib), CSV, JSONL\n"
            "Algorithm: TF-IDF + k-means (pure Python)\n\n"
            "Author: Vaibhav Deshmukh\n"
            "License: MIT",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the LitCluster GUI."""
    root = tk.Tk()
    LitClusterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
