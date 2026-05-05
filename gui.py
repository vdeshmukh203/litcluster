#!/usr/bin/env python3
"""
gui.py — Graphical interface for litcluster.

Requires Python >= 3.8 with Tkinter (included in most standard distributions).
Run standalone:  python gui.py
Or via CLI:      litcluster-gui
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    import litcluster as _lc_module
    from litcluster import LitCluster, Paper
except ImportError as _e:  # pragma: no cover
    raise SystemExit(
        "litcluster not found. "
        "Install it with: pip install litcluster\n"
        f"({_e})"
    ) from _e


_VERSION = getattr(_lc_module, "__version__", "0.1.0")
_SUPPORTED_FILETYPES = [
    ("All supported", "*.bib *.csv *.jsonl"),
    ("BibTeX", "*.bib"),
    ("CSV", "*.csv"),
    ("JSON Lines", "*.jsonl"),
    ("All files", "*"),
]


class _App(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("litcluster — Literature Clustering")
        self.geometry("1100x700")
        self.minsize(760, 480)

        self._lc: LitCluster | None = None
        self._input_path: Path | None = None

        self._build_menu()
        self._build_toolbar()
        self._build_main_area()
        self._build_statusbar()
        self._set_status("Open a BibTeX (.bib), CSV, or JSON Lines (.jsonl) file to begin.")

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        mb = tk.Menu(self)
        self.config(menu=mb)

        file_menu = tk.Menu(mb, tearoff=False)
        mb.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Open…", accelerator="Ctrl+O", command=self._browse
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Export CSV…", command=lambda: self._export("csv")
        )
        file_menu.add_command(
            label="Export JSON…", command=lambda: self._export("json")
        )
        file_menu.add_command(
            label="Export HTML Report…", command=lambda: self._export("html")
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Quit", accelerator="Ctrl+Q", command=self.destroy
        )

        help_menu = tk.Menu(mb, tearoff=False)
        mb.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About litcluster", command=self._about)

        self.bind("<Control-o>", lambda _e: self._browse())
        self.bind("<Control-q>", lambda _e: self.destroy())

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        bar = tk.Frame(self, bd=1, relief=tk.GROOVE, bg="#f0f0f0")
        bar.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(2, 0))

        # File path section
        tk.Label(bar, text="File:", bg="#f0f0f0").pack(side=tk.LEFT, padx=(6, 2))
        self._path_var = tk.StringVar(value="(none)")
        tk.Label(
            bar, textvariable=self._path_var, anchor="w", width=38,
            relief=tk.SUNKEN, bg="#ffffff", padx=3,
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(bar, text="Browse…", command=self._browse).pack(side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=4
        )

        # Parameters
        for label, var_name, default, lo, hi, width in [
            ("k:", "_k_var",        5,   1, 200,  4),
            ("Seed:", "_seed_var",  42,  0, 9999, 6),
            ("Max iter:", "_iter_var", 100, 1, 2000, 5),
            ("Min freq:", "_freq_var",  2, 1,   20, 4),
        ]:
            tk.Label(bar, text=label, bg="#f0f0f0").pack(side=tk.LEFT, padx=(0, 2))
            var = tk.IntVar(value=default)
            setattr(self, var_name, var)
            tk.Spinbox(
                bar, from_=lo, to=hi, width=width, textvariable=var,
            ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Separator(bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=4, pady=4
        )

        # Run button
        self._run_btn = tk.Button(
            bar, text="▶  Cluster", command=self._run,
            bg="#4e79a7", fg="white", padx=10, relief=tk.RAISED,
        )
        self._run_btn.pack(side=tk.LEFT, padx=6)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=4
        )

        # Export buttons
        for label, fmt in [("CSV", "csv"), ("JSON", "json"), ("HTML", "html")]:
            tk.Button(
                bar, text=f"Export {label}",
                command=lambda f=fmt: self._export(f),
            ).pack(side=tk.LEFT, padx=2)

    # ------------------------------------------------------------------
    # Main paned area
    # ------------------------------------------------------------------

    def _build_main_area(self) -> None:
        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED,
                              sashwidth=5)
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ---- Left: cluster tree ----
        left = tk.Frame(pane)
        pane.add(left, minsize=260, width=360)

        tk.Label(left, text="Clusters", font=("", 11, "bold"), anchor="w").pack(
            fill=tk.X, padx=6, pady=(4, 2)
        )

        tree_frame = tk.Frame(left)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        self._tree = ttk.Treeview(
            tree_frame, columns=("details",), show="tree headings", selectmode="browse"
        )
        self._tree.heading("#0", text="Name")
        self._tree.heading("details", text="Details")
        self._tree.column("#0", width=190, stretch=True)
        self._tree.column("details", width=150, stretch=True)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        self._tree.tag_configure("cluster", font=("", 10, "bold"))
        self._tree.tag_configure("paper", font=("", 9))
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        # ---- Right: detail panel ----
        right = tk.Frame(pane)
        pane.add(right, minsize=380)

        tk.Label(right, text="Details", font=("", 11, "bold"), anchor="w").pack(
            fill=tk.X, padx=6, pady=(4, 2)
        )

        detail_frame = tk.Frame(right)
        detail_frame.pack(fill=tk.BOTH, expand=True)

        self._detail = tk.Text(
            detail_frame, wrap=tk.WORD, state=tk.DISABLED,
            bg="#fafafa", relief=tk.FLAT, font=("", 10),
            padx=8, pady=6,
        )
        vsb2 = ttk.Scrollbar(detail_frame, orient="vertical",
                              command=self._detail.yview)
        self._detail.configure(yscrollcommand=vsb2.set)
        self._detail.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb2.pack(side=tk.RIGHT, fill=tk.Y)

        # Text tags for formatting
        self._detail.tag_configure("h1", font=("", 12, "bold"), spacing3=4)
        self._detail.tag_configure("h2", font=("", 10, "bold"), spacing3=2)
        self._detail.tag_configure("key", foreground="#555", font=("", 9))
        self._detail.tag_configure("val", font=("", 9))
        self._detail.tag_configure("abstract", font=("", 9), spacing1=4,
                                   foreground="#333")

        # Progress bar (hidden until needed)
        self._progress = ttk.Progressbar(self, mode="indeterminate")

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self, bd=1, relief=tk.SUNKEN)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self._status_var = tk.StringVar()
        tk.Label(bar, textvariable=self._status_var, anchor="w", padx=6).pack(
            fill=tk.X
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        path_str = filedialog.askopenfilename(
            title="Open literature file",
            filetypes=_SUPPORTED_FILETYPES,
        )
        if not path_str:
            return
        self._input_path = Path(path_str)
        self._path_var.set(str(self._input_path))
        self._lc = None
        self._clear_results()
        self._set_status(
            f"Loaded: {self._input_path.name} — press ▶ Cluster to run."
        )

    def _run(self) -> None:
        if self._input_path is None:
            messagebox.showwarning("No file", "Please open an input file first.")
            return
        self._run_btn.config(state=tk.DISABLED)
        self._progress.pack(fill=tk.X, padx=4, pady=2, before=self._progress.master)
        self._progress.start(12)
        self._clear_results()

        kwargs = dict(
            k=self._k_var.get(),
            seed=self._seed_var.get(),
            max_iter=self._iter_var.get(),
            min_term_freq=self._freq_var.get(),
        )

        def _worker() -> None:
            try:
                suffix = self._input_path.suffix.lower()
                if suffix == ".bib":
                    lc = LitCluster.from_bibtex(self._input_path, **kwargs)
                elif suffix == ".jsonl":
                    lc = LitCluster.from_jsonl(self._input_path, **kwargs)
                else:
                    lc = LitCluster.from_csv(self._input_path, **kwargs)
                lc.fit(
                    progress=lambda s: self.after(
                        0, lambda msg=s: self._set_status(msg)
                    )
                )
                self.after(0, lambda: self._on_fit_done(lc))
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._on_fit_error(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_fit_done(self, lc: LitCluster) -> None:
        self._lc = lc
        self._stop_progress()
        self._populate_tree()
        self._set_status(
            f"Done — {len(lc.papers)} papers clustered "
            f"into {len(lc.clusters)} groups."
        )

    def _on_fit_error(self, exc: Exception) -> None:
        self._stop_progress()
        self._set_status(f"Error: {exc}")
        messagebox.showerror("Clustering failed", str(exc))

    def _stop_progress(self) -> None:
        self._progress.stop()
        self._progress.pack_forget()
        self._run_btn.config(state=tk.NORMAL)

    # ------------------------------------------------------------------
    # Tree population
    # ------------------------------------------------------------------

    def _populate_tree(self) -> None:
        self._tree.delete(*self._tree.get_children())
        if not self._lc:
            return
        for cluster in self._lc.clusters:
            cnode = self._tree.insert(
                "", "end",
                iid=f"c_{cluster.cluster_id}",
                text=f"Cluster {cluster.cluster_id}",
                values=(
                    f"{len(cluster.papers)} papers | "
                    f"{', '.join(cluster.top_terms[:3])}",
                ),
                tags=("cluster",),
                open=True,
            )
            for paper in cluster.papers:
                self._tree.insert(
                    cnode, "end",
                    iid=f"p_{paper.paper_id}",
                    text=(paper.title[:72] or "(no title)"),
                    values=(f"{paper.year}  {paper.authors[:28]}",),
                    tags=("paper",),
                )

    # ------------------------------------------------------------------
    # Detail panel
    # ------------------------------------------------------------------

    def _on_select(self, _event=None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        item_id = sel[0]
        self._detail.config(state=tk.NORMAL)
        self._detail.delete("1.0", tk.END)

        if item_id.startswith("c_") and self._lc:
            cid = int(item_id[2:])
            cluster = next(
                (c for c in self._lc.clusters if c.cluster_id == cid), None
            )
            if cluster:
                self._detail.insert(tk.END, f"Cluster {cid}\n", "h1")
                self._detail.insert(
                    tk.END, f"{len(cluster.papers)} papers\n\n", "val"
                )
                self._detail.insert(tk.END, "Top terms\n", "h2")
                self._detail.insert(
                    tk.END,
                    "  " + "  ·  ".join(cluster.top_terms) + "\n",
                    "val",
                )

        elif item_id.startswith("p_") and self._lc:
            pid = item_id[2:]
            paper = next(
                (p for c in self._lc.clusters for p in c.papers
                 if p.paper_id == pid),
                None,
            )
            if paper:
                self._detail.insert(tk.END, (paper.title or "(no title)") + "\n", "h1")
                for key, val in [
                    ("Authors", paper.authors),
                    ("Year", paper.year),
                    ("Venue", paper.venue),
                    ("DOI", paper.doi),
                    ("ID", paper.paper_id),
                ]:
                    if val:
                        self._detail.insert(tk.END, f"{key}: ", "key")
                        self._detail.insert(tk.END, val + "\n", "val")
                if paper.abstract:
                    self._detail.insert(tk.END, "\nAbstract\n", "h2")
                    self._detail.insert(tk.END, paper.abstract + "\n", "abstract")

        self._detail.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(self, fmt: str) -> None:
        if not self._lc or not self._lc.clusters:
            messagebox.showwarning("No results", "Run clustering first.")
            return
        ext = {"csv": ".csv", "json": ".json", "html": ".html"}[fmt]
        ftypes = {
            "csv": [("CSV files", "*.csv")],
            "json": [("JSON files", "*.json")],
            "html": [("HTML files", "*.html")],
        }[fmt]
        default = (
            self._input_path.stem + ".clusters" + ext
            if self._input_path
            else f"clusters{ext}"
        )
        out = filedialog.asksaveasfilename(
            title=f"Export {fmt.upper()}",
            defaultextension=ext,
            filetypes=ftypes,
            initialfile=default,
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
                self._lc.export_html(out_path)
            self._set_status(f"Exported → {out_path.name}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export failed", str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clear_results(self) -> None:
        self._tree.delete(*self._tree.get_children())
        self._detail.config(state=tk.NORMAL)
        self._detail.delete("1.0", tk.END)
        self._detail.config(state=tk.DISABLED)

    def _set_status(self, msg: str) -> None:
        self._status_var.set(msg)

    def _about(self) -> None:
        messagebox.showinfo(
            "About litcluster",
            f"litcluster v{_VERSION}\n\n"
            "Topic-based clustering of scientific literature\n"
            "using TF-IDF + k-means (pure stdlib).\n\n"
            "© 2026 Vaibhav Deshmukh — MIT License",
        )


def run_gui() -> None:
    """Launch the litcluster graphical interface."""
    app = _App()
    app.mainloop()


if __name__ == "__main__":
    run_gui()
