#!/usr/bin/env python3
"""
litcluster_gui.py — Tkinter graphical front-end for litcluster.

Launch with:
    litcluster-gui          (if installed via pip)
    python litcluster_gui.py
"""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# Allow running as a standalone script without installing the package.
sys.path.insert(0, str(Path(__file__).parent))

from litcluster import LitCluster, __version__  # noqa: E402


class _App(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title(f"litcluster {__version__}")
        self.minsize(780, 540)
        self._lc: LitCluster | None = None
        self._build_menu()
        self._build_body()
        self._build_status()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        mb = tk.Menu(self)

        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="Open…",       accelerator="Ctrl+O", command=self._browse)
        fm.add_separator()
        fm.add_command(label="Export CSV…",  command=lambda: self._export("csv"))
        fm.add_command(label="Export JSON…", command=lambda: self._export("json"))
        fm.add_command(label="Export HTML…", command=lambda: self._export("html"))
        fm.add_separator()
        fm.add_command(label="Exit", command=self.quit)
        mb.add_cascade(label="File", menu=fm)

        hm = tk.Menu(mb, tearoff=0)
        hm.add_command(label="About", command=self._about)
        mb.add_cascade(label="Help", menu=hm)

        self.config(menu=mb)
        self.bind_all("<Control-o>", lambda _e: self._browse())

    def _build_body(self) -> None:
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ---- left panel ------------------------------------------------
        left = ttk.Frame(pw, padding=6)
        pw.add(left, weight=1)

        # File chooser
        ttk.Label(left, text="Input file", font=("", 9, "bold")).pack(anchor="w")
        ff = ttk.Frame(left)
        ff.pack(fill=tk.X, pady=(0, 10))
        self._file_var = tk.StringVar()
        ttk.Entry(ff, textvariable=self._file_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(ff, text="Browse…", command=self._browse).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        # Parameters
        ttk.Label(left, text="Parameters", font=("", 9, "bold")).pack(anchor="w")
        pf = ttk.Frame(left)
        pf.pack(fill=tk.X, pady=(0, 10))

        param_rows = [
            ("Clusters (k):",     "_k_var",    5,   1,   500),
            ("Seed:",             "_seed_var",  42,  0,   99999),
            ("Max iterations:",   "_iter_var",  100, 1,   9999),
            ("Min term freq:",    "_freq_var",  2,   1,   100),
        ]
        for label, attr, default, lo, hi in param_rows:
            row = ttk.Frame(pf)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=17, anchor="w").pack(side=tk.LEFT)
            var = tk.IntVar(value=default)
            setattr(self, attr, var)
            ttk.Spinbox(row, textvariable=var, from_=lo, to=hi, width=8).pack(
                side=tk.LEFT
            )

        # Run button + progress
        self._run_btn = ttk.Button(
            left, text="▶  Run Clustering", command=self._run
        )
        self._run_btn.pack(fill=tk.X, pady=(4, 2))
        self._progress = ttk.Progressbar(left, mode="indeterminate")
        self._progress.pack(fill=tk.X, pady=(0, 10))

        # Statistics
        ttk.Label(left, text="Statistics", font=("", 9, "bold")).pack(anchor="w")
        self._stats_var = tk.StringVar(value="—")
        ttk.Label(
            left, textvariable=self._stats_var,
            wraplength=200, justify="left", foreground="#444",
        ).pack(anchor="w", pady=(0, 10))

        # Export shortcuts
        ttk.Label(left, text="Export", font=("", 9, "bold")).pack(anchor="w")
        for fmt, label in [
            ("csv",  "Export CSV"),
            ("json", "Export JSON"),
            ("html", "Export HTML"),
        ]:
            ttk.Button(
                left, text=label, command=lambda f=fmt: self._export(f)
            ).pack(fill=tk.X, pady=2)

        # ---- right panel -----------------------------------------------
        right = ttk.Frame(pw, padding=6)
        pw.add(right, weight=3)

        nb = ttk.Notebook(right)
        nb.pack(fill=tk.BOTH, expand=True)
        self._nb = nb

        # Clusters tab
        cf = ttk.Frame(nb)
        nb.add(cf, text="Clusters")
        self._cluster_tree = self._scrolled_tree(
            cf,
            columns=("size", "top_terms"),
            headings={"size": ("Papers", 70), "top_terms": ("Top terms", 450)},
        )
        self._cluster_tree.bind("<<TreeviewSelect>>", self._on_cluster_select)

        # Papers tab
        pf2 = ttk.Frame(nb)
        nb.add(pf2, text="Papers")
        self._paper_tree = self._scrolled_tree(
            pf2,
            columns=("cluster", "title", "authors", "year"),
            headings={
                "cluster": ("Cluster", 70),
                "title":   ("Title",   320),
                "authors": ("Authors", 180),
                "year":    ("Year",     55),
            },
        )
        self._paper_tree.bind("<Double-1>", self._on_paper_dbl)

    def _scrolled_tree(
        self, parent, *, columns: list, headings: dict
    ) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for col, (head, width) in headings.items():
            tree.heading(col, text=head)
            tree.column(col, width=width, minwidth=40)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT,  fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(fill=tk.BOTH, expand=True)
        return tree

    def _build_status(self) -> None:
        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(
            self, textvariable=self._status_var,
            relief="sunken", anchor="w", padding=(4, 2),
        ).pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Open literature file",
            filetypes=[
                ("All supported", "*.bib *.csv *.jsonl"),
                ("BibTeX",        "*.bib"),
                ("CSV",           "*.csv"),
                ("JSON Lines",    "*.jsonl"),
                ("All files",     "*.*"),
            ],
        )
        if path:
            self._file_var.set(path)
            self._status(f"Loaded: {Path(path).name}")

    def _run(self) -> None:
        path_str = self._file_var.get().strip()
        if not path_str:
            messagebox.showwarning("No file", "Please select an input file first.")
            return
        path = Path(path_str)
        if not path.is_file():
            messagebox.showerror("File not found", f"Cannot find:\n{path}")
            return
        self._run_btn.config(state="disabled")
        self._progress.start(10)
        self._status("Clustering…")
        threading.Thread(
            target=self._do_cluster, args=(path,), daemon=True
        ).start()

    def _do_cluster(self, path: Path) -> None:
        try:
            kwargs = dict(
                k=self._k_var.get(),
                seed=self._seed_var.get(),
                max_iter=self._iter_var.get(),
                min_term_freq=self._freq_var.get(),
            )
            suffix = path.suffix.lower()
            if suffix == ".bib":
                lc = LitCluster.from_bibtex(path, **kwargs)
            elif suffix == ".jsonl":
                lc = LitCluster.from_jsonl(path, **kwargs)
            else:
                lc = LitCluster.from_csv(path, **kwargs)

            if not lc.papers:
                self.after(
                    0, lambda: messagebox.showerror("Empty", "No papers found in file.")
                )
                return
            lc.fit()
            self._lc = lc
            self.after(0, self._refresh_results)
        except Exception as exc:
            msg = str(exc)
            self.after(0, lambda: messagebox.showerror("Error", msg))
        finally:
            self.after(0, self._stop_progress)

    def _stop_progress(self) -> None:
        self._progress.stop()
        self._run_btn.config(state="normal")

    def _refresh_results(self) -> None:
        lc = self._lc
        if lc is None:
            return

        sil = lc.silhouette()
        self._stats_var.set(
            f"Papers:     {len(lc.papers)}\n"
            f"Clusters:   {len(lc.clusters)}\n"
            f"Vocab size: {len(lc._vocab)}\n"
            f"Silhouette: {sil:.3f}"
        )
        self._status(
            f"Done — {len(lc.papers)} papers in {len(lc.clusters)} clusters "
            f"(silhouette={sil:.3f})"
        )

        self._cluster_tree.delete(*self._cluster_tree.get_children())
        for c in lc.clusters:
            self._cluster_tree.insert(
                "", "end",
                iid=str(c.cluster_id),
                values=(len(c.papers), ", ".join(c.top_terms[:7])),
            )

        self._populate_papers(lc.clusters)

    def _populate_papers(self, clusters) -> None:
        self._paper_tree.delete(*self._paper_tree.get_children())
        for c in clusters:
            for p in c.papers:
                self._paper_tree.insert(
                    "", "end",
                    values=(c.cluster_id, p.title[:100], p.authors[:50], p.year),
                    tags=(str(c.cluster_id),),
                )

    def _on_cluster_select(self, _event) -> None:
        if self._lc is None:
            return
        sel = self._cluster_tree.selection()
        if not sel:
            self._populate_papers(self._lc.clusters)
            return
        cid = int(sel[0])
        filtered = [c for c in self._lc.clusters if c.cluster_id == cid]
        self._populate_papers(filtered)
        self._nb.select(1)  # switch to Papers tab

    def _on_paper_dbl(self, _event) -> None:
        """Show full paper details in a pop-up on double-click."""
        if self._lc is None:
            return
        sel = self._paper_tree.selection()
        if not sel:
            return
        vals = self._paper_tree.item(sel[0], "values")
        if not vals:
            return
        # Find the paper object by cluster_id and title match
        cid, title_prefix = int(vals[0]), vals[1]
        paper = None
        for c in self._lc.clusters:
            if c.cluster_id == cid:
                for p in c.papers:
                    if p.title[:100] == title_prefix:
                        paper = p
                        break
            if paper:
                break
        if paper is None:
            return

        dlg = tk.Toplevel(self)
        dlg.title("Paper details")
        dlg.resizable(True, True)
        dlg.minsize(500, 300)
        text = tk.Text(dlg, wrap="word", padx=10, pady=10, relief="flat")
        sb = ttk.Scrollbar(dlg, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True)

        for label, val in [
            ("Title",    paper.title),
            ("Authors",  paper.authors),
            ("Year",     paper.year),
            ("Venue",    paper.venue),
            ("DOI",      paper.doi),
            ("Keywords", paper.keywords),
            ("Abstract", paper.abstract),
        ]:
            text.insert("end", f"{label}:\n", "bold")
            text.insert("end", f"  {val or '—'}\n\n")
        text.tag_configure("bold", font=("", 9, "bold"))
        text.config(state="disabled")

        ttk.Button(dlg, text="Close", command=dlg.destroy).pack(pady=6)

    def _export(self, fmt: str) -> None:
        if self._lc is None:
            messagebox.showwarning("No results", "Run clustering first.")
            return
        ext = {"csv": ".csv", "json": ".json", "html": ".html"}[fmt]
        ftypes = {
            "csv":  [("CSV files",  "*.csv")],
            "json": [("JSON files", "*.json")],
            "html": [("HTML files", "*.html")],
        }[fmt]
        dest = filedialog.asksaveasfilename(
            title=f"Export as {fmt.upper()}",
            defaultextension=ext,
            filetypes=ftypes,
        )
        if not dest:
            return
        try:
            if fmt == "csv":
                self._lc.export_csv(dest)
            elif fmt == "json":
                self._lc.export_json(dest)
            elif fmt == "html":
                self._lc.export_html(dest)
            self._status(f"Exported {fmt.upper()} → {dest}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def _about(self) -> None:
        messagebox.showinfo(
            "About litcluster",
            f"litcluster  {__version__}\n\n"
            "Topic-based clustering of scientific literature.\n"
            "Pure Python standard library — no external dependencies.\n\n"
            "Author: Vaibhav Deshmukh\n"
            "License: MIT\n"
            "https://github.com/vdeshmukh203/litcluster",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _status(self, msg: str) -> None:
        self._status_var.set(msg)


def main() -> None:
    """Launch the litcluster GUI."""
    app = _App()
    app.mainloop()


if __name__ == "__main__":
    main()
