#!/usr/bin/env python3
"""
litcluster_gui.py — Graphical interface for litcluster.

Launch with:
    litcluster-gui
    litcluster --gui
    python litcluster_gui.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext
import tkinter as tk
import tkinter.ttk as ttk

from litcluster import LitCluster, Cluster, Paper


class _App:
    """Main application window."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("litcluster — Literature Clustering")
        self.root.minsize(960, 620)
        self._lc: LitCluster | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self._build_top_bar()
        self._build_main_pane()
        self._build_status_bar()

    def _build_top_bar(self) -> None:
        top = ttk.Frame(self.root, padding=(8, 6, 8, 4))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)

        # ── file row ──────────────────────────────────────────────────
        file_frame = ttk.LabelFrame(top, text="Input File", padding=(6, 4))
        file_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        file_frame.columnconfigure(0, weight=1)

        self._file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self._file_var).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(file_frame, text="Browse…", command=self._browse).grid(
            row=0, column=1
        )

        # ── parameters row ────────────────────────────────────────────
        params = ttk.LabelFrame(top, text="Parameters", padding=(6, 4))
        params.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        self._k_var = tk.IntVar(value=5)
        self._seed_var = tk.IntVar(value=42)
        self._max_iter_var = tk.IntVar(value=100)
        self._min_freq_var = tk.IntVar(value=2)

        param_defs = [
            ("Clusters (k):", self._k_var, 2, 50),
            ("Seed:", self._seed_var, 0, 9999),
            ("Max iterations:", self._max_iter_var, 10, 1000),
            ("Min term freq:", self._min_freq_var, 1, 20),
        ]
        for col, (label, var, lo, hi) in enumerate(param_defs):
            ttk.Label(params, text=label).grid(
                row=0, column=col * 2, sticky="e", padx=(10, 2)
            )
            ttk.Spinbox(
                params, from_=lo, to=hi, textvariable=var, width=7
            ).grid(row=0, column=col * 2 + 1, padx=(0, 4))

        # ── action buttons ────────────────────────────────────────────
        btn_row = ttk.Frame(top)
        btn_row.grid(row=2, column=0, sticky="w")

        self._run_btn = ttk.Button(
            btn_row, text="Run Clustering", command=self._run
        )
        self._run_btn.pack(side="left")

        self._export_csv_btn = ttk.Button(
            btn_row, text="Export CSV",
            command=self._export_csv, state="disabled",
        )
        self._export_csv_btn.pack(side="left", padx=(6, 0))

        self._export_json_btn = ttk.Button(
            btn_row, text="Export JSON",
            command=self._export_json, state="disabled",
        )
        self._export_json_btn.pack(side="left", padx=(6, 0))

    def _build_main_pane(self) -> None:
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

        # ── left: cluster / paper tree ────────────────────────────────
        left = ttk.Frame(paned)
        paned.add(left, weight=2)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="Clusters & Papers").grid(
            row=0, column=0, sticky="w"
        )
        self._tree = ttk.Treeview(
            left,
            columns=("size", "terms"),
            show="tree headings",
            selectmode="browse",
        )
        self._tree.heading("#0", text="Cluster / Paper")
        self._tree.heading("size", text="#")
        self._tree.heading("terms", text="Top Terms")
        self._tree.column("#0", width=220, minwidth=140)
        self._tree.column("size", width=40, anchor="center", minwidth=30)
        self._tree.column("terms", width=240, minwidth=100)

        vsb = ttk.Scrollbar(left, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        # ── right: detail pane ────────────────────────────────────────
        right = ttk.Frame(paned)
        paned.add(right, weight=3)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Details").grid(row=0, column=0, sticky="w")
        self._detail = scrolledtext.ScrolledText(
            right, wrap="word", state="disabled",
            font=("TkFixedFont", 10), relief="flat",
        )
        self._detail.grid(row=1, column=0, sticky="nsew")

    def _build_status_bar(self) -> None:
        self._status_var = tk.StringVar(
            value="Ready — select a file and click Run Clustering."
        )
        ttk.Label(
            self.root, textvariable=self._status_var,
            relief="sunken", anchor="w", padding=(4, 2),
        ).grid(row=2, column=0, sticky="ew")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Open literature file",
            filetypes=[
                ("All supported", "*.csv *.jsonl *.bib"),
                ("CSV", "*.csv"),
                ("JSONL", "*.jsonl"),
                ("BibTeX", "*.bib"),
                ("All files", "*"),
            ],
        )
        if path:
            self._file_var.set(path)

    def _run(self) -> None:
        path_str = self._file_var.get().strip()
        if not path_str:
            messagebox.showwarning("No file", "Please select an input file first.")
            return
        path = Path(path_str)
        if not path.is_file():
            messagebox.showerror("File not found", f"Cannot find:\n{path}")
            return

        self._run_btn.configure(state="disabled")
        self._export_csv_btn.configure(state="disabled")
        self._export_json_btn.configure(state="disabled")
        self._status_var.set("Clustering — please wait…")
        self._tree.delete(*self._tree.get_children())
        self._set_detail("")

        def _worker() -> None:
            try:
                kwargs = dict(
                    k=self._k_var.get(),
                    seed=self._seed_var.get(),
                    max_iter=self._max_iter_var.get(),
                    min_term_freq=self._min_freq_var.get(),
                )
                suffix = path.suffix.lower()
                if suffix == ".bib":
                    lc = LitCluster.from_bibtex(path, **kwargs)
                elif suffix == ".jsonl":
                    lc = LitCluster.from_jsonl(path, **kwargs)
                else:
                    lc = LitCluster.from_csv(path, **kwargs)
                lc.fit()
                self._lc = lc
                self.root.after(0, self._populate_results)
            except Exception as exc:  # pylint: disable=broad-except
                self.root.after(
                    0,
                    lambda: messagebox.showerror("Clustering failed", str(exc)),
                )
                self.root.after(
                    0, lambda: self._status_var.set("Error during clustering.")
                )
                self.root.after(
                    0, lambda: self._run_btn.configure(state="normal")
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _populate_results(self) -> None:
        lc = self._lc
        assert lc is not None
        self._tree.delete(*self._tree.get_children())

        for cluster in lc.clusters:
            cid_str = f"c{cluster.cluster_id}"
            self._tree.insert(
                "", "end", iid=cid_str,
                text=f"Cluster {cluster.cluster_id}",
                values=(
                    len(cluster.papers),
                    ", ".join(cluster.top_terms[:5]),
                ),
                open=False,
            )
            for paper in cluster.papers:
                short = (
                    paper.title[:58] + "…"
                    if len(paper.title) > 58
                    else paper.title
                )
                self._tree.insert(
                    cid_str, "end",
                    iid=f"p{paper.paper_id}",
                    text=short,
                    values=(1, paper.authors[:45] if paper.authors else ""),
                )

        self._run_btn.configure(state="normal")
        self._export_csv_btn.configure(state="normal")
        self._export_json_btn.configure(state="normal")

        first_line = lc.summary().splitlines()[0]
        self._status_var.set(first_line)
        self._set_detail(lc.summary())

    def _on_select(self, _event=None) -> None:
        sel = self._tree.selection()
        if not sel or self._lc is None:
            return
        iid = sel[0]

        if iid.startswith("c"):
            cid = int(iid[1:])
            cluster = next(
                (c for c in self._lc.clusters if c.cluster_id == cid), None
            )
            if cluster:
                self._set_detail(self._cluster_detail(cluster))

        elif iid.startswith("p"):
            pid = iid[1:]
            paper = next(
                (p for c in self._lc.clusters for p in c.papers
                 if p.paper_id == pid),
                None,
            )
            if paper:
                self._set_detail(self._paper_detail(paper))

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _export_csv(self) -> None:
        if self._lc is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save clusters as CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
        )
        if path:
            self._lc.export_csv(Path(path))
            self._status_var.set(f"Exported: {path}")

    def _export_json(self) -> None:
        if self._lc is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save clusters as JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*")],
        )
        if path:
            self._lc.export_json(Path(path))
            self._status_var.set(f"Exported: {path}")

    # ------------------------------------------------------------------
    # Detail text helpers
    # ------------------------------------------------------------------

    def _set_detail(self, text: str) -> None:
        self._detail.configure(state="normal")
        self._detail.delete("1.0", "end")
        if text:
            self._detail.insert("end", text)
        self._detail.configure(state="disabled")

    @staticmethod
    def _cluster_detail(cluster: Cluster) -> str:
        lines = [
            cluster.label,
            f"Papers in cluster: {len(cluster.papers)}",
            f"Top terms: {', '.join(cluster.top_terms)}",
            "",
        ]
        for p in cluster.papers:
            lines.append(f"  • {p.title}")
            if p.authors:
                lines.append(f"    Authors : {p.authors}")
            if p.year:
                lines.append(f"    Year    : {p.year}")
            if p.venue:
                lines.append(f"    Venue   : {p.venue}")
            if p.doi:
                lines.append(f"    DOI     : {p.doi}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _paper_detail(paper: Paper) -> str:
        fields = [
            ("Title",    paper.title),
            ("Authors",  paper.authors),
            ("Year",     paper.year),
            ("Venue",    paper.venue),
            ("DOI",      paper.doi),
            ("Keywords", paper.keywords),
        ]
        lines = [f"{k}: {v}" for k, v in fields if v]
        if paper.abstract:
            lines += ["", "Abstract:", paper.abstract]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    """Launch the litcluster GUI."""
    try:
        app = _App()
        app.run()
    except tk.TclError as exc:
        print(f"Cannot open GUI: {exc}", file=sys.stderr)
        print(
            "Ensure a display is available (e.g. set DISPLAY on Linux).",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
