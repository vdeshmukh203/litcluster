#!/usr/bin/env python3
"""
litcluster_gui.py — Tkinter GUI for the litcluster literature-clustering tool.

Launch
------
    python litcluster_gui.py

Requires only the Python standard library (tkinter is included in CPython).
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from typing import Optional

# Allow running from the repository root without a full install
sys.path.insert(0, str(Path(__file__).parent))
from litcluster import LitCluster, Cluster, Paper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int = 80) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class LitClusterApp(tk.Tk):
    """Top-level application window."""

    # ---- construction ------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        self.title("LitCluster — Literature Clustering Tool")
        self.minsize(820, 560)
        self._lc: Optional[LitCluster] = None
        self._build_menu()
        self._build_ui()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open…", accelerator="Ctrl+O",
                              command=self._browse)
        file_menu.add_separator()
        file_menu.add_command(label="Export CSV…", command=self._export_csv)
        file_menu.add_command(label="Export JSON…", command=self._export_json)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.destroy)
        self.bind_all("<Control-o>", lambda _e: self._browse())

        help_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # ---- Input row -------------------------------------------------------
        input_frame = ttk.LabelFrame(self, text="Input file", padding=6)
        input_frame.grid(row=0, column=0, sticky="ew", **pad)
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Path:").grid(row=0, column=0, sticky="w")
        self._path_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self._path_var).grid(
            row=0, column=1, sticky="ew", padx=(6, 6))
        ttk.Button(input_frame, text="Browse…", command=self._browse).grid(
            row=0, column=2)

        # ---- Parameters + actions -------------------------------------------
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.grid(row=1, column=0, sticky="ew", **pad)
        ctrl_frame.columnconfigure(4, weight=1)

        param_defs = [
            ("Clusters (k):", "_k_var", 1, 999, 5),
            ("Min term freq:", "_minfreq_var", 1, 999, 2),
            ("Seed:", "_seed_var", 0, 99999, 42),
            ("Max iterations:", "_maxiter_var", 1, 9999, 100),
        ]
        for col, (label, attr, lo, hi, default) in enumerate(param_defs):
            ttk.Label(ctrl_frame, text=label).grid(
                row=0, column=col * 2, sticky="e", padx=(8, 2))
            var = tk.IntVar(value=default)
            setattr(self, attr, var)
            ttk.Spinbox(ctrl_frame, textvariable=var, from_=lo, to=hi,
                        width=7).grid(row=0, column=col * 2 + 1, sticky="w")

        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.grid(row=0, column=9, sticky="e", padx=(16, 0))
        self._run_btn = ttk.Button(btn_frame, text="▶  Run", command=self._run,
                                   width=10)
        self._run_btn.pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Export CSV", command=self._export_csv,
                   width=12).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Export JSON", command=self._export_json,
                   width=12).pack(side="left", padx=4)

        # ---- Status bar ------------------------------------------------------
        self._status_var = tk.StringVar(value="Ready. Open a CSV, JSONL, or .bib file.")
        status_bar = ttk.Label(self, textvariable=self._status_var,
                               relief="sunken", anchor="w")
        status_bar.grid(row=3, column=0, sticky="ew", padx=0, pady=(2, 0))

        # ---- Results pane ----------------------------------------------------
        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.grid(row=2, column=0, sticky="nsew", **pad)

        # Left: cluster list
        left = ttk.LabelFrame(pane, text="Clusters", padding=4)
        pane.add(left, weight=1)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self._cluster_box = tk.Listbox(left, selectmode="single",
                                       activestyle="dotbox", width=32)
        self._cluster_box.grid(row=0, column=0, sticky="nsew")
        self._cluster_box.bind("<<ListboxSelect>>", self._on_cluster_select)
        _vsb(left, self._cluster_box).grid(row=0, column=1, sticky="ns")

        # Right: papers + abstract
        right = ttk.Frame(pane)
        pane.add(right, weight=3)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(right, text="Papers in selected cluster:").grid(
            row=0, column=0, sticky="w")

        self._paper_tree = ttk.Treeview(
            right,
            columns=("title", "authors", "year"),
            show="headings",
            selectmode="browse",
        )
        col_widths = {"title": 340, "authors": 160, "year": 55}
        for col, w in col_widths.items():
            self._paper_tree.heading(col, text=col.capitalize())
            self._paper_tree.column(col, width=w, minwidth=40, stretch=(col == "title"))
        self._paper_tree.grid(row=1, column=0, sticky="nsew")
        self._paper_tree.bind("<<TreeviewSelect>>", self._on_paper_select)
        _vsb(right, self._paper_tree).grid(row=1, column=1, sticky="ns")

        ttk.Label(right, text="Abstract / keywords:").grid(
            row=2, column=0, sticky="w", pady=(4, 0))
        self._abstract_box = tk.Text(
            right, height=5, wrap="word", state="disabled",
            relief="flat", background=self.cget("background"),
        )
        self._abstract_box.grid(row=3, column=0, columnspan=2,
                                sticky="ew", pady=(2, 4))

    # ---- event handlers ----------------------------------------------------

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select literature file",
            filetypes=[
                ("All supported", "*.csv *.jsonl *.bib"),
                ("CSV", "*.csv"),
                ("JSONL", "*.jsonl"),
                ("BibTeX", "*.bib"),
                ("All files", "*"),
            ],
        )
        if path:
            self._path_var.set(path)

    def _run(self) -> None:
        path_str = self._path_var.get().strip()
        if not path_str:
            messagebox.showwarning("No file selected",
                                   "Please select an input file first.")
            return
        path = Path(path_str)
        if not path.is_file():
            messagebox.showerror("File not found",
                                 f"Cannot open:\n{path}")
            return

        self._run_btn.configure(state="disabled")
        self._set_status("Running clustering…")
        self.update_idletasks()
        threading.Thread(target=self._do_clustering, args=(path,),
                         daemon=True).start()

    def _do_clustering(self, path: Path) -> None:
        try:
            kwargs = dict(
                k=self._k_var.get(),
                min_term_freq=self._minfreq_var.get(),
                seed=self._seed_var.get(),
                max_iter=self._maxiter_var.get(),
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
            self.after(0, self._populate_results)
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Clustering error", str(exc)))
            self.after(0, lambda: self._set_status("Error — see popup."))
        finally:
            self.after(0, lambda: self._run_btn.configure(state="normal"))

    def _populate_results(self) -> None:
        lc = self._lc
        self._cluster_box.delete(0, "end")
        self._paper_tree.delete(*self._paper_tree.get_children())
        self._set_abstract("")

        for c in lc.clusters:
            terms = ", ".join(c.top_terms[:4]) if c.top_terms else "—"
            self._cluster_box.insert(
                "end",
                f"[{c.cluster_id}] {_truncate(terms, 26)} ({len(c.papers)})",
            )

        n = sum(len(c.papers) for c in lc.clusters)
        self._set_status(
            f"Clustering complete — {n} papers across {len(lc.clusters)} clusters."
        )

    def _on_cluster_select(self, _event=None) -> None:
        if not self._lc:
            return
        sel = self._cluster_box.curselection()
        if not sel:
            return
        cluster: Cluster = self._lc.clusters[sel[0]]
        self._paper_tree.delete(*self._paper_tree.get_children())
        self._set_abstract("")
        for p in cluster.papers:
            self._paper_tree.insert(
                "", "end", iid=p.paper_id,
                values=(_truncate(p.title, 70), _truncate(p.authors, 30), p.year),
                tags=(p.abstract, p.keywords),
            )

    def _on_paper_select(self, _event=None) -> None:
        sel = self._paper_tree.selection()
        if not sel:
            return
        tags = self._paper_tree.item(sel[0], "tags")
        abstract = tags[0] if tags else ""
        keywords = tags[1] if len(tags) > 1 else ""
        parts = []
        if abstract:
            parts.append(abstract)
        if keywords:
            parts.append(f"Keywords: {keywords}")
        self._set_abstract("\n\n".join(parts) if parts else "(no abstract)")

    # ---- utilities ---------------------------------------------------------

    def _set_abstract(self, text: str) -> None:
        self._abstract_box.configure(state="normal")
        self._abstract_box.delete("1.0", "end")
        if text:
            self._abstract_box.insert("end", text)
        self._abstract_box.configure(state="disabled")

    def _set_status(self, msg: str) -> None:
        self._status_var.set(msg)

    def _export_csv(self) -> None:
        if not self._lc:
            messagebox.showwarning("No results", "Run clustering first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save cluster assignments",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
        )
        if path:
            self._lc.export_csv(Path(path))
            self._set_status(f"Exported CSV → {path}")

    def _export_json(self) -> None:
        if not self._lc:
            messagebox.showwarning("No results", "Run clustering first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save cluster data",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*")],
        )
        if path:
            self._lc.export_json(Path(path))
            self._set_status(f"Exported JSON → {path}")

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About LitCluster",
            "LitCluster v0.1.0\n\n"
            "Topic-based clustering of scientific literature\n"
            "using TF-IDF + k-means (pure Python stdlib).\n\n"
            "Author: Vaibhav Deshmukh\n"
            "License: MIT",
        )


# ---------------------------------------------------------------------------
# Widget helpers
# ---------------------------------------------------------------------------

def _vsb(parent, widget) -> ttk.Scrollbar:
    sb = ttk.Scrollbar(parent, orient="vertical", command=widget.yview)
    widget.configure(yscrollcommand=sb.set)
    return sb


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = LitClusterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
