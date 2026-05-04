#!/usr/bin/env python3
"""
litcluster_gui.py — Desktop GUI for litcluster
================================================
Provides a graphical interface for loading academic paper collections,
configuring clustering parameters, inspecting results, and exporting
cluster assignments.  Uses only Python's standard library (tkinter).

Launch with::

    python litcluster_gui.py
    # or, after installation:
    litcluster-gui
"""

from __future__ import annotations

import pathlib
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Ensure litcluster is importable from the same directory as this script.
sys.path.insert(0, str(pathlib.Path(__file__).parent))

try:
    import litcluster as _lc
    from litcluster import LitCluster, __version__
except ImportError as _exc:  # pragma: no cover
    raise SystemExit(
        "Cannot import litcluster.  Make sure litcluster.py is in the same "
        "directory as litcluster_gui.py, or that the package is installed."
    ) from _exc


# ---------------------------------------------------------------------------
# Tooltip helper
# ---------------------------------------------------------------------------

class _Tooltip:
    """Simple hover tooltip for any tkinter widget."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self._widget = widget
        self._text = text
        self._tip: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None) -> None:
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tip = tk.Toplevel(self._widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(
            self._tip, text=self._text,
            background="#ffffe0", relief=tk.SOLID, borderwidth=1,
            font=("TkDefaultFont", 9), wraplength=300, justify=tk.LEFT,
            padx=4, pady=2,
        )
        lbl.pack()

    def _hide(self, _event=None) -> None:
        if self._tip:
            self._tip.destroy()
            self._tip = None


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class LitClusterApp(tk.Tk):
    """Main application window for the litcluster GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.title(f"litcluster {__version__} — Literature Clustering Tool")
        self.minsize(960, 620)
        self._lc_result: LitCluster | None = None
        self._build_menu()
        self._build_toolbar()
        self._build_params()
        self._build_results()
        self._build_statusbar()
        self.resizable(True, True)
        self._set_status("Ready.  Open a BibTeX, CSV, or JSONL file to begin.")

    # ------------------------------------------------------------------
    # Layout builders
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(
            label="Open File…", command=self._open_file, accelerator="Ctrl+O"
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Export CSV…", command=self._export_csv, accelerator="Ctrl+S"
        )
        file_menu.add_command(
            label="Export JSON…", command=self._export_json
        )
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.destroy, accelerator="Ctrl+Q")
        menubar.add_cascade(label="File", menu=file_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About litcluster", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)
        self.bind("<Control-o>", lambda _: self._open_file())
        self.bind("<Control-s>", lambda _: self._export_csv())
        self.bind("<Control-q>", lambda _: self.destroy())

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self, padding=(6, 4))
        toolbar.pack(fill=tk.X, side=tk.TOP)

        ttk.Button(
            toolbar, text="Open File…", command=self._open_file, width=12
        ).grid(row=0, column=0, padx=2)

        self._path_var = tk.StringVar(value="No file selected")
        path_lbl = ttk.Label(
            toolbar, textvariable=self._path_var,
            foreground="#555555", font=("TkDefaultFont", 9),
        )
        path_lbl.grid(row=0, column=1, padx=8, sticky=tk.W)

        # Right-align the Run button
        toolbar.columnconfigure(2, weight=1)
        ttk.Button(
            toolbar, text="▶  Run Clustering", command=self._run, width=18
        ).grid(row=0, column=3, padx=4)

    def _build_params(self) -> None:
        frame = ttk.LabelFrame(self, text="Clustering Parameters", padding=(10, 6))
        frame.pack(fill=tk.X, padx=8, pady=(0, 4))

        self._k_var = tk.IntVar(value=5)
        self._seed_var = tk.IntVar(value=42)
        self._minfreq_var = tk.IntVar(value=2)
        self._maxiter_var = tk.IntVar(value=100)

        params = [
            ("Clusters (k):", self._k_var, 2, 50,
             "Number of topic clusters to create."),
            ("Seed:", self._seed_var, 0, 9999,
             "Random seed for reproducible results."),
            ("Min Term Freq:", self._minfreq_var, 1, 20,
             "Minimum number of documents a term must appear in to enter the vocabulary."),
            ("Max Iterations:", self._maxiter_var, 10, 500,
             "Maximum number of k-means iterations."),
        ]
        for col, (label, var, from_, to, tip) in enumerate(params):
            lbl = ttk.Label(frame, text=label)
            lbl.grid(row=0, column=col * 2, padx=(8, 2), sticky=tk.E)
            spn = ttk.Spinbox(
                frame, textvariable=var, from_=from_, to=to, width=7
            )
            spn.grid(row=0, column=col * 2 + 1, padx=(0, 8))
            _Tooltip(lbl, tip)
            _Tooltip(spn, tip)

    def _build_results(self) -> None:
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # ---- Left panel: cluster list ----
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        ttk.Label(left, text="Clusters", font=("TkDefaultFont", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 2)
        )

        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._cluster_lb = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=("Courier", 10),
            selectbackground="#4a90d9",
            selectforeground="white",
            activestyle="none",
        )
        scrollbar.config(command=self._cluster_lb.yview)
        self._cluster_lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._cluster_lb.bind("<<ListboxSelect>>", self._on_cluster_select)

        # ---- Right panel: paper details ----
        right = ttk.Frame(paned)
        paned.add(right, weight=3)

        ttk.Label(
            right, text="Papers in selected cluster",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor=tk.W, pady=(0, 2))

        # Top-terms banner
        self._terms_var = tk.StringVar(value="")
        ttk.Label(
            right, textvariable=self._terms_var,
            foreground="#1a5f9e", wraplength=600,
            font=("TkDefaultFont", 9, "italic"),
        ).pack(anchor=tk.W, padx=2, pady=(0, 4))

        # Papers treeview
        tree_frame = ttk.Frame(right)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("title", "authors", "year", "venue")
        self._tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", selectmode="extended"
        )
        col_widths = {"title": 340, "authors": 160, "year": 55, "venue": 160}
        for col in columns:
            self._tree.heading(col, text=col.capitalize(),
                               command=lambda c=col: self._sort_tree(c))
            self._tree.column(col, width=col_widths[col], anchor=tk.W, stretch=True)

        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self._tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        self._tree.bind("<Double-1>", self._on_paper_double_click)

    def _build_statusbar(self) -> None:
        self._status_var = tk.StringVar(value="")
        bar = ttk.Label(
            self, textvariable=self._status_var,
            relief=tk.SUNKEN, anchor=tk.W, padding=(6, 2),
            font=("TkDefaultFont", 9),
        )
        bar.pack(fill=tk.X, side=tk.BOTTOM)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self._status_var.set(msg)

    def _clear_results(self) -> None:
        self._cluster_lb.delete(0, tk.END)
        for row in self._tree.get_children():
            self._tree.delete(row)
        self._terms_var.set("")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open paper collection",
            filetypes=[
                ("All supported", "*.bib *.csv *.jsonl"),
                ("BibTeX", "*.bib"),
                ("CSV", "*.csv"),
                ("JSON Lines", "*.jsonl"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._path_var.set(path)
            self._set_status(f"File selected: {path}")

    def _run(self) -> None:
        path_str = self._path_var.get()
        if path_str in ("No file selected", ""):
            messagebox.showwarning("No file selected", "Please open a file first.")
            return
        path = pathlib.Path(path_str)
        if not path.is_file():
            messagebox.showerror("File not found", f"Cannot find:\n{path}")
            return

        self._clear_results()
        self._set_status("Clustering… please wait.")
        self.update_idletasks()

        def _worker() -> None:
            try:
                kwargs = dict(
                    k=self._k_var.get(),
                    seed=self._seed_var.get(),
                    min_term_freq=self._minfreq_var.get(),
                    max_iter=self._maxiter_var.get(),
                )
                suffix = path.suffix.lower()
                if suffix == ".bib":
                    obj = LitCluster.from_bibtex(path, **kwargs)
                elif suffix == ".jsonl":
                    obj = LitCluster.from_jsonl(path, **kwargs)
                else:
                    obj = LitCluster.from_csv(path, **kwargs)
                obj.fit()
                self._lc_result = obj
                self.after(0, self._populate_clusters)
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._on_run_error(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _populate_clusters(self) -> None:
        if self._lc_result is None:
            return
        self._cluster_lb.delete(0, tk.END)
        for c in self._lc_result.clusters:
            terms_preview = ", ".join(c.top_terms[:4])
            self._cluster_lb.insert(
                tk.END,
                f"[{c.cluster_id:2d}]  {len(c.papers):4d} papers  {terms_preview}",
            )
        n_papers = len(self._lc_result.papers)
        n_clusters = len(self._lc_result.clusters)
        self._set_status(f"Done: {n_papers} papers → {n_clusters} clusters.")

    def _on_cluster_select(self, _event=None) -> None:
        sel = self._cluster_lb.curselection()
        if not sel or self._lc_result is None:
            return
        cluster = self._lc_result.clusters[sel[0]]

        for row in self._tree.get_children():
            self._tree.delete(row)
        for p in cluster.papers:
            self._tree.insert("", tk.END, values=(
                p.title or "(no title)",
                p.authors,
                p.year,
                p.venue,
            ), tags=(p.paper_id,))

        self._terms_var.set(
            "Top terms:  " + "  ·  ".join(cluster.top_terms)
        )

    def _on_paper_double_click(self, _event=None) -> None:
        """Show a detail pop-up for the double-clicked paper."""
        sel = self._tree.selection()
        if not sel or self._lc_result is None:
            return
        item = sel[0]
        values = self._tree.item(item, "values")
        title, authors, year, venue = values

        # Locate the Paper object to access abstract and DOI
        paper = None
        for p in self._lc_result.papers:
            if p.title == title and p.authors == authors:
                paper = p
                break

        if paper is None:
            return

        win = tk.Toplevel(self)
        win.title("Paper details")
        win.geometry("620x380")
        win.resizable(True, True)

        txt = tk.Text(win, wrap=tk.WORD, font=("TkDefaultFont", 10), padx=8, pady=8)
        sb = ttk.Scrollbar(win, orient=tk.VERTICAL, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        fields = [
            ("Title", paper.title),
            ("Authors", paper.authors),
            ("Year", paper.year),
            ("Venue", paper.venue),
            ("DOI", paper.doi),
            ("Keywords", paper.keywords),
            ("Abstract", paper.abstract),
        ]
        txt.tag_configure("field_name", font=("TkDefaultFont", 10, "bold"))
        for name, value in fields:
            if value:
                txt.insert(tk.END, f"{name}:\n", "field_name")
                txt.insert(tk.END, f"{value}\n\n")
        txt.configure(state=tk.DISABLED)

    def _sort_tree(self, col: str) -> None:
        """Sort the papers treeview by *col*."""
        data = [
            (self._tree.set(child, col), child)
            for child in self._tree.get_children("")
        ]
        data.sort(key=lambda t: t[0].lower())
        for index, (_, child) in enumerate(data):
            self._tree.move(child, "", index)

    def _export_csv(self) -> None:
        if self._lc_result is None:
            messagebox.showwarning("No results", "Run clustering before exporting.")
            return
        path = filedialog.asksaveasfilename(
            title="Export clusters as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            try:
                self._lc_result.export_csv(pathlib.Path(path))
                self._set_status(f"Exported CSV: {path}")
            except OSError as exc:
                messagebox.showerror("Export failed", str(exc))

    def _export_json(self) -> None:
        if self._lc_result is None:
            messagebox.showwarning("No results", "Run clustering before exporting.")
            return
        path = filedialog.asksaveasfilename(
            title="Export clusters as JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            try:
                self._lc_result.export_json(pathlib.Path(path))
                self._set_status(f"Exported JSON: {path}")
            except OSError as exc:
                messagebox.showerror("Export failed", str(exc))

    def _on_run_error(self, exc: Exception) -> None:
        self._set_status(f"Error: {exc}")
        messagebox.showerror(
            "Clustering failed",
            f"{type(exc).__name__}: {exc}\n\n"
            "Common causes:\n"
            "• k is larger than the number of papers\n"
            "• All papers have empty title and abstract text\n"
            "• min-freq is too high for the corpus size",
        )

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About litcluster",
            f"litcluster {__version__}\n\n"
            "Cluster academic papers by topic using TF-IDF\n"
            "vectorisation and k-means (Lloyd's algorithm).\n\n"
            "All functionality uses Python's standard library.\n\n"
            "Author: Vaibhav Deshmukh\n"
            "Licence: MIT",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the litcluster GUI."""
    app = LitClusterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
