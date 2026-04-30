#!/usr/bin/env python3
"""
litcluster_gui.py — Tkinter GUI for the litcluster tool.

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

# Resolve litcluster.py from the same directory as this file
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from litcluster import LitCluster, Cluster, Paper, __version__


class _AboutDialog(tk.Toplevel):
    """Simple modal About dialog."""

    def __init__(self, parent: tk.Tk) -> None:
        super().__init__(parent)
        self.title("About litcluster")
        self.resizable(False, False)
        self.grab_set()
        tk.Label(
            self,
            text="litcluster",
            font=("TkDefaultFont", 16, "bold"),
            pady=10,
        ).pack()
        tk.Label(
            self,
            text=(
                f"Version {__version__}\n\n"
                "Topic-based clustering of scientific literature\n"
                "using TF-IDF + k-means.\n\n"
                "Pure Python — zero external dependencies.\n"
            ),
            justify="center",
            pady=4,
        ).pack()
        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 12))
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")


class LitClusterApp(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("litcluster — Literature Clustering Tool")
        self.geometry("960x680")
        self.minsize(700, 520)

        self._lc: LitCluster | None = None
        self._build_menu()
        self._build_ui()

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        self.configure(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open…", accelerator="Ctrl+O", command=self._browse)
        file_menu.add_separator()
        file_menu.add_command(label="Export CSV…", command=self._export_csv)
        file_menu.add_command(label="Export JSON…", command=self._export_json)
        file_menu.add_command(label="Export HTML…", command=self._export_html)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", accelerator="Ctrl+Q", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=lambda: _AboutDialog(self))
        menubar.add_cascade(label="Help", menu=help_menu)

        self.bind_all("<Control-o>", lambda _e: self._browse())
        self.bind_all("<Control-q>", lambda _e: self.quit())

    # ------------------------------------------------------------------
    # Main UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ---- Input / parameters frame ----
        top = ttk.LabelFrame(self, text="Input & Parameters", padding=8)
        top.pack(fill="x", padx=10, pady=(8, 4))

        # File row
        ttk.Label(top, text="File:").grid(row=0, column=0, sticky="w")
        self._file_var = tk.StringVar()
        ttk.Entry(top, textvariable=self._file_var, width=60).grid(
            row=0, column=1, columnspan=4, padx=(4, 4), sticky="ew"
        )
        ttk.Button(top, text="Browse…", command=self._browse).grid(
            row=0, column=5, padx=(0, 4)
        )
        top.columnconfigure(1, weight=1)

        # Parameters row
        params: list[tuple[str, str, int, int, int]] = [
            ("k (clusters):", "_k_var", 5, 2, 99),
            ("Seed:", "_seed_var", 42, 0, 9999),
            ("Max iter:", "_maxiter_var", 100, 10, 9999),
            ("Min freq:", "_minfreq_var", 2, 1, 50),
        ]
        for col, (lbl, attr, default, lo, hi) in enumerate(params):
            ttk.Label(top, text=lbl).grid(
                row=1, column=col, sticky="e", padx=(8 if col else 0, 2), pady=(6, 0)
            )
            var = tk.IntVar(value=default)
            setattr(self, attr, var)
            ttk.Spinbox(top, from_=lo, to=hi, textvariable=var, width=7).grid(
                row=1, column=col, sticky="w", padx=(70, 0), pady=(6, 0)
            )

        self._run_btn = ttk.Button(
            top, text="▶  Run Clustering", command=self._run
        )
        self._run_btn.grid(row=1, column=5, pady=(6, 0))

        # ---- Paned area: tree + detail ----
        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=10, pady=4)

        # Left: cluster / paper tree
        left = ttk.LabelFrame(pane, text="Clusters & Papers", padding=4)
        pane.add(left, weight=3)

        self._tree = ttk.Treeview(
            left,
            columns=("info",),
            show="tree headings",
            selectmode="browse",
        )
        self._tree.heading("#0", text="Cluster / Paper title")
        self._tree.heading("info", text="Year / Size")
        self._tree.column("#0", width=320, minwidth=180)
        self._tree.column("info", width=90, minwidth=60, anchor="center")
        ysb = ttk.Scrollbar(left, orient="vertical", command=self._tree.yview)
        xsb = ttk.Scrollbar(left, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        # Right: detail panel
        right = ttk.LabelFrame(pane, text="Paper Details", padding=6)
        pane.add(right, weight=2)

        self._detail = tk.Text(
            right,
            wrap="word",
            state="disabled",
            font=("TkDefaultFont", 9),
            bg="#fdfdfd",
            relief="flat",
            cursor="arrow",
        )
        det_sb = ttk.Scrollbar(right, orient="vertical", command=self._detail.yview)
        self._detail.configure(yscrollcommand=det_sb.set)
        self._detail.grid(row=0, column=0, sticky="nsew")
        det_sb.grid(row=0, column=1, sticky="ns")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        # Configure text tags for detail panel
        self._detail.tag_configure("heading", font=("TkDefaultFont", 10, "bold"))
        self._detail.tag_configure("label", foreground="#555555")
        self._detail.tag_configure("value", foreground="#111111")

        # ---- Bottom: export buttons + status bar ----
        bot = ttk.Frame(self)
        bot.pack(fill="x", padx=10, pady=(0, 8))

        ttk.Button(bot, text="Export CSV…", command=self._export_csv).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(bot, text="Export JSON…", command=self._export_json).pack(
            side="left", padx=4
        )
        ttk.Button(bot, text="Export HTML…", command=self._export_html).pack(
            side="left", padx=4
        )

        self._status_var = tk.StringVar(value="Ready. Open a file to begin.")
        ttk.Label(
            bot, textvariable=self._status_var, foreground="#555", anchor="e"
        ).pack(side="right", padx=6)

        self._progress = ttk.Progressbar(bot, mode="indeterminate", length=80)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Open literature file",
            filetypes=[
                ("All supported", "*.csv *.jsonl *.bib"),
                ("CSV files", "*.csv"),
                ("JSONL files", "*.jsonl"),
                ("BibTeX files", "*.bib"),
                ("All files", "*"),
            ],
        )
        if path:
            self._file_var.set(path)

    def _run(self) -> None:
        path_str = self._file_var.get().strip()
        if not path_str:
            messagebox.showerror("No file selected", "Please select an input file first.")
            return
        path = pathlib.Path(path_str)
        if not path.is_file():
            messagebox.showerror("File not found", f"Cannot find:\n{path}")
            return

        self._run_btn.configure(state="disabled")
        self._status("Clustering…")
        self._progress.pack(side="left", padx=8)
        self._progress.start(12)
        self._tree.delete(*self._tree.get_children())
        self._clear_detail()

        def _worker() -> None:
            try:
                suffix = path.suffix.lower()
                kwargs: dict = dict(
                    k=self._k_var.get(),
                    seed=self._seed_var.get(),
                    max_iter=self._maxiter_var.get(),
                    min_term_freq=self._minfreq_var.get(),
                )
                if suffix == ".bib":
                    lc = LitCluster.from_bibtex(path, **kwargs)
                elif suffix == ".jsonl":
                    lc = LitCluster.from_jsonl(path, **kwargs)
                else:
                    lc = LitCluster.from_csv(path, **kwargs)
                lc.fit()
                self._lc = lc
                self.after(0, self._populate_tree)
            except Exception as exc:
                msg = str(exc)
                self.after(0, lambda: (
                    messagebox.showerror("Clustering failed", msg),
                    self._status(f"Error: {msg}"),
                    self._stop_progress(),
                    self._run_btn.configure(state="normal"),
                ))

        threading.Thread(target=_worker, daemon=True).start()

    def _stop_progress(self) -> None:
        self._progress.stop()
        self._progress.pack_forget()

    def _populate_tree(self) -> None:
        lc = self._lc
        self._tree.delete(*self._tree.get_children())
        for c in lc.clusters:
            cid = self._tree.insert(
                "",
                "end",
                iid=f"c{c.cluster_id}",
                text=c.label,
                values=(f"{len(c.papers)} papers",),
                open=False,
                tags=("cluster",),
            )
            for p in c.papers:
                # Use a safe iid: prefix with 'p' and cluster id to avoid collisions
                iid = f"p{c.cluster_id}_{p.paper_id}"
                self._tree.insert(
                    cid,
                    "end",
                    iid=iid,
                    text=(p.title or "(no title)")[:100],
                    values=(p.year or "",),
                    tags=("paper",),
                )
        self._tree.tag_configure("cluster", font=("TkDefaultFont", 9, "bold"))
        self._tree.tag_configure("paper", font=("TkDefaultFont", 9))
        self._stop_progress()
        self._status(
            f"Done — {len(lc.papers)} papers in {len(lc.clusters)} clusters."
        )
        self._run_btn.configure(state="normal")

    def _on_select(self, _event=None) -> None:
        sel = self._tree.selection()
        if not sel or not self._lc:
            return
        iid = sel[0]
        if iid.startswith("c"):
            try:
                cid = int(iid[1:])
            except ValueError:
                return
            cluster = next(
                (c for c in self._lc.clusters if c.cluster_id == cid), None
            )
            if cluster:
                self._show_cluster(cluster)
        elif iid.startswith("p"):
            # iid format: p{cluster_id}_{paper_id}
            rest = iid[1:]
            parts = rest.split("_", 1)
            if len(parts) == 2:
                paper_id = parts[1]
                paper = next(
                    (p for p in self._lc.papers if p.paper_id == paper_id), None
                )
                if paper:
                    self._show_paper(paper)

    # ------------------------------------------------------------------
    # Detail panel helpers
    # ------------------------------------------------------------------

    def _clear_detail(self) -> None:
        self._detail.configure(state="normal")
        self._detail.delete("1.0", "end")
        self._detail.configure(state="disabled")

    def _show_paper(self, paper: Paper) -> None:
        self._detail.configure(state="normal")
        self._detail.delete("1.0", "end")
        fields = [
            ("Title", paper.title or "—"),
            ("Authors", paper.authors or "—"),
            ("Year", paper.year or "—"),
            ("Venue", paper.venue or "—"),
            ("DOI", paper.doi or "—"),
            ("Keywords", paper.keywords or "—"),
        ]
        for label, value in fields:
            self._detail.insert("end", f"{label}:  ", "label")
            self._detail.insert("end", f"{value}\n", "value")
        self._detail.insert("end", "\nAbstract\n", "heading")
        self._detail.insert("end", paper.abstract or "(no abstract)", "value")
        self._detail.configure(state="disabled")

    def _show_cluster(self, cluster: Cluster) -> None:
        self._detail.configure(state="normal")
        self._detail.delete("1.0", "end")
        self._detail.insert("end", f"{cluster.label}\n", "heading")
        self._detail.insert("end", f"\nPapers:    ", "label")
        self._detail.insert("end", f"{len(cluster.papers)}\n", "value")
        self._detail.insert("end", "Top terms: ", "label")
        self._detail.insert("end", ", ".join(cluster.top_terms) + "\n", "value")
        self._detail.configure(state="disabled")

    def _status(self, msg: str) -> None:
        self._status_var.set(msg)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _check_lc(self) -> bool:
        if not self._lc or not self._lc.clusters:
            messagebox.showinfo("No results", "Run clustering first.")
            return False
        return True

    def _export_csv(self) -> None:
        if not self._check_lc():
            return
        path = filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*")],
        )
        if path:
            try:
                self._lc.export_csv(pathlib.Path(path))
                self._status(f"Saved: {path}")
            except Exception as exc:
                messagebox.showerror("Export failed", str(exc))

    def _export_json(self) -> None:
        if not self._check_lc():
            return
        path = filedialog.asksaveasfilename(
            title="Save JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*")],
        )
        if path:
            try:
                self._lc.export_json(pathlib.Path(path))
                self._status(f"Saved: {path}")
            except Exception as exc:
                messagebox.showerror("Export failed", str(exc))

    def _export_html(self) -> None:
        if not self._check_lc():
            return
        path = filedialog.asksaveasfilename(
            title="Save interactive HTML report",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*")],
        )
        if path:
            try:
                self._lc.export_html(pathlib.Path(path))
                self._status(f"Saved: {path}")
            except Exception as exc:
                messagebox.showerror("Export failed", str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the litcluster GUI application."""
    app = LitClusterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
