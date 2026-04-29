#!/usr/bin/env python3
"""
litcluster_gui.py — Tkinter graphical front-end for litcluster.

Launch with::

    python litcluster_gui.py

or, after installation::

    litcluster-gui

The GUI lets you:

* Browse for a BibTeX (.bib), CSV, or JSONL input file.
* Tune clustering parameters (k, seed, max iterations, minimum term frequency).
* Run clustering in a background thread so the UI stays responsive.
* Inspect a plain-text summary and per-cluster paper lists.
* Export results as CSV or JSON.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

# Allow running directly from the project root without installation.
sys.path.insert(0, str(Path(__file__).parent))
import litcluster as _lc


class _SpinRow:
    """Helper that places a label + spinbox pair into a grid cell."""

    def __init__(
        self,
        parent: tk.Widget,
        label: str,
        variable: tk.Variable,
        from_: float,
        to: float,
        col: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=0, column=col * 2, padx=(10, 2), sticky="w")
        ttk.Spinbox(parent, textvariable=variable, from_=from_, to=to, width=7).grid(
            row=0, column=col * 2 + 1, padx=(0, 8), sticky="w"
        )


class LitClusterApp:
    """Main application window for the litcluster GUI.

    Parameters
    ----------
    root:
        The Tk root window.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("litcluster — Literature Clustering")
        root.minsize(740, 560)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

        self._lc_obj: _lc.LitCluster | None = None

        self._build_file_frame()
        self._build_param_frame()
        self._build_notebook()
        self._build_button_bar()
        self._build_status_bar()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_file_frame(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Input file", padding=8)
        frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Path:").grid(row=0, column=0, sticky="w")
        self._path_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self._path_var).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(frame, text="Browse…", command=self._browse).grid(row=0, column=2)

        ttk.Label(
            frame,
            text="Supported formats: BibTeX (.bib) · CSV (.csv) · JSON Lines (.jsonl)",
            foreground="grey",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 0))

    def _build_param_frame(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Clustering parameters", padding=8)
        frame.grid(row=1, column=0, sticky="ew", padx=10, pady=4)

        self._k_var = tk.IntVar(value=5)
        self._seed_var = tk.IntVar(value=42)
        self._iter_var = tk.IntVar(value=100)
        self._freq_var = tk.IntVar(value=2)

        params = [
            ("Clusters (k):", self._k_var, 2, 100),
            ("Seed:", self._seed_var, 0, 99999),
            ("Max iterations:", self._iter_var, 10, 2000),
            ("Min term freq:", self._freq_var, 1, 50),
        ]
        for col, (label, var, lo, hi) in enumerate(params):
            _SpinRow(frame, label, var, lo, hi, col)

    def _build_notebook(self) -> None:
        self._nb = ttk.Notebook(self.root)
        self._nb.grid(row=2, column=0, sticky="nsew", padx=10, pady=4)

        self._summary_text = scrolledtext.ScrolledText(
            self._nb, state="disabled", wrap="word", font=("Courier", 10)
        )
        self._nb.add(self._summary_text, text="  Summary  ")

        self._detail_text = scrolledtext.ScrolledText(
            self._nb, state="disabled", wrap="word", font=("Courier", 10)
        )
        self._nb.add(self._detail_text, text="  Cluster Details  ")

    def _build_button_bar(self) -> None:
        bar = ttk.Frame(self.root, padding=(10, 0))
        bar.grid(row=3, column=0, sticky="w", pady=(0, 4))

        self._run_btn = ttk.Button(bar, text="▶  Run Clustering", command=self._run)
        self._run_btn.pack(side="left", padx=(0, 6))

        ttk.Button(
            bar, text="Export CSV…", command=lambda: self._export("csv")
        ).pack(side="left", padx=(0, 4))

        ttk.Button(
            bar, text="Export JSON…", command=lambda: self._export("json")
        ).pack(side="left")

    def _build_status_bar(self) -> None:
        self._status_var = tk.StringVar(
            value="Ready. Select a BibTeX, CSV, or JSONL file and click Run."
        )
        self._progress = ttk.Progressbar(self.root, mode="indeterminate", length=120)
        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 6))
        status_frame.columnconfigure(0, weight=1)

        ttk.Label(
            status_frame, textvariable=self._status_var,
            relief="sunken", anchor="w", padding=(4, 2),
        ).grid(row=0, column=0, sticky="ew")

        self._progress.grid(row=0, column=1, padx=(6, 0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_text(self, widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.configure(state="disabled")

    def _set_status(self, msg: str) -> None:
        self._status_var.set(msg)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        p = filedialog.askopenfilename(
            title="Open literature file",
            filetypes=[
                ("All supported", "*.bib *.csv *.jsonl"),
                ("BibTeX", "*.bib"),
                ("CSV", "*.csv"),
                ("JSON Lines", "*.jsonl"),
                ("All files", "*.*"),
            ],
        )
        if p:
            self._path_var.set(p)

    def _run(self) -> None:
        path_str = self._path_var.get().strip()
        if not path_str:
            messagebox.showwarning("No file selected", "Please select an input file first.")
            return
        path = Path(path_str)
        if not path.is_file():
            messagebox.showerror("File not found", f"Cannot open:\n{path}")
            return

        self._run_btn.configure(state="disabled")
        self._set_status("Clustering… please wait.")
        self._set_text(self._summary_text, "Running clustering, please wait…")
        self._set_text(self._detail_text, "")
        self._lc_obj = None
        self._progress.start(12)

        def _worker() -> None:
            try:
                kwargs = dict(
                    k=self._k_var.get(),
                    seed=self._seed_var.get(),
                    max_iter=self._iter_var.get(),
                    min_term_freq=self._freq_var.get(),
                )
                suffix = path.suffix.lower()
                if suffix == ".bib":
                    obj = _lc.LitCluster.from_bibtex(path, **kwargs)
                elif suffix == ".jsonl":
                    obj = _lc.LitCluster.from_jsonl(path, **kwargs)
                else:
                    obj = _lc.LitCluster.from_csv(path, **kwargs)
                obj.fit()
                self._lc_obj = obj
                self.root.after(0, self._on_done)
            except Exception as exc:
                self.root.after(0, lambda: self._on_error(str(exc)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_done(self) -> None:
        self._progress.stop()
        obj = self._lc_obj

        # Summary tab
        self._set_text(self._summary_text, obj.summary())

        # Cluster details tab
        lines: list[str] = []
        for c in obj.clusters:
            lines.append(f"{'=' * 60}")
            lines.append(f"Cluster {c.cluster_id}  ({len(c.papers)} papers)")
            lines.append(f"Top terms: {', '.join(c.top_terms[:8])}")
            lines.append("")
            for p in c.papers:
                title = p.title or "(no title)"
                year = f" ({p.year})" if p.year else ""
                lines.append(f"  • {title}{year}")
                if p.authors:
                    lines.append(f"    {p.authors[:90]}")
                if p.doi:
                    lines.append(f"    DOI: {p.doi}")
            lines.append("")
        self._set_text(self._detail_text, "\n".join(lines))

        n_papers = len(obj.papers)
        n_clusters = len(obj.clusters)
        self._set_status(
            f"Done — {n_papers} paper{'s' if n_papers != 1 else ''} "
            f"→ {n_clusters} cluster{'s' if n_clusters != 1 else ''}."
        )
        self._run_btn.configure(state="normal")
        self._nb.select(0)

    def _on_error(self, msg: str) -> None:
        self._progress.stop()
        self._set_status(f"Error: {msg}")
        self._set_text(self._summary_text, f"An error occurred:\n\n{msg}")
        self._run_btn.configure(state="normal")
        messagebox.showerror("Clustering failed", msg)

    def _export(self, fmt: str) -> None:
        if self._lc_obj is None:
            messagebox.showwarning("No results", "Run clustering first, then export.")
            return
        ext = ".csv" if fmt == "csv" else ".json"
        filetypes = [("CSV files", "*.csv")] if fmt == "csv" else [("JSON files", "*.json")]
        dest = filedialog.asksaveasfilename(
            title="Save results",
            defaultextension=ext,
            filetypes=filetypes,
        )
        if not dest:
            return
        try:
            if fmt == "csv":
                self._lc_obj.export_csv(Path(dest))
            else:
                self._lc_obj.export_json(Path(dest))
            self._set_status(f"Exported to {dest}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the litcluster GUI application."""
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.25)
    except Exception:
        pass
    LitClusterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
