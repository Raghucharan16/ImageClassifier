"""
Entry point for the standalone index.exe.

CLI usage:
    index.exe --raw <scan folder> --enhanced <enhanced folder> --output <output folder>

Double-click usage:
    Opens folder-picker dialogs for raw scans, enhanced TIFs, and output.
"""
from __future__ import annotations

import argparse
import logging
import sys

from index_batch import run

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _pick_folders_gui():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError:
        print("No arguments given and tkinter is unavailable.")
        print("Run from a terminal:  index.exe --raw <folder> --enhanced <folder> --output <folder>")
        try:
            input("\nPress Enter to exit...")
        except EOFError:
            pass
        raise SystemExit(1)

    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo(
        "LIC Document Indexer",
        "You will be asked to select 3 folders:\n\n"
        "  1. RAW scan folder  (original TIF scans, for barcode detection)\n"
        "  2. ENHANCED folder  (output from enhance.exe)\n"
        "  3. OUTPUT folder    (where APP_01/, APP_02/... will be created)"
    )

    raw = filedialog.askdirectory(title="1 of 3 — Select RAW scan folder (original TIF scans)")
    if not raw:
        messagebox.showwarning("Cancelled", "No raw folder selected. Exiting.")
        raise SystemExit(0)

    enh = filedialog.askdirectory(title="2 of 3 — Select ENHANCED folder (output of enhance.exe)")
    if not enh:
        messagebox.showwarning("Cancelled", "No enhanced folder selected. Exiting.")
        raise SystemExit(0)

    out = filedialog.askdirectory(title="3 of 3 — Select OUTPUT folder (APP_01/ etc. will be created here)")
    if not out:
        messagebox.showwarning("Cancelled", "No output folder selected. Exiting.")
        raise SystemExit(0)

    root.destroy()
    return raw, enh, out


def main():
    ap = argparse.ArgumentParser(
        prog="index",
        description="Index enhanced LIC pages into per-application multi-page TIFFs")
    ap.add_argument("--raw",
                    help="original scan folder (TIFs, used for barcode separator detection)")
    ap.add_argument("--enhanced",
                    help="stage-1 enhanced folder (output of enhance.exe)")
    ap.add_argument("--output",
                    help="output folder (APP_01/, APP_02/... created here)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-apps", type=int, default=None,
                    help="process only first N applications (quick test)")
    a = ap.parse_args()

    raw, enh, out = a.raw, a.enhanced, a.output

    if not raw or not enh or not out:
        raw, enh, out = _pick_folders_gui()

    def _pause(msg):
        try:
            input(msg)
        except EOFError:
            pass

    try:
        run(raw, enh, out, a.workers, a.max_apps)
    except Exception as exc:
        logging.error(f"Fatal error: {exc}", exc_info=True)
        _pause("\nPress Enter to exit...")
        raise SystemExit(1)

    _pause("\nDone! Press Enter to exit...")


if __name__ == "__main__":
    main()
