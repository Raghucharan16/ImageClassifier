"""
Entry point for the standalone index.exe.

Reads the output folder produced by enhance.exe (which contains APP_01/,
APP_02/... each holding flat enhanced TIFs), classifies every page via
OCR+rules, and combines pages of the same category into a single multi-page
TIFF inside each APP folder.

CLI:
    index.exe --apps <folder>

Double-click: opens a single folder-picker dialog.
"""
from __future__ import annotations

import multiprocessing
multiprocessing.freeze_support()   # must be before any other imports for frozen EXE

import argparse
import logging
import os
import sys

if getattr(sys, "frozen", False):
    sys.path.insert(0, sys._MEIPASS)

from index_batch import run

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _pick_folder_gui():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError:
        print("tkinter unavailable. Run from terminal: index.exe --apps <folder>")
        try:
            input("\nPress Enter to exit...")
        except EOFError:
            pass
        raise SystemExit(1)

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "LIC Document Indexer",
        "Select the OUTPUT folder from enhance.exe.\n\n"
        "It should contain APP_01/, APP_02/... subfolders.\n\n"
        "Each app folder will get proposal_forms.tif, kyc.tif, etc.\n"
        "Individual page TIFs will be removed after combining."
    )
    folder = filedialog.askdirectory(
        title="Select folder containing APP_01/, APP_02/... (enhance.exe output)")
    if not folder:
        messagebox.showwarning("Cancelled", "No folder selected. Exiting.")
        raise SystemExit(0)
    root.destroy()
    return folder


def main():
    ap = argparse.ArgumentParser(
        prog="index",
        description="Classify and index APP_XX folders into per-category multi-page TIFFs")
    ap.add_argument("--apps",
                    help="folder containing APP_01/, APP_02/... (output of enhance.exe)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    apps_dir = a.apps
    if not apps_dir:
        apps_dir = _pick_folder_gui()

    def _pause(msg):
        try:
            input(msg)
        except EOFError:
            pass

    try:
        run(apps_dir, a.workers)
    except Exception as exc:
        logging.error(f"Fatal error: {exc}", exc_info=True)
        _pause("\nPress Enter to exit...")
        raise SystemExit(1)

    _pause("\nDone! Press Enter to exit...")


if __name__ == "__main__":
    main()
