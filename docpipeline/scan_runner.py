"""
Stage 0 - Scanning: group raw TIF files into per-application folders.

Uses barcode separator pages (>= 2 barcodes on one page) to detect
application boundaries and copies each application's TIF files into
APP_01/, APP_02/... subfolders in the output directory.

JPGs are NOT copied (they are paired colour scans kept for later steps).
Separator sheets themselves are not copied anywhere.

CLI:
    scan.exe --input <raw scan folder> --output <output folder>

Double-click: opens folder-picker dialogs.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image as PILImage

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

TIF_EXTS = (".tif", ".tiff")


# ---------------------------------------------------------------- barcode scan
def _barcode_count(path: str, max_side: int = 1600) -> int:
    try:
        import zxingcpp
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            g = np.array(PILImage.open(path).convert("L"))
        s = max_side / max(g.shape[:2])
        if s < 1:
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return len(zxingcpp.read_barcodes(g))
    except Exception:
        return 0


def _find_separators(in_dir: str, tifs: list[str], workers: int) -> set[str]:
    """Return set of filenames that are separator sheets (>= 2 barcodes)."""
    def _job(f):
        return f, _barcode_count(os.path.join(in_dir, f))

    separators: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f, n in ex.map(_job, tifs):
            if n >= 2:
                separators.add(f)
    return separators


def _group(tifs: list[str], separators: set[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    cur: list[str] = []
    for f in tifs:
        if f in separators:
            if cur:
                groups.append(cur)
            cur = []
        else:
            cur.append(f)
    if cur:
        groups.append(cur)
    return groups


# ---------------------------------------------------------------- main
def run(in_dir: str, out_dir: str, workers: int | None = None) -> None:
    workers = workers or max(4, min(8, (os.cpu_count() or 4) - 1))

    all_tifs = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(TIF_EXTS))
    if not all_tifs:
        logging.error(f"No TIF files found in {in_dir}")
        return

    total_jpgs = sum(1 for f in os.listdir(in_dir)
                     if f.lower().endswith((".jpg", ".jpeg")))
    logging.info(f"{len(all_tifs)} TIF pages  |  {total_jpgs} JPGs (not copied)")
    logging.info(f"Scanning for separator sheets ...")
    t0 = time.time()

    separators = _find_separators(in_dir, all_tifs, workers)
    groups = _group(all_tifs, separators)

    logging.info(f"  {len(separators)} separators  ->  {len(groups)} applications "
                 f"(pages: {[len(g) for g in groups]})")

    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for gi, grp in enumerate(groups, 1):
        app = f"APP_{gi:02d}"
        app_dir = os.path.join(out_dir, app)
        os.makedirs(app_dir, exist_ok=True)
        for f in grp:
            shutil.copy2(os.path.join(in_dir, f), os.path.join(app_dir, f))
            rows.append([app, f])
        logging.info(f"  {app}: {len(grp)} pages copied -> {app_dir}")

    el = time.time() - t0
    total_pages = sum(len(g) for g in groups)

    with open(os.path.join(out_dir, "scan_report.csv"),
              "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["application", "file"])
        w.writerows(rows)

    with open(os.path.join(out_dir, "scan_summary.json"),
              "w", encoding="utf-8") as fh:
        json.dump({
            "input": in_dir,
            "applications": len(groups),
            "total_pages": total_pages,
            "separators": sorted(separators),
            "seconds": round(el, 1),
            "apps": [{"app": f"APP_{gi:02d}", "pages": len(g),
                      "files": g}
                     for gi, g in enumerate(groups, 1)],
        }, fh, indent=1)

    logging.info("=" * 64)
    logging.info(f"applications={len(groups)}  pages_copied={total_pages}  "
                 f"separators_skipped={len(separators)}")
    logging.info(f"time={el:.1f}s")
    logging.info(f"output: {out_dir}")
    logging.info(f"next step: run enhance.exe on  {out_dir}")
    logging.info("=" * 64)


# ---------------------------------------------------------------- GUI
def _pick_folders_gui():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError:
        print("tkinter unavailable.")
        print("Run: scan.exe --input <raw scan folder> --output <output folder>")
        try:
            input("\nPress Enter to exit...")
        except EOFError:
            pass
        raise SystemExit(1)

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "LIC Document Scanner",
        "This tool groups raw TIF files into APP_01/, APP_02/... folders.\n\n"
        "Step 1: Select the INPUT folder  (raw scanned TIF files)\n"
        "Step 2: Select the OUTPUT folder (APP_01/ etc. created here)"
    )
    in_dir = filedialog.askdirectory(title="Select INPUT folder (raw scanned TIF files)")
    if not in_dir:
        messagebox.showwarning("Cancelled", "No input folder selected.")
        raise SystemExit(0)
    out_dir = filedialog.askdirectory(title="Select OUTPUT folder (APP_01/ etc. created here)")
    if not out_dir:
        messagebox.showwarning("Cancelled", "No output folder selected.")
        raise SystemExit(0)
    root.destroy()
    return in_dir, out_dir


# ---------------------------------------------------------------- entry point
def main():
    ap = argparse.ArgumentParser(
        prog="scan",
        description="Group raw TIF scans into per-application folders by barcode separator")
    ap.add_argument("--input",  help="raw scan folder (TIF files)")
    ap.add_argument("--output", help="output folder (APP_01/ etc. created here)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    in_dir, out_dir = a.input, a.output
    if not in_dir or not out_dir:
        in_dir, out_dir = _pick_folders_gui()

    def _pause(msg):
        try:
            input(msg)
        except EOFError:
            pass

    try:
        run(in_dir, out_dir, a.workers)
    except Exception as exc:
        logging.error(f"Fatal error: {exc}", exc_info=True)
        _pause("\nPress Enter to exit...")
        raise SystemExit(1)

    _pause("\nDone! Press Enter to exit...")


if __name__ == "__main__":
    main()
