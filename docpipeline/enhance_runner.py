"""
Standalone enhancement runner - entry point for enhance.exe.

Workflow
--------
1. Scans raw TIF files in the input folder.
2. Detects barcode separator pages (>= 2 barcodes = separator).
3. Groups pages into applications -> creates APP_01/, APP_02/... in output.
4. Enhances each TIF and saves it directly into its APP folder.
5. Always writes side-by-side before|after preview JPGs in output/previews/.
6. Blank pages go to output/skipped_blank/ (originals, for review).

CLI:
    enhance.exe --input <scan folder> --output <output folder>

Double-click: opens folder-picker dialogs.
"""
from __future__ import annotations

import os
import sys

if getattr(sys, "frozen", False):
    _BASE = sys._MEIPASS
    os.environ.setdefault("RAPID_ORIENTATION_CONFIG",
                          os.path.join(_BASE, "orientation_config.yaml"))
    sys.path.insert(0, _BASE)

import argparse
import csv
import json
import time
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image as PILImage

import enhance as E

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PREVIEW_H = 1000


def default_workers(cpu=None):
    cpu = cpu or os.cpu_count() or 4
    return max(4, min(8, cpu - 1))


# ---------------------------------------------------------------- barcode scan
def _barcode_count(path: str, max_side: int = 1600) -> int:
    """Return number of barcodes found on this page (fast, no OCR)."""
    try:
        import zxingcpp
    except ImportError:
        return 0
    try:
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            g = np.array(PILImage.open(path).convert("L"))
        s = max_side / max(g.shape[:2])
        if s < 1:
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return len(zxingcpp.read_barcodes(g))
    except Exception:
        return 0


def _find_groups(in_dir: str, tifs: list[str], workers: int) -> list[list[str]]:
    """Detect separator pages and split TIF list into per-application groups."""
    logging.info("Scanning for application separators (barcode pages)...")

    def _job(f):
        return f, _barcode_count(os.path.join(in_dir, f))

    separators: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f, n in ex.map(_job, tifs):
            if n >= 2:
                separators.add(f)

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

    logging.info(f"  {len(separators)} separators -> {len(groups)} applications "
                 f"(sizes: {[len(g) for g in groups]})")
    return groups


# ---------------------------------------------------------------- preview
def _make_preview(src_path: str, enhanced: np.ndarray, out_path: str) -> None:
    """Side-by-side before|after preview JPG."""
    try:
        before = E.load_bitonal(src_path)
        if before is None:
            return
        def fit(img):
            h, w = img.shape[:2]
            s = PREVIEW_H / h
            return cv2.resize(img, (max(1, int(w * s)), PREVIEW_H),
                              interpolation=cv2.INTER_AREA)
        a, b = fit(before), fit(enhanced)
        gap = np.full((PREVIEW_H, 14), 180, np.uint8)
        cv2.imwrite(out_path, np.hstack([a, gap, b]),
                    [cv2.IMWRITE_JPEG_QUALITY, 75])
    except Exception as ex:
        logging.debug(f"preview failed for {src_path}: {ex}")


# ---------------------------------------------------------------- per-page job
def _one(job):
    in_dir, app_dir, blank_dir, prev_dir, seq, f = job
    src = os.path.join(in_dir, f)
    t0 = time.time()
    img, info = E.enhance_page(src)
    info["seq"] = seq

    if img is None and info.get("status") == "blank":
        try:
            shutil.copy2(src, os.path.join(blank_dir, f))
        except Exception as ex:
            logging.error(f"could not copy blank {f}: {ex}")
    elif img is not None:
        out_name = os.path.splitext(f)[0] + ".tif"
        info["out"] = out_name
        info["app_dir"] = app_dir
        info["bytes"] = E.save_group4(img, os.path.join(app_dir, out_name))
        # always write preview: raw (left) | enhanced (right)
        _make_preview(src, img,
                      os.path.join(prev_dir, os.path.splitext(f)[0] + ".jpg"))

    info["ms"] = round((time.time() - t0) * 1000)
    info["in_bytes"] = os.path.getsize(src)
    return info


# ---------------------------------------------------------------- main runner
def run(in_dir: str, out_root: str, workers: int | None = None,
        limit: int | None = None) -> None:
    blank_dir = os.path.join(out_root, "skipped_blank")
    prev_dir  = os.path.join(out_root, "previews")
    os.makedirs(blank_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    all_tifs = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(E.DOC_EXT))
    jpgs     = sum(1 for f in os.listdir(in_dir) if f.lower().endswith(E.PHOTO_EXT))
    if limit:
        all_tifs = all_tifs[:limit]
    if not all_tifs:
        logging.error(f"No TIF pages found in {in_dir}")
        return

    workers = workers or default_workers()
    logging.info(f"{len(all_tifs)} TIF pages | {jpgs} JPGs (kept as-is) | {workers} workers")
    logging.info(f"  output  -> {out_root}")
    logging.info(f"  previews-> {prev_dir}")

    # --- group by application via barcode separators ---
    groups = _find_groups(in_dir, all_tifs, workers)

    # build file -> app_dir mapping; separator pages have no APP folder
    separators = set(all_tifs) - {f for g in groups for f in g}
    app_dirs: dict[str, str] = {}
    for gi, grp in enumerate(groups, 1):
        app_dir = os.path.join(out_root, f"APP_{gi:02d}")
        os.makedirs(app_dir, exist_ok=True)
        for f in grp:
            app_dirs[f] = app_dir

    logging.info(f"APP folders created: {[f'APP_{i:02d}' for i in range(1, len(groups)+1)]}")

    # --- enhance ---
    t0 = time.time()
    jobs = [
        (in_dir, app_dirs[f], blank_dir, prev_dir, i, f)
        for i, f in enumerate(all_tifs, 1)
        if f not in separators
    ]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, info in enumerate(ex.map(_one, jobs), 1):
            results.append(info)
            if i % 50 == 0 or i == len(jobs):
                el = time.time() - t0
                logging.info(f"  [{i}/{len(jobs)}] {el:.0f}s  {el/i*1000:.0f} ms/page")

    el = time.time() - t0
    results.sort(key=lambda r: r["seq"])

    # --- report ---
    with open(os.path.join(out_root, "report.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["seq", "source_file", "app", "status", "output_file",
                     "rotation_deg", "skew_deg", "holes_removed", "patches_removed",
                     "content_ink_pct", "in_KB", "out_KB", "ms"])
        for r in results:
            app_label = os.path.basename(r.get("app_dir", ""))
            wr.writerow([r.get("seq"), r.get("file"), app_label, r.get("status"),
                         r.get("out", ""), r.get("rotation", ""), r.get("skew", ""),
                         r.get("holes_removed", 0), r.get("patches_removed", 0),
                         round(r.get("content_ink_pct", 0), 2),
                         round(r.get("in_bytes", 0) / 1024),
                         round(r.get("bytes", 0) / 1024) if r.get("bytes") else "",
                         r.get("ms")])

    with open(os.path.join(out_root, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "input": in_dir, "output": out_root, "workers": workers,
            "applications": len(groups), "separators": sorted(separators),
            "seconds": round(el, 1),
            "apps": [{"app": f"APP_{gi:02d}", "pages": len(g)}
                     for gi, g in enumerate(groups, 1)],
            "pages": results,
        }, fh, indent=1)

    ok    = [r for r in results if r.get("status") == "ok"]
    blank = [r for r in results if r.get("status") == "blank"]
    err   = [r for r in results if r.get("status") == "error"]
    sizes = [r["bytes"] for r in ok if "bytes" in r]

    logging.info("=" * 64)
    logging.info(f"applications={len(groups)}  pages={len(results)}  "
                 f"enhanced={len(ok)}  blank={len(blank)}  errors={len(err)}")
    logging.info(f"time={el:.0f}s  {el/max(1,len(results))*1000:.0f} ms/page")
    if sizes:
        logging.info(f"output: Group4 TIFF  mean={sum(sizes)/len(sizes)/1024:.0f} KB  "
                     f"max={max(sizes)/1024:.0f} KB")
    logging.info(f"previews: {prev_dir}")
    logging.info(f"next step: run index.exe on  {out_root}")
    logging.info("=" * 64)


# ---------------------------------------------------------------- GUI picker
def _pick_folders_gui():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError:
        print("No arguments given and tkinter is unavailable.")
        print("Run from a terminal:  enhance.exe --input <folder> --output <folder>")
        try:
            input("\nPress Enter to exit...")
        except EOFError:
            pass
        raise SystemExit(1)

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "LIC Document Enhancer",
        "Step 1: Select the INPUT folder  (raw scanned TIF files)\n"
        "Step 2: Select the OUTPUT folder (APP_01/, APP_02/... will be created here)\n\n"
        "Previews (before vs after) are always written."
    )
    in_dir = filedialog.askdirectory(title="Select INPUT folder (raw scanned TIF files)")
    if not in_dir:
        messagebox.showwarning("Cancelled", "No input folder selected. Exiting.")
        raise SystemExit(0)
    out_dir = filedialog.askdirectory(title="Select OUTPUT folder (APP_01/ etc. created here)")
    if not out_dir:
        messagebox.showwarning("Cancelled", "No output folder selected. Exiting.")
        raise SystemExit(0)
    root.destroy()
    return in_dir, out_dir


# ---------------------------------------------------------------- entry point
def main():
    ap = argparse.ArgumentParser(
        prog="enhance",
        description="Enhance scanned LIC TIFs and group into APP_01/, APP_02/... folders")
    ap.add_argument("--input",  help="folder containing raw scanned TIF files")
    ap.add_argument("--output", help="output folder (APP_01/ etc. will be created here)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit",   type=int, default=None,
                    help="process only first N pages (quick test)")
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
        run(in_dir, out_dir, a.workers, a.limit)
    except Exception as exc:
        logging.error(f"Fatal error: {exc}", exc_info=True)
        _pause("\nPress Enter to exit...")
        raise SystemExit(1)

    _pause("\nDone! Press Enter to exit...")


if __name__ == "__main__":
    main()
