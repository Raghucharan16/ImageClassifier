"""
Standalone enhancement runner - entry point for the enhance-only EXE.

Usage:
    enhance.exe --input <scan folder> --output <output folder> [--previews] [--workers N]

Output layout:
    <output>/enhanced/   one Group4 TIFF per enhanced page (blank pages skipped)
    <output>/skipped_blank/  originals of blank pages for review
    <output>/previews/   side-by-side before|after JPGs (if --previews is given)
    <output>/report.csv
    <output>/manifest.json

Only TIF files are processed; JPGs stay untouched.
"""
import os
import sys

# PyInstaller onefile: models are extracted to _MEIPASS
if getattr(sys, "frozen", False):
    _BASE = sys._MEIPASS
    # make sure the bundled rapid-orientation config is found
    os.environ.setdefault("RAPID_ORIENTATION_CONFIG",
                          os.path.join(_BASE, "orientation_config.yaml"))
    # add bundle root to path so relative imports inside enhance work
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

import enhance as E

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PREVIEW_H = 1000


def default_workers(cpu=None):
    cpu = cpu or os.cpu_count() or 4
    return max(4, min(8, cpu - 1))


def _preview(before, after, path):
    def fit(img):
        h, w = img.shape[:2]
        s = PREVIEW_H / h
        return cv2.resize(img, (max(1, int(w * s)), PREVIEW_H), interpolation=cv2.INTER_AREA)
    a, b = fit(before), fit(after)
    gap = np.full((PREVIEW_H, 12), 127, np.uint8)
    cv2.imwrite(path, np.hstack([a, gap, b]), [cv2.IMWRITE_JPEG_QUALITY, 70])


def _one(job):
    in_dir, out_dir, blank_dir, prev_dir, seq, f = job
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
        info["bytes"] = E.save_group4(img, os.path.join(out_dir, out_name))
        if prev_dir:
            try:
                before = E.load_bitonal(src)
                if before is not None:
                    _preview(before, img,
                             os.path.join(prev_dir, os.path.splitext(f)[0] + ".jpg"))
            except Exception as ex:
                logging.debug(f"preview failed for {f}: {ex}")

    info["ms"] = round((time.time() - t0) * 1000)
    info["in_bytes"] = os.path.getsize(src)
    return info


def run(in_dir, out_root, workers=None, limit=None, previews=False):
    enh = os.path.join(out_root, "enhanced")
    blank = os.path.join(out_root, "skipped_blank")
    prev = os.path.join(out_root, "previews") if previews else None
    for d in (enh, blank, prev):
        if d:
            os.makedirs(d, exist_ok=True)

    docs = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(E.DOC_EXT))
    jpgs = sum(1 for f in os.listdir(in_dir) if f.lower().endswith(E.PHOTO_EXT))
    if limit:
        docs = docs[:limit]
    if not docs:
        logging.error(f"no TIF pages found in {in_dir}")
        return

    workers = workers or default_workers()
    logging.info(f"{len(docs)} pages to enhance ({jpgs} JPGs kept as-is) | {workers} workers")
    logging.info(f"  enhanced   -> {enh}")
    logging.info(f"  blanks     -> {blank}")
    if prev:
        logging.info(f"  previews   -> {prev}")

    jobs = [(in_dir, enh, blank, prev, i, f) for i, f in enumerate(docs, 1)]
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, info in enumerate(ex.map(_one, jobs), 1):
            results.append(info)
            if i % 50 == 0 or i == len(docs):
                el = time.time() - t0
                logging.info(f"  [{i}/{len(docs)}] {el:.0f}s  {el/i*1000:.0f} ms/page")

    el = time.time() - t0
    results.sort(key=lambda r: r["seq"])

    with open(os.path.join(out_root, "report.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["seq", "source_file", "status", "output_file", "rotation_deg",
                     "skew_deg", "feed_holes_removed", "dark_patches_removed",
                     "content_ink_pct", "rebuilt_from_jpg", "in_KB", "out_KB", "ms"])
        for r in results:
            wr.writerow([r.get("seq"), r.get("file"), r.get("status"), r.get("out", ""),
                         r.get("rotation", ""), r.get("skew", ""),
                         r.get("holes_removed", 0), r.get("patches_removed", 0),
                         round(r.get("content_ink_pct", 0), 2), r.get("rebuilt_from", ""),
                         round(r.get("in_bytes", 0) / 1024),
                         round(r.get("bytes", 0) / 1024) if r.get("bytes") else "",
                         r.get("ms")])

    with open(os.path.join(out_root, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"input": in_dir, "output": out_root, "workers": workers,
                   "seconds": round(el, 1), "pages": results}, fh, indent=1)

    ok = [r for r in results if r.get("status") == "ok"]
    bl = [r for r in results if r.get("status") == "blank"]
    er = [r for r in results if r.get("status") == "error"]
    sizes = [r["bytes"] for r in ok if "bytes" in r]

    logging.info("=" * 64)
    logging.info(f"pages={len(results)}  enhanced={len(ok)}  blank={len(bl)}  errors={len(er)}")
    logging.info(f"time={el:.0f}s   {el/max(1,len(results))*1000:.0f} ms/page")
    if sizes:
        logging.info(f"output: Group4 TIFF, mean={sum(sizes)/len(sizes)/1024:.0f} KB, "
                     f"max={max(sizes)/1024:.0f} KB")
    logging.info(f"review: {out_root}")
    logging.info("=" * 64)


def _pick_folders_gui():
    """Open folder-picker dialogs when the EXE is double-clicked with no args."""
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
        "Step 1: Select the INPUT folder (scanned TIF files)\n"
        "Step 2: Select the OUTPUT folder (enhanced files will go here)"
    )

    in_dir = filedialog.askdirectory(title="Select INPUT folder (scanned TIF files)")
    if not in_dir:
        messagebox.showwarning("Cancelled", "No input folder selected. Exiting.")
        raise SystemExit(0)

    out_dir = filedialog.askdirectory(title="Select OUTPUT folder (will be created if missing)")
    if not out_dir:
        messagebox.showwarning("Cancelled", "No output folder selected. Exiting.")
        raise SystemExit(0)

    want_previews = messagebox.askyesno(
        "Previews", "Also generate before/after preview images?")

    root.destroy()
    return in_dir, out_dir, want_previews


def main():
    ap = argparse.ArgumentParser(
        prog="enhance",
        description="Enhance scanned LIC document pages (Stage 1 only, no indexing)")
    ap.add_argument("--input",
                    help="folder containing scanned TIF (and JPG) files")
    ap.add_argument("--output",
                    help="output folder (will be created if missing)")
    ap.add_argument("--previews", action="store_true",
                    help="also write before|after preview JPGs")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel workers (default: CPU count - 1, capped 4-8)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only first N pages (for a quick test)")
    a = ap.parse_args()

    in_dir = a.input
    out_dir = a.output
    previews = a.previews

    # Double-clicked with no arguments: open folder-picker dialogs
    if not in_dir or not out_dir:
        in_dir, out_dir, previews = _pick_folders_gui()

    def _pause(msg):
        try:
            input(msg)
        except EOFError:
            pass

    try:
        run(in_dir, out_dir, a.workers, a.limit, previews)
    except Exception as exc:
        logging.error(f"Fatal error: {exc}", exc_info=True)
        _pause("\nPress Enter to exit...")
        raise SystemExit(1)

    _pause("\nDone! Press Enter to exit...")


if __name__ == "__main__":
    main()
