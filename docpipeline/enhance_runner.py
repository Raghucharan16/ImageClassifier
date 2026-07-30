"""
Standalone enhancement runner - entry point for enhance.exe.

Stage 2 of the pipeline: it consumes scan.exe's OUTPUT, which has already
grouped the pages into applications, so no barcode work happens here.

Input (produced by scan.exe):
    <input>/
        APP_01/   00000002.tif  00000004.tif  ...
        APP_02/   00000036.tif  ...
        scan_summary.json       <- records the original raw scan folder

Workflow
--------
1. Reads the APP_xx folders from the input directory.
2. Recovers the ORIGINAL raw scan folder from scan_summary.json -- scan.exe
   copies only TIFs, so the paired colour JPGs (needed to rebuild dark ID-card
   photocopies, and later to crop the applicant photo) are still back there.
3. Enhances each TIF into the matching APP folder under output.
4. Always writes side-by-side before|after preview JPGs in output/previews/.
5. Blank pages go to output/skipped_blank/ (originals, for review).
6. Writes manifest.json recording the RAW scan folder, which index.exe reads to
   find the JPG for the applicant-photo crop.

CLI:
    enhance.exe --input <scan.exe output folder> --output <output folder>

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

import enhance as E

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PREVIEW_H = 1000


def default_workers(cpu=None):
    cpu = cpu or os.cpu_count() or 4
    return max(4, min(8, cpu - 1))


# ---------------------------------------------------------------- scan.exe input
def _read_app_folders(in_dir: str) -> dict[str, list[str]]:
    """{APP_xx: [tif filenames]} from scan.exe's output folder."""
    apps: dict[str, list[str]] = {}
    for d in sorted(os.listdir(in_dir)):
        p = os.path.join(in_dir, d)
        if not (os.path.isdir(p) and d.startswith("APP_")):
            continue
        apps[d] = sorted(f for f in os.listdir(p) if f.lower().endswith(E.DOC_EXT))
    return apps


def _raw_scan_dir(in_dir: str) -> str | None:
    """The ORIGINAL raw scan folder, recorded by scan.exe in scan_summary.json.

    Needed because scan.exe copies only TIFs into its APP folders. The paired
    colour JPGs stay behind, and they are what rebuild_from_photo uses to
    recover an unreadable dark ID-card photocopy, and what index.exe later
    crops the applicant photo from.
    """
    p = os.path.join(in_dir, "scan_summary.json")
    if not os.path.exists(p):
        logging.warning("scan_summary.json not found -- paired JPGs unavailable, "
                        "so dark ID-card pages cannot be rebuilt and photo.jpg "
                        "will be skipped later")
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            raw = json.load(fh).get("input")
    except Exception as exc:
        logging.warning(f"could not read scan_summary.json: {exc}")
        return None
    if raw and os.path.isdir(raw):
        return raw
    logging.warning(f"raw scan folder from scan_summary.json is missing: {raw}")
    return None


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
    src_dir, app_dir, blank_dir, prev_dir, seq, f, photo_dir = job
    src = os.path.join(src_dir, f)
    t0 = time.time()
    img, info = E.enhance_page(src, photo_dir=photo_dir)
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

    # --- applications already grouped by scan.exe ---
    apps = _read_app_folders(in_dir)
    if not apps:
        logging.error(f"No APP_xx folders found in {in_dir}. "
                      f"Run scan.exe first and point --input at its output folder.")
        return

    photo_dir = _raw_scan_dir(in_dir)
    if photo_dir:
        logging.info(f"Paired JPGs -> {photo_dir}")

    total = sum(len(v) for v in apps.values())
    workers = workers or default_workers()
    logging.info(f"{len(apps)} applications | {total} TIF pages | {workers} workers")
    logging.info(f"  output  -> {out_root}")
    logging.info(f"  previews-> {prev_dir}")

    # --- build jobs, preserving the APP grouping ---
    jobs = []
    seq = 0
    for app, tifs in apps.items():
        app_dir = os.path.join(out_root, app)
        os.makedirs(app_dir, exist_ok=True)
        for f in tifs:
            if limit and seq >= limit:
                break
            seq += 1
            jobs.append((os.path.join(in_dir, app), app_dir, blank_dir,
                         prev_dir, seq, f, photo_dir))

    logging.info(f"APP folders: {list(apps)}")

    # --- enhance ---
    t0 = time.time()
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
                     "gentle", "rotation_deg", "skew_deg", "holes_removed",
                     "patches_removed", "outside_noise_removed",
                     "content_ink_pct", "in_KB", "out_KB", "ms"])
        for r in results:
            app_label = os.path.basename(r.get("app_dir", ""))
            wr.writerow([r.get("seq"), r.get("file"), app_label, r.get("status"),
                         r.get("out", ""), r.get("gentle", ""),
                         r.get("rotation", ""), r.get("skew", ""),
                         r.get("holes_removed", 0), r.get("patches_removed", 0),
                         r.get("outside_noise_removed", 0),
                         round(r.get("content_ink_pct", 0), 2),
                         round(r.get("in_bytes", 0) / 1024),
                         round(r.get("bytes", 0) / 1024) if r.get("bytes") else "",
                         r.get("ms")])

    with open(os.path.join(out_root, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({
            # "input" must be the RAW scan folder, not scan.exe's output: it is
            # where the paired colour JPGs live, and index.exe reads this field
            # to find the JPG it crops the applicant photo from.
            "input": photo_dir or in_dir,
            "scan_output": in_dir,
            "output": out_root, "workers": workers,
            "applications": len(apps),
            "seconds": round(el, 1),
            "apps": [{"app": a, "pages": len(t)} for a, t in apps.items()],
            "pages": results,
        }, fh, indent=1)

    ok     = [r for r in results if r.get("status") == "ok"]
    blank  = [r for r in results if r.get("status") == "blank"]
    err    = [r for r in results if r.get("status") == "error"]
    sizes  = [r["bytes"] for r in ok if "bytes" in r]
    gentle = sum(1 for r in ok if r.get("gentle"))

    logging.info("=" * 64)
    logging.info(f"applications={len(apps)}  pages={len(results)}  "
                 f"enhanced={len(ok)}  blank={len(blank)}  errors={len(err)}")
    logging.info(f"gentle(document pages)={gentle}  full(sparse pages)={len(ok)-gentle}")
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
        "Step 1: Select the INPUT folder  (the OUTPUT folder from scan.exe,\n"
        "        which already contains APP_01/, APP_02/... )\n"
        "Step 2: Select the OUTPUT folder (enhanced APP_01/, APP_02/... go here)\n\n"
        "Previews (before vs after) are always written."
    )
    in_dir = filedialog.askdirectory(
        title="Select INPUT folder (scan.exe output, containing APP_01/, APP_02/...)")
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
        description="Enhance the APP_xx folders produced by scan.exe")
    ap.add_argument("--input",
                    help="scan.exe output folder (contains APP_01/, APP_02/...)")
    ap.add_argument("--output", help="output folder (enhanced APP_01/ etc. go here)")
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
