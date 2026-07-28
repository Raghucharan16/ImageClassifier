"""
Stage 1 batch processor - produces a review-ready output tree.

    python process_batch.py --input <scan folder> --output <folder> [--previews]

Output layout:
    <output>/enhanced/        cleaned bitonal Group4 TIFF, ORIGINAL filename kept
                              (so sorted order == original scan order, and every
                               page maps 1:1 back to its source file)
    <output>/skipped_blank/   the ORIGINAL file of every page judged blank, so a
                              reviewer can confirm nothing real was dropped
    <output>/previews/        optional side-by-side "before | after" JPGs for
                              fast visual review
    <output>/report.csv       one row per page in scan order
    <output>/manifest.json    full machine-readable detail

Only the TIFs are processed: the scanner writes a colour JPG and a bitonal TIF
of each physical page, and the TIF is the document image (higher dpi, already
black & white). The JPGs stay untouched for the later applicant-photo step.
"""
from __future__ import annotations

import os
import csv
import json
import time
import shutil
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import enhance as E

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PREVIEW_H = 1000


def default_workers(cpu: int | None = None) -> int:
    cpu = cpu or os.cpu_count() or 4
    return max(4, min(8, cpu - 1))


def _preview(before: np.ndarray, after: np.ndarray, path: str) -> None:
    """Write a single 'before | after' image, both scaled to equal height."""
    def fit(img):
        h, w = img.shape[:2]
        s = PREVIEW_H / h
        return cv2.resize(img, (max(1, int(w * s)), PREVIEW_H),
                          interpolation=cv2.INTER_AREA)
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
        # keep the original so a reviewer can confirm it really was empty
        try:
            shutil.copy2(src, os.path.join(blank_dir, f))
        except Exception as e:
            logging.error(f"could not copy blank {f}: {e}")
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
            except Exception as e:
                logging.debug(f"preview failed for {f}: {e}")

    info["ms"] = round((time.time() - t0) * 1000)
    info["in_bytes"] = os.path.getsize(src)
    return info


def run(in_dir: str, out_root: str, workers: int | None = None,
        limit: int | None = None, previews: bool = False):
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
    logging.info(f"{len(docs)} pages to enhance ({jpgs} JPGs left for the photo step) "
                 f"| {workers} workers | previews={'on' if previews else 'off'}")
    logging.info(f"  enhanced -> {enh}")
    logging.info(f"  blanks   -> {blank}")

    jobs = [(in_dir, enh, blank, prev, i, f) for i, f in enumerate(docs, 1)]
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, info in enumerate(ex.map(_one, jobs), 1):
            results.append(info)
            if i % 100 == 0 or i == len(docs):
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
    rot = sum(1 for r in ok if r.get("rotation"))
    sizes = [r["bytes"] for r in ok if "bytes" in r]

    logging.info("=" * 64)
    logging.info(f"pages={len(results)}  enhanced={len(ok)}  skipped_blank={len(bl)}  errors={len(er)}")
    logging.info(f"time={el:.0f}s   {el/len(results)*1000:.0f} ms/page   "
                 f"{len(results)/el:.1f} pages/s")
    logging.info(f"orientation corrected on {rot} pages; "
                 f"feed holes={sum(r.get('holes_removed',0) for r in ok)}, "
                 f"dark patches={sum(r.get('patches_removed',0) for r in ok)}, "
                 f"rebuilt from JPG={sum(1 for r in ok if r.get('rebuilt_from'))}")
    if sizes:
        logging.info(f"output: bitonal G4 TIFF, mean={sum(sizes)/len(sizes)/1024:.0f} KB, "
                     f"max={max(sizes)/1024:.0f} KB")
    logging.info(f"review: {out_root}")
    logging.info("=" * 64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Enhance a scanned batch into a review-ready tree")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--previews", action="store_true",
                    help="also write side-by-side before|after JPGs for review")
    a = ap.parse_args()
    run(a.input, a.output, a.workers, a.limit, a.previews)
