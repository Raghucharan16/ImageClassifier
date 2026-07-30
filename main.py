"""
LIC document pipeline - single entry point for all three stages.

    raw scan folder            (TIFs + paired colour JPGs from the scanner)
        |
        |  1. SCAN       barcode separator sheets split the flat run of pages
        v                into per-application groups
    <out>/scanned/             APP_01/ APP_02/ ... (TIFs only)
        |
        |  2. ENHANCE    de-skew, de-speckle, punch-hole and black-line removal,
        v                auto-orientation, edge crop
    <out>/enhanced/            APP_01/ APP_02/ ... (clean Group4 TIFs)
        |
        |  3. INDEX      OCR + rules classify every page, then pages of the same
        v                class are merged into one multi-page TIFF per class
    <out>/indexed/             APP_01/proposal_forms.tif, kyc.tif, enclosure.tif,
                               review_slips.tif, medical_report.tif,
                               signature.tif, photo.jpg

Only TIFs are processed. The colour JPGs are touched for exactly one purpose:
cropping the applicant's photograph, which becomes page 1 of proposal_forms.tif.

Usage
-----
    python main.py --input "<raw scan folder>"
    python main.py --input <raw> --output <work folder>
    python main.py --input <raw> --apps 2        # first 2 applications only
    python main.py --input <raw> --stage index   # re-run one stage

Timings for each stage are reported at the end.
"""
from __future__ import annotations

import multiprocessing
multiprocessing.freeze_support()   # before other imports, for frozen EXEs

import argparse
import logging
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.join(_HERE, "docpipeline")
for _p in (_PIPE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

STAGES = ("scan", "enhance", "index")


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {seconds % 60:04.1f}s"


def _count_pages(root: str) -> int:
    """TIFs inside APP_xx folders under root."""
    n = 0
    if not os.path.isdir(root):
        return 0
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p) and d.startswith("APP_"):
            n += sum(1 for f in os.listdir(p) if f.lower().endswith((".tif", ".tiff")))
    return n


def _keep_first_apps(root: str, keep: int) -> int:
    """Delete all but the first `keep` APP_xx folders. Returns how many remain.

    Used by --apps to time the pipeline on a couple of applications instead of a
    whole batch: indexing is OCR-bound at seconds per page, so a 500-page batch
    takes minutes, while two applications give the same per-page figure in well
    under one.
    """
    apps = sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and d.startswith("APP_"))
    for d in apps[keep:]:
        shutil.rmtree(os.path.join(root, d), ignore_errors=True)
    return min(keep, len(apps))


# ------------------------------------------------------------------ stages
def stage_scan(raw_dir: str, scanned_dir: str, workers: int | None) -> float:
    import scan_runner
    logging.info("=" * 70)
    logging.info("STAGE 1/3  SCAN      group pages into applications by barcode")
    logging.info("=" * 70)
    t0 = time.time()
    scan_runner.run(raw_dir, scanned_dir, workers)
    return time.time() - t0


def stage_enhance(scanned_dir: str, enhanced_dir: str, workers: int | None) -> float:
    import enhance_runner
    logging.info("=" * 70)
    logging.info("STAGE 2/3  ENHANCE   clean, orient and crop every page")
    logging.info("=" * 70)
    t0 = time.time()
    enhance_runner.run(scanned_dir, enhanced_dir, workers)
    return time.time() - t0


def stage_index(enhanced_dir: str, indexed_dir: str, workers: int | None) -> float:
    import index_batch
    logging.info("=" * 70)
    logging.info("STAGE 3/3  INDEX     classify pages and merge per category")
    logging.info("=" * 70)
    # index_batch works IN PLACE (it replaces the individual page TIFs with the
    # merged per-category files), so the enhanced output is copied first. That
    # keeps enhanced/ reusable for a re-run of indexing alone, which matters
    # because indexing is by far the slowest stage to repeat.
    if os.path.abspath(indexed_dir) != os.path.abspath(enhanced_dir):
        if os.path.isdir(indexed_dir):
            shutil.rmtree(indexed_dir, ignore_errors=True)
        shutil.copytree(enhanced_dir, indexed_dir)
    t0 = time.time()
    index_batch.run(indexed_dir, workers)
    return time.time() - t0


# ------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        prog="main",
        description="Run the full LIC document pipeline: scan -> enhance -> index")
    ap.add_argument("--input", required=True,
                    help="raw scan folder (TIFs + paired colour JPGs)")
    ap.add_argument("--output",
                    help="work folder for scanned/, enhanced/ and indexed/ "
                         "(default: <input>_pipeline next to the input folder)")
    ap.add_argument("--stage", choices=STAGES, default=None,
                    help="run a single stage instead of all three "
                         "(expects the previous stage's output to exist)")
    ap.add_argument("--apps", type=int, default=None, metavar="N",
                    help="after scanning, keep only the first N applications "
                         "(quick timing run)")
    ap.add_argument("--workers", type=int, default=None,
                    help="worker count (default: auto from CPU/GPU)")
    a = ap.parse_args()

    raw = os.path.abspath(a.input)
    if not os.path.isdir(raw):
        logging.error(f"input folder does not exist: {raw}")
        return 1

    out_root = os.path.abspath(a.output) if a.output else raw.rstrip("\\/") + "_pipeline"
    scanned  = os.path.join(out_root, "scanned")
    enhanced = os.path.join(out_root, "enhanced")
    indexed  = os.path.join(out_root, "indexed")
    os.makedirs(out_root, exist_ok=True)

    from ocr_engine import cuda_available
    logging.info(f"input   : {raw}")
    logging.info(f"work dir: {out_root}")
    logging.info(f"OCR on  : {'GPU (CUDA)' if cuda_available() else 'CPU'}")

    run_all = a.stage is None
    times: dict[str, float] = {}
    pages: dict[str, int] = {}

    try:
        if run_all or a.stage == "scan":
            times["scan"] = stage_scan(raw, scanned, a.workers)
            if a.apps:
                kept = _keep_first_apps(scanned, a.apps)
                logging.info(f"--apps {a.apps}: kept {kept} application(s) for this run")
            pages["scan"] = _count_pages(scanned)

        if run_all or a.stage == "enhance":
            times["enhance"] = stage_enhance(scanned, enhanced, a.workers)
            pages["enhance"] = _count_pages(enhanced)

        if run_all or a.stage == "index":
            pages["index"] = _count_pages(enhanced)
            times["index"] = stage_index(enhanced, indexed, a.workers)
    except KeyboardInterrupt:
        logging.warning("interrupted by user")
        return 130

    # ------------------------------------------------------------- report
    print()
    print("=" * 70)
    print("TIMING REPORT")
    print("=" * 70)
    print(f"  {'stage':10s} {'pages':>7s} {'total':>12s} {'per page':>12s}")
    print(f"  {'-'*10} {'-'*7} {'-'*12} {'-'*12}")
    for s in STAGES:
        if s not in times:
            continue
        n = pages.get(s, 0)
        per = f"{times[s] / n * 1000:.0f} ms" if n else "-"
        print(f"  {s:10s} {n:7d} {_fmt(times[s]):>12s} {per:>12s}")
    if len(times) > 1:
        total = sum(times.values())
        print(f"  {'-'*10} {'-'*7} {'-'*12} {'-'*12}")
        print(f"  {'TOTAL':10s} {'':7s} {_fmt(total):>12s}")
    print()
    print(f"  scanned : {scanned}")
    print(f"  enhanced: {enhanced}")
    print(f"  indexed : {indexed}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
