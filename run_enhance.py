"""
Stage 1 runner - enhance a scan folder in parallel.

    python run_enhance.py --input <scan folder> --output <folder> [--workers N]

Processes the TIF of each scanned page (the JPGs are left for the later
applicant-photo step) and writes a cleaned bitonal Group4 TIFF per page, plus
a manifest.json describing what happened to every file.

Parallelism: enhancement is pure OpenCV/NumPy, which releases the GIL, so a
thread pool gives real speedup without the memory cost or frozen-exe
re-extraction problems of separate processes. Default worker count is capped
low (4-8) so it behaves on modest deployment hardware.
"""
from __future__ import annotations

import os
import json
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

import enhance as E

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def default_workers(cpu: int | None = None) -> int:
    cpu = cpu or os.cpu_count() or 4
    return max(4, min(8, cpu - 1))


def _one(job):
    in_dir, out_dir, f = job
    t0 = time.time()
    img, info = E.enhance_page(os.path.join(in_dir, f))
    if img is not None:
        out_name = os.path.splitext(f)[0] + ".tif"
        info["out"] = out_name
        info["bytes"] = E.save_group4(img, os.path.join(out_dir, out_name))
    info["ms"] = round((time.time() - t0) * 1000)
    return info


def run(in_dir: str, out_dir: str, workers: int | None = None, limit: int | None = None):
    os.makedirs(out_dir, exist_ok=True)
    docs = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(E.DOC_EXT))
    photos = [f for f in os.listdir(in_dir) if f.lower().endswith(E.PHOTO_EXT)]
    if limit:
        docs = docs[:limit]
    if not docs:
        logging.error(f"no TIF pages found in {in_dir}")
        return

    workers = workers or default_workers()
    logging.info(f"{len(docs)} document pages ({len(photos)} JPGs left for the photo step) "
                 f"| {workers} workers -> {out_dir}")

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, info in enumerate(ex.map(_one, [(in_dir, out_dir, f) for f in docs]), 1):
            results.append(info)
            if i % 50 == 0 or i == len(docs):
                el = time.time() - t0
                logging.info(f"[{i}/{len(docs)}] {el:.0f}s  {el/i*1000:.0f} ms/page  "
                             f"({i/el:.1f} pages/s)")

    el = time.time() - t0
    ok = sum(1 for r in results if r.get("status") == "ok")
    blank = sum(1 for r in results if r.get("status") == "blank")
    err = sum(1 for r in results if r.get("status") == "error")
    holes = sum(r.get("holes_removed", 0) for r in results)
    skews = [abs(r.get("skew", 0)) for r in results if r.get("status") == "ok"]
    sizes = [r["bytes"] for r in results if "bytes" in r]

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"input": in_dir, "workers": workers,
                   "seconds": round(el, 1), "ms_per_page": round(el / len(docs) * 1000),
                   "pages": results}, fh, indent=1)

    logging.info("=" * 60)
    logging.info(f"pages={len(docs)}  ok={ok}  blank={blank}  error={err}")
    logging.info(f"time={el:.0f}s  {el/len(docs)*1000:.0f} ms/page  {len(docs)/el:.1f} pages/s")
    logging.info(f"feed holes removed={holes}  "
                 f"skew corrected: max={max(skews) if skews else 0:.1f} deg, "
                 f"mean={sum(skews)/len(skews) if skews else 0:.2f} deg")
    if sizes:
        logging.info(f"output TIFF size: mean={sum(sizes)/len(sizes)/1024:.0f} KB  "
                     f"max={max(sizes)/1024:.0f} KB")
    logging.info("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Enhance a scanned batch folder (stage 1)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.input, a.output, a.workers, a.limit)
