"""
Stage 2 - indexing: split an enhanced batch into per-application multi-page TIFFs.

    python index_batch.py --raw <scan folder> --enhanced <stage-1 enhanced folder>
                          --output <folder> [--workers N]

Structure produced per application:
    <output>/<APP>/proposal_forms.tif    <- all proposal form pages combined
                  /review_slips.tif
                  /kyc.tif
                  /enclosure.tif
                  /bank.tif
                  /medical_report.tif
                  /unidentified.tif
                  /signature.tif         <- customer signatures (pages 6 & 7 of
                                            proposal forms, right-hand side only)

A category file is only created when at least one page belongs to it.
All pages within a file are in original scan order.

Application boundaries come from the separator sheets, identified by carrying
MULTIPLE barcodes on one page. That rule is site-independent and does not
mistake a KYC page's single Aadhaar QR code for a separator.

Folder naming: APP_01, APP_02, ... (application numbers are handwritten and
require a separate read-numbers pass -- see apply_numbers.py).
"""
from __future__ import annotations

import os
import re
import csv
import json
import time
import shutil
import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import zxingcpp
from PIL import Image, TiffImagePlugin

import appnumber as AN

_OFF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "offlineImageClassification")
import sys
sys.path.insert(0, _OFF)
from ocr_engine import RapidOCREngine          # noqa: E402
from rules import RuleBasedClassifier          # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# classifier category -> output filename (without .tif)
CATEGORY_FILE = {
    "KYC_documents":         "kyc",
    "Proposal_form":         "proposal_forms",
    "Proposal_review_slip":  "review_slips",
    "Proposal_enclosures":   "enclosure",
    "Bank":                  "bank",
    "Medical_report":        "medical_report",
    "unidentified":          "unidentified",
}

SIG_SOURCE = "Proposal_form"   # signatures come from proposal forms only
# Page positions (1-indexed within the application's proposal form pages) where
# customer signatures appear: pages 6 and 7 of Form 300.
SIG_PAGES = {6, 7}

SIG_MODEL = os.path.abspath(os.path.join(_OFF, "models", "signature", "signature.onnx"))

_tls = threading.local()


def _ocr():
    e = getattr(_tls, "ocr", None)
    if e is None:
        e = _tls.ocr = RapidOCREngine(intra_threads=1)
    return e


def _clf():
    c = getattr(_tls, "clf", None)
    if c is None:
        c = _tls.clf = RuleBasedClassifier()
    return c


def _sig():
    s = getattr(_tls, "sig", None)
    if s is None:
        from signature_detector import SignatureDetector
        s = _tls.sig = SignatureDetector(model_path=SIG_MODEL, conf_thresh=0.5)
    return s


# ---------------------------------------------------------------- multi-page TIFF
def save_multipage_tiff(paths: list[str], out_path: str,
                        compression: str = "group4") -> int:
    """Combine TIF pages into one multi-page TIFF. Returns file size in bytes."""
    pages = []
    for p in paths:
        try:
            img = Image.open(p)
            # convert to 1-bit for group4 (bitonal), or keep as-is for lzw
            if compression == "group4":
                if img.mode not in ("1", "L"):
                    img = img.convert("L")
                if img.mode == "L":
                    img = img.point(lambda x: 0 if x < 128 else 255, "1")
            else:
                if img.mode not in ("L", "RGB"):
                    img = img.convert("L")
            pages.append(img.copy())
            img.close()
        except Exception as e:
            logging.warning(f"could not open {p} for multipage tiff: {e}")
    if not pages:
        return 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pages[0].save(
        out_path,
        save_all=True,
        append_images=pages[1:],
        compression=compression,
    )
    for p in pages:
        p.close()
    return os.path.getsize(out_path)


def save_signature_tif(crops: list[np.ndarray], out_path: str) -> int:
    """Write signature crops as a multi-page TIFF (LZW, grayscale)."""
    if not crops:
        return 0
    pages = []
    for c in crops:
        if c is None or c.size == 0:
            continue
        if len(c.shape) == 3:
            c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        pages.append(Image.fromarray(c))
    if not pages:
        return 0
    pages[0].save(out_path, save_all=True, append_images=pages[1:], compression="tiff_lzw")
    for p in pages:
        p.close()
    return os.path.getsize(out_path)


# ---------------------------------------------------------------- separators
def barcode_count(path: str, max_side: int = 1600) -> tuple[int, list[str]]:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        try:
            g = np.array(Image.open(path).convert("L"))
        except Exception:
            return 0, []
    s = max_side / max(g.shape[:2])
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    try:
        res = zxingcpp.read_barcodes(g)
    except Exception:
        return 0, []
    return len(res), sorted({r.text[:24] for r in res})


def find_separators(raw_dir: str, files: list[str], workers: int,
                    min_barcodes: int = 2) -> set[str]:
    def job(f):
        n, texts = barcode_count(os.path.join(raw_dir, f))
        return f, n, texts
    seps = set()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f, n, _t in ex.map(job, files):
            if n >= min_barcodes:
                seps.add(f)
    return seps


def group_applications(files: list[str], seps: set[str]) -> list[list[str]]:
    groups, cur = [], []
    for f in files:
        if f in seps:
            if cur:
                groups.append(cur)
            cur = []
        else:
            cur.append(f)
    if cur:
        groups.append(cur)
    return groups


# ---------------------------------------------------------------- page work
def classify_page(enh_dir: str, f: str, max_side: int = 1000):
    p = os.path.join(enh_dir, f)
    if not os.path.exists(p):
        return f, None, "", None
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if g is None:
        try:
            g = np.array(Image.open(p).convert("L"))
        except Exception:
            return f, None, "", None
    s = max_side / max(g.shape[:2])
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    text = _ocr().text_of(np.stack([g] * 3, axis=-1))
    cat, label, _ = _clf().classify(text)
    return f, cat, text, label


def extract_right_side_signatures(enh_path: str) -> list[np.ndarray]:
    """Detect signatures on the right half of the page only (customer section).

    The proposal form has two signature boxes per page: agent on the left,
    customer on the right. We keep only detections whose centre x > 50% of
    the page width.
    """
    det = _sig()
    crops = []
    try:
        full = cv2.imread(enh_path, cv2.IMREAD_GRAYSCALE)
        if full is None:
            full = np.array(Image.open(enh_path).convert("L"))
        page_w = full.shape[1]
        right_start = page_w // 2

        dets = det.detect(enh_path)
        for d in dets:
            box = d["box"]   # [x1, y1, x2, y2] in original pixels
            x_center = (box[0] + box[2]) / 2
            if x_center < right_start:
                continue  # left half = agent/witness signature, skip
            img = det.crop(enh_path, box)
            if img is not None and img.size:
                crops.append(img)
    except Exception as e:
        logging.debug(f"signature detect failed on {enh_path}: {e}")
    return crops


# ---------------------------------------------------------------- main
def run(raw_dir: str, enh_dir: str, out_dir: str, workers: int | None = None,
        max_apps: int | None = None):
    workers = workers or max(4, min(8, (os.cpu_count() or 4) - 1))
    raw_tifs = sorted(f for f in os.listdir(raw_dir) if f.lower().endswith((".tif", ".tiff")))
    enh_have = {f for f in os.listdir(enh_dir) if f.lower().endswith((".tif", ".tiff"))}

    logging.info(f"{len(raw_tifs)} scanned pages | {len(enh_have)} enhanced | {workers} workers")
    t0 = time.time()

    logging.info("finding separator sheets (pages with multiple barcodes)...")
    if max_apps:
        seps_found, seps = 0, set()
        for i, f in enumerate(raw_tifs):
            n, _t = barcode_count(os.path.join(raw_dir, f))
            if n >= 2:
                seps.add(f)
                seps_found += 1
                if seps_found >= max_apps:
                    raw_tifs = raw_tifs[:i + 1]
                    break
    else:
        seps = find_separators(raw_dir, raw_tifs, workers)
    groups = group_applications(raw_tifs, seps)
    if max_apps:
        groups = groups[:max_apps]
    logging.info(f"  {len(seps)} separators -> {len(groups)} applications "
                 f"(sizes {[len(g) for g in groups]})")

    # OCR + classify every enhanced page once
    todo = [f for g in groups for f in g if f in enh_have]
    logging.info(f"classifying {len(todo)} enhanced pages...")
    page_cat: dict[str, str] = {}
    page_label: dict[str, str] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f, cat, _text, label in ex.map(lambda x: classify_page(enh_dir, x), todo):
            if cat:
                page_cat[f] = cat
                page_label[f] = label
            done += 1
            if done % 100 == 0 or done == len(todo):
                logging.info(f"  [{done}/{len(todo)}] {time.time()-t0:.0f}s")

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    summary = []

    for gi, grp in enumerate(groups, 1):
        app = f"APP_{gi:02d}"
        app_dir = os.path.join(out_dir, app)
        os.makedirs(app_dir, exist_ok=True)

        # Group pages by category (in scan order)
        cat_pages: dict[str, list[str]] = {k: [] for k in CATEGORY_FILE.values()}
        proposal_form_sequence = []  # track which proposal-form pages and their sequence position

        for f in grp:
            if f not in enh_have:
                continue
            cat = page_cat.get(f, "unidentified")
            fname = CATEGORY_FILE.get(cat, "unidentified")
            cat_pages[fname].append(os.path.join(enh_dir, f))
            if cat == SIG_SOURCE:
                proposal_form_sequence.append(f)
            rows.append([app, f, cat, fname, page_label.get(f, "")])

        # Write one multi-page TIFF per category that has pages
        cat_sizes = {}
        for fname, paths in cat_pages.items():
            if not paths:
                continue
            out_path = os.path.join(app_dir, f"{fname}.tif")
            sz = save_multipage_tiff(paths, out_path, compression="group4")
            cat_sizes[fname] = {"pages": len(paths), "bytes": sz}
            logging.debug(f"  {app}/{fname}.tif: {len(paths)} pages, {sz//1024}KB")

        # Extract customer signatures from pages 6 & 7 of the proposal form
        # (right-hand side only = customer box, not agent box on the left)
        sig_crops = []
        for seq_idx, f in enumerate(proposal_form_sequence, 1):
            if seq_idx not in SIG_PAGES:
                continue
            enh_path = os.path.join(enh_dir, f)
            if not os.path.exists(enh_path):
                continue
            crops = extract_right_side_signatures(enh_path)
            sig_crops.extend(crops)
            logging.debug(f"  {app} sig page {seq_idx} ({f}): {len(crops)} crops")

        sig_size = 0
        if sig_crops:
            sig_path = os.path.join(app_dir, "signature.tif")
            sig_size = save_signature_tif(sig_crops, sig_path)

        total_pages = sum(v["pages"] for v in cat_sizes.values())
        summary.append({
            "app": app,
            "pages": len(grp),
            "filed": total_pages,
            "signature_crops": len(sig_crops),
            "categories": {k: v["pages"] for k, v in cat_sizes.items()},
        })
        logging.info(f"  {app}: {len(grp)} pages filed={total_pages} "
                     f"sigs={len(sig_crops)} cats={list(cat_sizes)}")

    with open(os.path.join(out_dir, "index_report.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["application", "page_file", "category", "subfolder", "label"])
        w.writerows(rows)
    with open(os.path.join(out_dir, "index_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"applications": len(groups), "separators": sorted(seps),
                   "seconds": round(time.time() - t0, 1), "detail": summary}, fh, indent=1)

    el = time.time() - t0
    total_filed = sum(s["filed"] for s in summary)
    logging.info("=" * 64)
    logging.info(f"applications={len(groups)}  pages_filed={total_filed}  "
                 f"signatures={sum(s['signature_crops'] for s in summary)}")
    logging.info(f"time={el:.0f}s ({el/max(1,total_filed)*1000:.0f} ms/page)")
    logging.info(f"output: {out_dir}")
    logging.info("=" * 64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Index an enhanced batch into per-application multi-page TIFFs")
    ap.add_argument("--raw", required=True, help="original scan folder (for barcode separators)")
    ap.add_argument("--enhanced", required=True, help="stage-1 enhanced folder")
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-apps", type=int, default=None,
                    help="only index the first N applications (for a quick trial)")
    a = ap.parse_args()
    run(a.raw, a.enhanced, a.output, a.workers, a.max_apps)
