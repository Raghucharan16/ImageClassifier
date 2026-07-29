"""
Stage 2 - indexing: classify enhanced pages inside APP_XX folders and
combine them into per-category multi-page TIFFs.

Input folder structure (output of enhance.exe):
    <apps_dir>/
        APP_01/   00000002.tif  00000004.tif  ...
        APP_02/   00000036.tif  ...
        previews/
        skipped_blank/
        report.csv

Output (written IN PLACE inside each APP folder):
    APP_01/
        proposal_forms.tif    <- all proposal-form pages combined (Group4)
        review_slips.tif
        kyc.tif
        enclosure.tif
        bank.tif
        medical_report.tif
        unidentified.tif
        signature.tif         <- customer signatures, pages 6+7, right side
    (individual page TIFs are removed after combining)

Usage:
    python index_batch.py --apps <apps_dir> [--workers N]
"""
from __future__ import annotations

import os
import sys
import csv
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image

import appnumber as AN

_OFF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "offlineImageClassification")
sys.path.insert(0, _OFF)
from ocr_engine import RapidOCREngine      # noqa: E402
from rules import RuleBasedClassifier      # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CATEGORY_FILE = {
    "KYC_documents":        "kyc",
    "Proposal_form":        "proposal_forms",
    "Proposal_review_slip": "review_slips",
    "Proposal_enclosures":  "enclosure",
    "Bank":                 "bank",
    "Medical_report":       "medical_report",
    "unidentified":         "unidentified",
}

SIG_SOURCE = "Proposal_form"
SIG_PAGES  = {6, 7}          # 1-indexed within the app's proposal-form pages

# In a frozen EXE the model is bundled into _MEIPASS; otherwise use the repo path
if getattr(sys, "frozen", False):
    SIG_MODEL = os.path.join(sys._MEIPASS, "models", "signature", "signature.onnx")
else:
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
    """Combine individual TIF pages into one multi-page TIFF.

    Uses Group4 (bitonal, smallest) for document pages and LZW for
    signature crops (grayscale). Returns file size in bytes.
    """
    pages = []
    for p in paths:
        try:
            img = Image.open(p)
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
            logging.warning(f"skipping {p} in multipage tiff: {e}")
    if not pages:
        return 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pages[0].save(out_path, save_all=True, append_images=pages[1:],
                  compression=compression)
    for p in pages:
        p.close()
    return os.path.getsize(out_path)


def save_signature_tif(crops: list[np.ndarray], out_path: str) -> int:
    """Write signature crops as a multi-page LZW TIFF."""
    pages = []
    for c in crops:
        if c is None or c.size == 0:
            continue
        if len(c.shape) == 3:
            c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        pages.append(Image.fromarray(c))
    if not pages:
        return 0
    pages[0].save(out_path, save_all=True, append_images=pages[1:],
                  compression="tiff_lzw")
    for p in pages:
        p.close()
    return os.path.getsize(out_path)


# ---------------------------------------------------------------- page OCR/classify
def classify_page(app_dir: str, f: str, max_side: int = 1000):
    """OCR + classify one enhanced page. Returns (filename, category, label)."""
    p = os.path.join(app_dir, f)
    try:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            g = np.array(Image.open(p).convert("L"))
        s = max_side / max(g.shape[:2])
        if s < 1:
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        text = _ocr().text_of(np.stack([g] * 3, axis=-1))
        cat, label, _ = _clf().classify(text)
        return f, cat, label
    except Exception as e:
        logging.debug(f"classify failed {f}: {e}")
        return f, "unidentified", "error"


# ---------------------------------------------------------------- signatures
def extract_right_signatures(page_path: str) -> list[np.ndarray]:
    """Signatures on right half only (customer box, not agent box on left)."""
    det = _sig()
    crops = []
    try:
        full = cv2.imread(page_path, cv2.IMREAD_GRAYSCALE)
        if full is None:
            full = np.array(Image.open(page_path).convert("L"))
        right_start = full.shape[1] // 2
        for d in det.detect(page_path):
            box = d["box"]
            if (box[0] + box[2]) / 2 < right_start:
                continue
            img = det.crop(page_path, box)
            if img is not None and img.size:
                crops.append(img)
    except Exception as e:
        logging.debug(f"sig detect failed {page_path}: {e}")
    return crops


# ---------------------------------------------------------------- index one app
def index_app(app_dir: str, workers: int) -> dict:
    """Classify all TIFs in app_dir, write category TIFFs, remove individual pages."""
    tifs = sorted(f for f in os.listdir(app_dir)
                  if f.lower().endswith((".tif", ".tiff"))
                  and not any(f == f"{cat}.tif" for cat in CATEGORY_FILE.values())
                  and f != "signature.tif")
    if not tifs:
        return {"pages": 0, "categories": {}, "signature_crops": 0}

    # OCR + classify all pages in parallel
    page_cat: dict[str, str] = {}
    page_label: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fname, cat, label in ex.map(lambda f: classify_page(app_dir, f), tifs):
            page_cat[fname] = cat
            page_label[fname] = label

    # Group paths by category (preserving scan order)
    cat_paths: dict[str, list[str]] = {v: [] for v in CATEGORY_FILE.values()}
    proposal_seq: list[str] = []
    for f in tifs:
        cat   = page_cat.get(f, "unidentified")
        fname = CATEGORY_FILE.get(cat, "unidentified")
        cat_paths[fname].append(os.path.join(app_dir, f))
        if cat == SIG_SOURCE:
            proposal_seq.append(f)

    # Write one multi-page TIFF per category
    cat_sizes: dict[str, int] = {}
    for fname, paths in cat_paths.items():
        if not paths:
            continue
        sz = save_multipage_tiff(paths, os.path.join(app_dir, f"{fname}.tif"),
                                 compression="group4")
        cat_sizes[fname] = len(paths)
        logging.debug(f"  {os.path.basename(app_dir)}/{fname}.tif: "
                      f"{len(paths)} pages, {sz//1024}KB")

    # Extract customer signatures from pages 6 & 7 of proposal form
    sig_crops: list[np.ndarray] = []
    for seq_i, f in enumerate(proposal_seq, 1):
        if seq_i not in SIG_PAGES:
            continue
        sig_crops.extend(extract_right_signatures(os.path.join(app_dir, f)))

    sig_count = 0
    if sig_crops:
        save_signature_tif(sig_crops, os.path.join(app_dir, "signature.tif"))
        sig_count = len(sig_crops)

    # Remove individual page TIFs (now combined into category TIFFs)
    for f in tifs:
        try:
            os.remove(os.path.join(app_dir, f))
        except Exception:
            pass

    return {
        "pages": len(tifs),
        "categories": cat_sizes,
        "signature_crops": sig_count,
    }


# ---------------------------------------------------------------- main
def run(apps_dir: str, workers: int | None = None) -> None:
    workers = workers or max(4, min(8, (os.cpu_count() or 4) - 1))

    app_dirs = sorted(
        d for d in os.listdir(apps_dir)
        if os.path.isdir(os.path.join(apps_dir, d)) and d.startswith("APP_")
    )
    if not app_dirs:
        logging.error(f"No APP_XX folders found in {apps_dir}. "
                      f"Run enhance.exe first.")
        return

    logging.info(f"Indexing {len(app_dirs)} applications in {apps_dir}")
    t0 = time.time()
    summary = []
    rows = []

    for app in app_dirs:
        app_dir = os.path.join(apps_dir, app)
        logging.info(f"  {app} ...")
        result = index_app(app_dir, workers)
        summary.append({"app": app, **result})
        for cat, n in result["categories"].items():
            rows.append([app, cat, n])
        logging.info(f"  {app}: {result['pages']} pages -> "
                     f"{list(result['categories'])} sigs={result['signature_crops']}")

    el = time.time() - t0
    total_pages = sum(s["pages"] for s in summary)

    with open(os.path.join(apps_dir, "index_report.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["application", "category", "pages"])
        w.writerows(rows)

    with open(os.path.join(apps_dir, "index_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"applications": len(app_dirs),
                   "seconds": round(el, 1),
                   "detail": summary}, fh, indent=1)

    logging.info("=" * 64)
    logging.info(f"applications={len(app_dirs)}  total_pages_filed={total_pages}")
    logging.info(f"time={el:.0f}s  ({el/max(1,total_pages)*1000:.0f} ms/page)")
    logging.info(f"output: {apps_dir}")
    logging.info("=" * 64)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Index APP_XX folders into per-category multi-page TIFFs")
    ap.add_argument("--apps", required=True,
                    help="folder containing APP_01/, APP_02/... (output of enhance.exe)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    run(a.apps, a.workers)
