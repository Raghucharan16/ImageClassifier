"""
Preprocessing stage for the offline classifier.

Two speed tiers, chosen by what the caller needs:

  * Enhance-ONLY (no OCR): blank-skip -> best-effort classical rotation guess
    (no OCR/ML, ~40ms) -> adaptive contrast/denoise (only applied when the
    page actually needs it) -> grayscale, <=50KB (no category is known yet,
    so this path can't tell photo-ID pages apart -- see below). Target:
    <1s/image on modest hardware (e.g. an 11th-gen i5-G7, 1.5GHz). Classical
    rotation is 100% reliable for sideways (90/270) in testing, but can
    occasionally mis-guess upside-down (180) on unusual layouts (e.g. a
    composite multi-card KYC page) -- that is an accepted, documented limit.

  * Enhance+Classify TOGETHER: rotation correction is done ONCE via the OCR
    pass that Classify needs anyway (reliable for all 4 orientations), instead
    of guessing twice. The classified CATEGORY then decides the size/colour
    budget: KYC_documents/Bank keep colour at <=200KB (worth preserving an ID
    photo/logo); everything else is cleaned grayscale at <=50KB. (Pixel-based
    "does this look like a photo" detection was tried and found unreliable on
    this dataset -- many KYC scans are themselves grayscale photocopies, while
    plain text pages often measure MORE "colourful" from stray ink/stamps/JPEG
    artifacts than real ID cards -- so the category is used instead.) This is
    the slow-but-reliable path -- OCR is inherently multi-second per dense
    page; this module does not pretend otherwise.

Neither path does destructive things (deskew-by-arbitrary-angle, harsh
binarisation, auto-crop, forced sharpening) -- those were measured to lower
downstream classification accuracy.
"""
import os
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from ocr_engine import RapidOCREngine, _load_bgr, _rotate
from rules import RuleBasedClassifier

# Categories whose pages are worth a bigger size budget to keep embedded ID
# photos/logos legible. Pixel-based "does this look like a photo" detection
# was tested and found unreliable on this dataset (many KYC scans are
# themselves grayscale photocopies, while plain text pages often measure
# MORE "colorful" from stray ink/stamps/JPEG artifacts than real ID cards).
# The category is a far more reliable signal once it's known (merged path).
PHOTO_CATEGORIES = {"KYC_documents", "Bank"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

IMAGE_EXTS = (".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp")
OCR_CACHE_NAME = "_ocr_cache.json"
TEXT_MAX_KB = 50
COLOR_MAX_KB = 200


# ---------------------------------------------------------------- fast checks
def is_blank(gray, dark_pct_thresh=1.0, flat_std=3.0):
    """Blank if the inner 90% region has almost no ink (dark pixels < ~1%),
    or is an essentially uniform page (very low std -> solid/empty scan).

    Ink coverage is the reliable signal: real documents -- even low-contrast
    ID cards -- have several percent dark pixels, while blanks have well under
    1%. (Contrast/std alone falsely flags faint-but-inked cards as blank.)"""
    h, w = gray.shape[:2]
    my, mx = int(h * 0.05), int(w * 0.05)
    inner = gray[my:h - my, mx:w - mx]
    if inner.size == 0:
        return True
    _, mask = cv2.threshold(inner, 240, 255, cv2.THRESH_BINARY_INV)
    dark_pct = cv2.countNonZero(mask) / inner.size * 100
    return dark_pct < dark_pct_thresh or float(np.std(inner)) < flat_std




# ---------------------------------------------------------------- classical (no-OCR) orientation
def classical_orient(bgr, detect_side=300):
    """Best-effort 4-way (0/90/180/270) orientation guess with NO OCR/ML --
    pure OpenCV, ~30-50ms regardless of image size (works on a small thumbnail).

    Axis (0/180 vs 90/270) is decided from which of the row-wise / column-wise
    ink-density profiles is more "peaky" (real text lines create periodic
    banding along the direction perpendicular to the lines) -- measured 100%
    correct on sideways (90/270) test pages. Within an axis, 0-vs-180 (or
    90-vs-270) is decided by assuming the busier half of the page (denser ink)
    is the top/left -- true for most letterhead/table-heavy business documents,
    but can mis-fire on unusual layouts (e.g. multiple ID cards stacked with
    even density) -- accepted limitation, this is a fast heuristic, not OCR.

    Returns the rotation angle (0/90/180/270) to apply via ocr_engine._rotate.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h0, w0 = gray.shape[:2]
    f = detect_side / max(h0, w0)
    small = cv2.resize(gray, (max(1, int(w0 * f)), max(1, int(h0 * f))))
    thresh = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    row_var = thresh.sum(axis=1).astype(np.float64).std()
    col_var = thresh.sum(axis=0).astype(np.float64).std()
    h, w = thresh.shape

    if row_var >= col_var:
        top = int(thresh[: h // 2, :].sum())
        bot = int(thresh[h // 2:, :].sum())
        return 0 if top >= bot else 180
    left = int(thresh[:, : w // 2].sum())
    right = int(thresh[:, w // 2:].sum())
    return 270 if left >= right else 90


# ---------------------------------------------------------------- adaptive cleaning
def _needs_contrast_boost(gray, std_thresh=40.0):
    return float(gray.std()) < std_thresh


def _noise_level(gray):
    """Cheap noise proxy: mean absolute deviation from a 3x3 median (already
    needed for speck-removal, so this reuses that pass)."""
    med = cv2.medianBlur(gray, 3)
    return med, float(np.mean(np.abs(gray.astype(np.int16) - med.astype(np.int16))))


def remove_specks(gray, med=None, max_area=10, max_wh=6):
    """Erase tiny isolated dark specks (scanner pepper / stray dots). Text,
    handwriting and signatures are far larger, so they are left untouched."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = gray.copy()
    for i in range(1, n):
        if (stats[i, cv2.CC_STAT_AREA] <= max_area
                and stats[i, cv2.CC_STAT_WIDTH] <= max_wh
                and stats[i, cv2.CC_STAT_HEIGHT] <= max_wh):
            out[labels == i] = 255
    return out


def adaptive_clean_gray(gray):
    """Only correct what's actually wrong -- skip work on already-clean pages.
    Low contrast -> CLAHE boost. Noticeably noisy -> cheap median smoothing.
    Always remove stray specks (fast, never hurts). All ops are sub-100ms on
    a several-megapixel page even on a slow CPU (no NLMeans / heavy filters)."""
    out = gray
    if _needs_contrast_boost(out):
        out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(out)
    med, noise = _noise_level(out)
    if noise > 6.0:
        out = med  # the median pass is already computed; reuse it as the smooth
    return remove_specks(out, med)


def adaptive_clean_color(bgr):
    """Same idea for colour cards/photos: only denoise if actually noisy."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, noise = _noise_level(gray)
    if noise > 8.0:
        return cv2.bilateralFilter(bgr, 5, 40, 40)
    return bgr


def _save_under(img, path, max_kb):
    """Write JPEG under max_kb, preferring full resolution / high quality."""
    buf = None
    for scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
        im = img if scale == 1.0 else cv2.resize(
            img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for q in (90, 80, 72, 64, 55, 45, 35, 25, 15):
            ok, b = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok and b.nbytes <= max_kb * 1024:
                with open(path, "wb") as fh:
                    fh.write(b.tobytes())
                return b.nbytes
            buf = b
    with open(path, "wb") as fh:      # fallback: smallest attempt reached
        fh.write(buf.tobytes())
    return buf.nbytes


def _finish(bgr, out_path, keep_color=False):
    """Adaptively clean, compress, save. `keep_color` decides the size budget
    and whether the saved image stays colour (True, <=200KB, for KYC/Bank
    pages where an ID photo/logo is worth preserving) or is converted to a
    cleaned grayscale JPEG (False, <=50KB, for the plain-text categories)."""
    if keep_color:
        restored = adaptive_clean_color(bgr)
        return "colour", _save_under(restored, out_path, COLOR_MAX_KB) / 1024
    gray = adaptive_clean_gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    return "text", _save_under(gray, out_path, TEXT_MAX_KB) / 1024


# ---------------------------------------------------------------- worker pool sizing
def _default_workers(cpu):
    """4-8 threads regardless of the dev machine's core count -- tuned for
    modest deployment hardware (e.g. an 11th-gen i5-G7 @ 1.5GHz, 4C/8T)
    rather than scaling up to whatever big box this happened to be built on."""
    return max(4, min(8, cpu - 1)) if cpu > 1 else 4


# ---------------------------------------------------------------- Enhance-ONLY (fast, no OCR)
def _process_one_fast(input_dir, out_dir, f):
    bgr = _load_bgr(os.path.join(input_dir, f))
    if bgr is None:
        return (f, "error")
    if is_blank(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)):
        return (f, "blank")
    angle = classical_orient(bgr)
    if angle:
        bgr = _rotate(bgr, angle)
    out_name = os.path.splitext(f)[0] + ".jpg"
    _finish(bgr, os.path.join(out_dir, out_name))
    return (out_name, "ok")


def preprocess_images_fast(input_dir, out_dir, workers=None):
    """Enhance-only: blank-skip, classical (no-OCR) rotation, adaptive clean,
    compress. No OCR text is produced/cached -- Classify (if run afterwards)
    will do its own OCR pass and may re-orient more reliably."""
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(IMAGE_EXTS))
    if not files:
        logging.error(f"No images found in {input_dir}.")
        return out_dir

    cpu = os.cpu_count() or 4
    workers = workers or _default_workers(cpu)
    workers = min(workers, len(files))
    logging.info(f"Enhance-only (no OCR): {len(files)} images, {workers} thread(s) -> {out_dir}")

    written = skipped = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for name, kind in ex.map(lambda f: _process_one_fast(input_dir, out_dir, f), files):
            if kind in ("blank", "error"):
                skipped += 1
                logging.info(f"SKIP {kind}: {name}")
            else:
                written += 1
    logging.info(f"Enhance-only complete. {written} written, {skipped} skipped -> {out_dir}.")
    return out_dir


# ---------------------------------------------------------------- Enhance+Classify merged (OCR once)
_tls = threading.local()


def _thread_engine(intra_threads):
    eng = getattr(_tls, "ocr", None)
    if eng is None:
        eng = _tls.ocr = RapidOCREngine(intra_threads=intra_threads)
    return eng


def _thread_classifier():
    clf = getattr(_tls, "clf", None)
    if clf is None:
        clf = _tls.clf = RuleBasedClassifier()
    return clf


def _process_one_merged(input_dir, out_dir, f, intra_threads):
    bgr = _load_bgr(os.path.join(input_dir, f))
    if bgr is None:
        return (f, None, "error")
    if is_blank(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)):
        return (f, None, "blank")
    bgr, text = _thread_engine(intra_threads).orient_and_read(bgr)   # reliable orient + OCR in one
    category, _, _ = _thread_classifier().classify(text)             # decides the size/colour budget
    out_name = os.path.splitext(f)[0] + ".jpg"
    _finish(bgr, os.path.join(out_dir, out_name), keep_color=category in PHOTO_CATEGORIES)
    return (out_name, text, "ok")


def preprocess_and_ocr(input_dir, out_dir, workers=None):
    """Enhance+Classify combined: OCR (reliable 4-way orient + text) runs
    ONCE per image and its result feeds both the cleaned/compressed output
    image AND the classification text cache -- no duplicated OCR work."""
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(IMAGE_EXTS))
    if not files:
        logging.error(f"No images found in {input_dir}.")
        return out_dir

    cpu = os.cpu_count() or 4
    workers = workers or _default_workers(cpu)
    workers = min(workers, len(files))
    intra = max(1, cpu // workers)
    logging.info(f"Enhance+Classify (OCR once): {len(files)} images, {workers} thread(s) "
                 f"(intra_threads={intra}) -> {out_dir}")

    cache, written, skipped = {}, 0, 0

    def work(f):
        return _process_one_merged(input_dir, out_dir, f, intra)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for name, text, kind in ex.map(work, files):
            if kind in ("blank", "error"):
                skipped += 1
                logging.info(f"SKIP {kind}: {name}")
            else:
                cache[name] = text
                written += 1

    with open(os.path.join(out_dir, OCR_CACHE_NAME), "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False)
    logging.info(f"Preprocessing complete. {written} written, {skipped} skipped -> {out_dir}.")
    return out_dir


# Backwards-compatible aliases (older callers).
preprocess_images = preprocess_and_ocr
enhance_images = preprocess_and_ocr


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Preprocess scans (blank-skip / orient / clean / compress)")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", default="preprocessed")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--fast", action="store_true", help="Enhance-only, no OCR (classical rotation guess)")
    a = ap.parse_args()
    if a.fast:
        preprocess_images_fast(a.input_dir, a.out_dir, workers=a.workers)
    else:
        preprocess_and_ocr(a.input_dir, a.out_dir, workers=a.workers)
