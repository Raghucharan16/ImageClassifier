"""
Stage 2 - indexing: classify enhanced pages inside APP_XX folders and
combine them into per-category multi-page TIFFs.

Input folder structure (output of enhance.exe):
    <apps_dir>/
        APP_01/   00000002.tif  00000004.tif  ...
        APP_02/   00000036.tif  ...
        manifest.json           <- written by enhance.exe; contains original input path
        previews/
        skipped_blank/

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
        photo.jpg             <- applicant photo cropped from top-right of proposal
                                 form page 1 colour JPG (from original scan folder)
    (individual page TIFs are removed after combining)

Speed notes:
  - All pages across ALL apps are classified in one flat parallel pool so all
    workers stay busy even when individual apps have few pages.
  - OCR image is capped at 700px (down from 1000): halves pixel count, ~50%
    faster per page with no meaningful drop in keyword-match accuracy.

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
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from ocr_engine import RapidOCREngine, cuda_available   # noqa: E402
from rules import RuleBasedClassifier                   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CATEGORY_FILE = {
    "KYC_documents":        "kyc",
    "Proposal_form":        "proposal_forms",
    "Proposal_review_slip": "review_slips",
    "Proposal_enclosures":  "enclosure",
    # The LIC policy slip is filed with the enclosures, pinned to the last page
    # (see LAST_PAGE_CATS below).
    "LIC_slip":             "enclosure",
    "Medical_report":       "medical_report",
    "unidentified":         "unidentified",
}

# Categories that must appear at the END of their combined TIFF, after every
# other page routed to the same file, regardless of scan order.
LAST_PAGE_CATS = frozenset({"LIC_slip"})

# Height/width below which an enhanced page is the wide LIC policy slip.
# Measured on enhanced batches: slips 0.25-0.37, all other landscape pages
# (content-cropped ID cards and bank sheets) 0.59-0.95 -- 0.45 sits in the
# middle of that empty gap.
WIDE_RATIO = 0.45

SIG_SOURCE = "Proposal_form"
SIG_PAGES  = {6, 7}   # 1-indexed within the app's proposal-form pages

PHOTO_CROP_TOP  = 0.28   # top fraction of JPG to include in photo crop
PHOTO_CROP_RIGHT = 0.30  # right fraction of JPG to include in photo crop

# Signature crops: the detector boxes the pen strokes only, but the printed
# caption underneath ("Signature or Thumb impression of the Life to be assured")
# is what identifies WHOSE signature it is, so the box is grown to take it in.
#
# The caption is a full sentence and is much WIDER than the signature above it,
# so padding by a fraction of the signature's own width truncates it -- that cut
# "...of the Life to be assured" down to "...Thumb impressic". The horizontal
# extent is therefore measured from the caption itself (see _caption_span)
# rather than guessed from the box, and these fractions only seed the search.
SIG_PAD_BELOW   = 0.85   # extra height below the box, as a fraction of box height
SIG_PAD_SIDE    = 0.06   # minimum extra width each side, fraction of box width
SIG_PAD_ABOVE   = 0.10   # small headroom so descenders/flourishes are not clipped
SIG_CANVAS      = (1000, 300)  # (width, height) px of every signature page

# Caption-span search: the caption sits within this fraction of the page width
# either side of the signature, and words in it are separated by gaps no larger
# than SIG_WORD_GAP px (anything bigger is a different column of the form).
SIG_CAPTION_SEARCH = 0.42
SIG_WORD_GAP       = 42
# The search band starts this fraction of the box height ABOVE the box's bottom
# edge. The detector's box usually already overlaps the caption -- measured on a
# real page the caption baseline sat at y=703 inside a box ending at y=713 -- so
# a band starting at the bottom edge begins BELOW the lettering and finds only
# descenders, which is what truncated the sentence to "...Thumb impressic".
# Reaching up also picks up the horizontal rule above the caption, which runs to
# the same width and makes the span easier to follow.
SIG_CAPTION_UP     = 0.35

# In a frozen EXE the model is bundled into _MEIPASS; otherwise use the repo path
if getattr(sys, "frozen", False):
    SIG_MODEL = os.path.join(sys._MEIPASS, "models", "signature", "signature.onnx")
else:
    SIG_MODEL = os.path.join(_HERE, "models", "signature", "signature.onnx")

_tls = threading.local()


def _ocr():
    """OCR engine for this worker.

    use_cuda is left at None so RapidOCREngine auto-detects: GPU when the
    installed onnxruntime offers the CUDA provider, CPU otherwise. Set
    RAPIDOCR_CUDA=0 or 1 to override the detection.
    """
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
    """Signature detector for this worker.

    The detector is not fully reliable and cannot be made so by tuning: after a
    few stray noise pixels near one signature were cleaned up by enhancement, it
    stopped proposing a box for a signature that was still plainly there and
    pixel-identical, and lowering the threshold to 0.2 did not bring the box
    back -- the proposal was gone, not merely low-scoring. signatures_by_caption
    exists to cover that case.
    """
    s = getattr(_tls, "sig", None)
    if s is None:
        from signature_detector import SignatureDetector
        s = _tls.sig = SignatureDetector(model_path=SIG_MODEL, conf_thresh=0.5)
    return s


# ---------------------------------------------------------------- multi-page TIFF
def save_proposal_tiff(page_paths: list[str], photo_path: str | None,
                       out_path: str) -> int:
    """Proposal-form TIFF with the applicant's colour photo as PAGE 1.

    The two page types need DIFFERENT compression in one file, which a single
    Image.save(save_all=True) cannot express -- it applies one setting to every
    page. That matters because the photo is a continuous-tone colour crop of a
    face: stored as Group4 bitonal it collapses into unrecognisable black-and-
    white blotches, which is exactly why it cannot simply be prepended to the
    existing Group4 file.

    PIL's AppendingTiffWriter builds the file one frame at a time, each with its
    own encoder, so the photo goes in as lossless colour (deflate) while the
    document pages stay Group4 -- full quality for the face, no size penalty for
    the text. Written through PIL rather than tifffile because tifffile routes
    Group4, LZW and JPEG alike through the optional `imagecodecs` package, which
    is a poor thing to bolt onto an offline deployment; deflate and Group4 are
    both built into PIL.

    Falls back to a plain Group4 file (text pages only) if anything about the
    photo fails, so a missing or unreadable JPG can never cost us the proposal
    pages themselves.
    """
    from PIL import TiffImagePlugin

    photo = None
    if photo_path and os.path.exists(photo_path):
        try:
            with Image.open(photo_path) as im:
                photo = im.convert("RGB").copy()
        except Exception as e:
            logging.debug(f"proposal photo unreadable {photo_path}: {e}")

    if photo is None:
        return save_multipage_tiff(page_paths, out_path, compression="group4")

    text_pages = []
    for p in page_paths:
        try:
            img = Image.open(p)
            if img.mode not in ("1", "L"):
                img = img.convert("L")
            if img.mode == "L":
                img = img.point(lambda x: 0 if x < 128 else 255, "1")
            text_pages.append(img.copy())
            img.close()
        except Exception as e:
            logging.warning(f"skipping {p} in proposal tiff: {e}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        tw = TiffImagePlugin.AppendingTiffWriter(out_path, True)
        try:
            photo.save(tw, format="TIFF", compression="tiff_deflate",
                       dpi=(200, 200))
            tw.newFrame()
            for img in text_pages:
                img.save(tw, format="TIFF", compression="group4", dpi=(200, 200))
                tw.newFrame()
        finally:
            tw.close()
        return os.path.getsize(out_path)
    except Exception as e:
        # Never lose the document pages over a photo/codec problem.
        logging.warning(f"proposal tiff with photo failed ({e}); "
                        f"writing text-only Group4 instead")
        return save_multipage_tiff(page_paths, out_path, compression="group4")
    finally:
        photo.close()
        for img in text_pages:
            img.close()


def save_multipage_tiff(paths: list[str], out_path: str,
                        compression: str = "group4") -> int:
    """Combine individual TIF pages into one multi-page TIFF."""
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


def _fit_canvas(img: np.ndarray, size: tuple[int, int] = SIG_CANVAS) -> np.ndarray:
    """Letterbox a crop onto a fixed white canvas, preserving aspect ratio.

    Scaled down only (never up), so a small signature is not blurred into a
    pixelated mess just to fill the frame; it is centred on white instead.
    """
    cw, ch = size
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((ch, cw), 255, np.uint8)
    s = min(cw / w, ch / h, 1.0)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((ch, cw), 255, np.uint8)
    y0, x0 = (ch - nh) // 2, (cw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def save_signature_tif(crops: list[np.ndarray], out_path: str) -> int:
    """Write signature crops as a multi-page LZW TIFF, one standard size each."""
    pages = []
    for c in crops:
        if c is None or c.size == 0:
            continue
        if len(c.shape) == 3:
            c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        pages.append(Image.fromarray(_fit_canvas(c)))
    if not pages:
        return 0
    pages[0].save(out_path, save_all=True, append_images=pages[1:],
                  compression="tiff_lzw")
    for p in pages:
        p.close()
    return os.path.getsize(out_path)


# ---------------------------------------------------------------- page OCR/classify
def classify_page(app_dir: str, f: str, max_side: int = 1000):
    """OCR + classify one enhanced page. Returns (filename, category, label).

    The WHOLE page is read, at max_side=1000. Both of those matter and neither is
    a free parameter to tune for speed:

      * Reading only the top of the page breaks the classifier. Several rules in
        rules.py gate on total text length (`len(low) < 900`) to tell a small ID
        card from a densely printed form page. Cropping to the top half roughly
        halves the text, so form pages read as "short" and match the ID-card
        rules -- that filed a Form-300 "KYC & PMLA" page into kyc.tif. It also
        lets the loose _proposal_context catch-all win on a review slip's header
        area and return early, filing review slips as proposal forms.
      * Dropping to 700 px costs the dot-matrix review-slip header its
        legibility, which is the signal _review_slip depends on.

    So the length thresholds in rules.py are calibrated against a full-page read
    at this resolution; changing either side of that silently misclassifies.
    """
    p = os.path.join(app_dir, f)
    try:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            g = np.array(Image.open(p).convert("L"))
        s = max_side / max(g.shape[:2])
        if s < 1:
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

        h, w = g.shape[:2]
        # Wide strip => LIC policy slip. See RuleBasedClassifier.classify for why
        # the ratio (not plain landscape) is the test.
        is_wide = (h / w) < WIDE_RATIO if w else False

        text = _ocr().text_of(np.stack([g] * 3, axis=-1))
        cat, label, _ = _clf().classify(text, is_wide=is_wide)
        return f, cat, label

    except Exception as exc:
        logging.debug(f"classify failed {f}: {exc}")
        return f, "unidentified", "error"


def _classify_job(args: tuple[str, str]):
    # Prevent OpenMP from spawning extra threads inside each worker process;
    # without this ONNX/MKL creates N worker threads per process and all
    # competing processes oversubscribe the CPU.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    app_dir, f = args
    _, cat, label = classify_page(app_dir, f)
    return app_dir, f, cat, label


# ---------------------------------------------------------------- signatures
def _caption_span(gray: np.ndarray, x1: int, x2: int, y_from: int, y_to: int
                  ) -> tuple[int, int]:
    """Horizontal extent of the printed caption line under a signature.

    Projects the caption band onto the x axis and walks outward from the
    signature's own columns, absorbing further ink while the gaps stay smaller
    than a word space. That is what lets the crop take in the whole sentence:
    the caption runs well past both ends of the signature, so growing by a
    multiple of the signature's width either clips the sentence or, if made
    generous enough to always cover it, drags in the neighbouring form column.
    Stopping at the first gap wider than a word space follows the real text
    instead of assuming a width.

    Returns (x0, x1) in page coordinates.
    """
    h, w = gray.shape[:2]
    y_from = max(0, min(h - 1, y_from))
    y_to = max(y_from + 1, min(h, y_to))
    band = gray[y_from:y_to, :] < 128
    if not band.any():
        return x1, x2
    cols = band.sum(axis=0) > 0

    reach = int(w * SIG_CAPTION_SEARCH)
    lo, hi = max(0, x1 - reach), min(w, x2 + reach)

    left = x1
    gap = 0
    for x in range(x1 - 1, lo - 1, -1):
        if cols[x]:
            left, gap = x, 0
        else:
            gap += 1
            if gap > SIG_WORD_GAP:
                break

    right = x2
    gap = 0
    for x in range(x2, hi):
        if cols[x]:
            right, gap = x + 1, 0
        else:
            gap += 1
            if gap > SIG_WORD_GAP:
                break
    return left, right


def _grow_signature_box(box, gray: np.ndarray) -> tuple[int, int, int, int]:
    """Expand a detected signature box to take in its full printed caption.

    The detector boxes the pen strokes alone, but on Form 300 the line
    identifying the signatory ("Signature or Thumb impression of the Life to be
    assured") is printed BELOW the signature, so a tight crop yields an anonymous
    squiggle. The box grows downward to reach that line, then outward to the
    line's measured width so the sentence is captured whole.
    """
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)

    top = max(0, y1 - int(bh * SIG_PAD_ABOVE))
    bottom = min(h, y2 + int(bh * SIG_PAD_BELOW))

    # Measure the caption across a band that starts just inside the box bottom.
    cx0, cx1 = _caption_span(gray, x1, x2,
                             y2 - int(bh * SIG_CAPTION_UP), bottom)

    left = min(max(0, x1 - int(bw * SIG_PAD_SIDE)), cx0)
    right = max(min(w, x2 + int(bw * SIG_PAD_SIDE)), cx1)
    return left, top, right, bottom


# The printed line beneath a real signature on Form 300. This, not the size or
# shape of the ink, is what makes a signature identifiable.
# Matched against the caption with spaces/punctuation/DIGITS stripped, because
# OCR of this small printed line is reliably imperfect: a real caption came back
# as "are or Thumbimpressi0" -- space lost, trailing 'o' read as a zero. Stripping
# non-letters turns that into "areorthumbimpressi", which these stems still hit.
# "signatu" rather than "signat" on purpose: the form also prints "Designation",
# and "designation" contains "signat" but not "signatu".
SIG_CAPTIONS = ("signatu", "thumbimpress", "impressi")


def has_signature_caption(crop: np.ndarray) -> bool:
    """True when the crop carries a printed 'Signature / Thumb impression' line.

    The signature MODEL detects handwriting, not signatures, so on a filled
    Form 300 it fires on every pen mark: measured on real pages it returned the
    date '2026', a phone number '9483471237', a handwritten name under "First
    Name", 'M.B.A', and answers like 'No', 'NA' and 'Yes, satisfied'.

    Geometry cannot separate those from the real thing -- 'Yes, satisfied' has a
    larger maximum stroke (5222 px) than a genuine signature (3651 px), and the
    widest-stroke fraction overlaps completely. What DOES separate them is
    context: a real signature sits directly above a printed caption
    ("Signature or Thumb impression of the Life to be assured"), and none of the
    false positives do. _grow_signature_box already extends the box downward far
    enough to take that line in, so reading the crop answers the question.

    A crop that fails this test is dropped rather than guessed at: an empty
    signature.tif is more useful than one holding the word 'No'.
    """
    try:
        c = crop
        if min(c.shape[:2]) < 60:      # help OCR on small crops
            c = cv2.resize(c, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        txt = _ocr().text_of(np.stack([c] * 3, axis=-1)).lower()
        compact = "".join(ch for ch in txt if ch.isalpha())
        return any(m in compact for m in SIG_CAPTIONS)
    except Exception as exc:
        logging.debug(f"signature caption read failed: {exc}")
        return False


# How far above the caption line the signature sits, as a multiple of the
# caption's own text height. A signature and the date line above it occupy
# roughly three caption-heights; taking that much guarantees the strokes are
# whole, and the crop is trimmed back to its own ink afterwards.
SIG_ABOVE_CAPTION = 3.2

# For the caption SEARCH every one of these must appear in the same OCR line.
# Requiring both is what keeps the search precise: the page also carries an
# empty "Signature : ___" field (matches "signatu" alone) and body text about
# how "the thumb impression should be attested" (matches "impressi" alone), and
# either one on its own produced a crop containing no signature at all. Only the
# real caption -- "Signature or Thumb impression of the Life to be assured" --
# carries both. Verification of a detector hit stays looser, since there the box
# already localises actual ink.
SIG_CAPTION_STRICT = ("signatu", "impressi")

# A caption is a whole sentence, so it spans a good part of the page. Used with
# the stem test to reject short labels that merely contain the words.
SIG_CAPTION_MIN_W = 0.15


def signatures_by_caption(page_path: str, max_side: int = 1600
                          ) -> list[np.ndarray]:
    """Locate signatures from their printed caption instead of the ONNX detector.

    Used when the detector proposes nothing usable. The caption is a far steadier
    anchor than the detector: it is printed, always present, and always in the
    same place relative to the signature, whereas the detector silently dropped a
    signature after unrelated specks near it were cleaned away.

    OCR returns a box per text line, so the caption's line is found by matching
    SIG_CAPTIONS against the recognised text, and the signature is then the
    region directly ABOVE that box. The caption's own box also gives the exact
    horizontal extent to use, which is what makes the sentence come out whole.
    """
    crops: list[np.ndarray] = []
    try:
        full = cv2.imread(page_path, cv2.IMREAD_GRAYSCALE)
        if full is None:
            full = np.array(Image.open(page_path).convert("L"))
        h, w = full.shape[:2]
        s = min(1.0, max_side / max(h, w))
        small = cv2.resize(full, None, fx=s, fy=s,
                           interpolation=cv2.INTER_AREA) if s < 1 else full

        result, _ = _ocr().engine(np.stack([small] * 3, axis=-1))
        if not result:
            return crops

        right_start = w // 2
        for item in result:
            box, text = item[0], item[1]
            compact = "".join(ch for ch in str(text).lower() if ch.isalpha())
            if not all(m in compact for m in SIG_CAPTION_STRICT):
                continue
            pts = np.asarray(box, dtype=np.float32) / max(s, 1e-6)
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            if (x2 - x1) < SIG_CAPTION_MIN_W * w:
                continue          # too short to be the caption sentence
            if (x1 + x2) / 2 < right_start:
                continue          # left-hand box is the agent's, not the customer's
            ch_ = max(1.0, y2 - y1)
            top = int(max(0, y1 - ch_ * SIG_ABOVE_CAPTION))
            crop = full[top:int(min(h, y2 + ch_ * 0.35)),
                        int(max(0, x1 - 12)):int(min(w, x2 + 12))]
            if crop.size:
                crops.append(crop)
    except Exception as exc:
        logging.debug(f"caption-based signature search failed {page_path}: {exc}")
    return crops


def extract_right_signatures(page_path: str) -> list[np.ndarray]:
    """Signatures on right half only (customer box, not agent box on left).

    Each crop includes the printed caption below the signature, is verified
    against that caption (see has_signature_caption), and is taken from the
    full-resolution page so strokes stay intact.
    """
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
            x1, y1, x2, y2 = _grow_signature_box(box, full)
            img = full[y1:y2, x1:x2]
            if img.size and has_signature_caption(img):
                crops.append(img)
    except Exception as e:
        logging.debug(f"sig detect failed {page_path}: {e}")

    # The detector can silently omit a signature that is plainly present; fall
    # back to finding it by its caption rather than losing it.
    if not crops:
        crops = signatures_by_caption(page_path)
    return crops


# ---------------------------------------------------------------- photo crop (Point 4)
def _paired_jpg_path(tif_name: str, scan_dir: str) -> str | None:
    """Find the colour JPG paired with a TIF file in the original scan folder.

    The dual-stream scanner writes JPG first: TIF 00000002 pairs with JPG 00000001.
    """
    stem, _ = os.path.splitext(tif_name)
    if not stem.isdigit():
        return None
    cand = f"{int(stem) - 1:0{len(stem)}d}"
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
        p = os.path.join(scan_dir, cand + ext)
        if os.path.exists(p):
            return p
    return None


def crop_proposal_photo(tif_name: str, scan_dir: str, out_path: str) -> bool:
    """Crop the applicant photo from the top-right corner of the paired colour JPG.

    Saves as JPEG to out_path. Returns True on success.
    """
    jpg = _paired_jpg_path(tif_name, scan_dir)
    if not jpg:
        logging.debug(f"no paired JPG found for {tif_name} in {scan_dir}")
        return False
    try:
        img = Image.open(jpg)
        w, h = img.size
        left   = int(w * (1 - PHOTO_CROP_RIGHT))
        upper  = 0
        right  = w
        lower  = int(h * PHOTO_CROP_TOP)
        crop   = img.crop((left, upper, right, lower))
        crop.save(out_path, "JPEG", quality=85)
        return True
    except Exception as e:
        logging.debug(f"photo crop failed for {jpg}: {e}")
        return False


# ---------------------------------------------------------------- app number
def read_app_number(app_dir: str, review_pages: list[str]) -> str:
    """Handwritten 9-digit application number from this app's review slip.

    Read from the review slip because the number is written there by hand and
    appears in printed form nowhere in the packet (see appnumber.py). Returns ""
    when no confident reading is available -- the value is reported for a human
    to check, never used to name anything, so an empty result is harmless.
    """
    if not review_pages:
        return ""
    try:
        import appnumber
    except Exception as exc:
        logging.debug(f"appnumber unavailable: {exc}")
        return ""
    for f in review_pages:
        p = os.path.join(app_dir, f)
        try:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if g is None:
                g = np.array(Image.open(p).convert("L"))
            _crop, _hw, run = appnumber.extract(g, ocr=_ocr())
            if run and run.get("guess"):
                return str(run["guess"])
        except Exception as exc:
            logging.debug(f"app number read failed {f}: {exc}")
    return ""


# ---------------------------------------------------------------- index one app
def _save_app(app_dir: str, page_cats: dict[str, str],
              tifs: list[str], scan_dir: str | None) -> dict:
    """Given classified pages, write category TIFFs, extract signatures,
    crop applicant photo, remove individual TIFs."""

    # Pages routed to each output file, split so that categories pinned to the
    # end (LIC_slip) are appended after everything else in scan order.
    cat_paths: dict[str, list[str]] = {v: [] for v in CATEGORY_FILE.values()}
    cat_tail:  dict[str, list[str]] = {v: [] for v in CATEGORY_FILE.values()}
    proposal_seq: list[str] = []
    review_seq: list[str] = []

    for f in tifs:
        cat   = page_cats.get(f, "unidentified")
        fname = CATEGORY_FILE.get(cat, "unidentified")
        bucket = cat_tail if cat in LAST_PAGE_CATS else cat_paths
        bucket[fname].append(os.path.join(app_dir, f))
        if cat == SIG_SOURCE:
            proposal_seq.append(f)
        if cat == "Proposal_review_slip":
            review_seq.append(f)

    # Applicant photo first: it becomes PAGE 1 of the proposal TIFF, so it has to
    # exist before that file is written. Cropped from the top-right of the colour
    # JPG paired with proposal-form page 1.
    photo_path = os.path.join(app_dir, "photo.jpg")
    photo_ok = False
    if proposal_seq and scan_dir:
        photo_ok = crop_proposal_photo(proposal_seq[0], scan_dir, photo_path)

    proposal_file = CATEGORY_FILE[SIG_SOURCE]
    cat_sizes: dict[str, int] = {}
    for fname in cat_paths:
        paths = cat_paths[fname] + cat_tail[fname]
        if not paths:
            continue
        out = os.path.join(app_dir, f"{fname}.tif")
        if fname == proposal_file:
            save_proposal_tiff(paths, photo_path if photo_ok else None, out)
        else:
            save_multipage_tiff(paths, out, compression="group4")
        cat_sizes[fname] = len(paths)

    # Signatures from pages 6 & 7 of proposal form
    sig_crops: list[np.ndarray] = []
    for seq_i, f in enumerate(proposal_seq, 1):
        if seq_i not in SIG_PAGES:
            continue
        sig_crops.extend(extract_right_signatures(os.path.join(app_dir, f)))
    if sig_crops:
        save_signature_tif(sig_crops, os.path.join(app_dir, "signature.tif"))

    # Handwritten application number off the review slip (report only -- the
    # folder keeps its APP_xx name, since a single misread digit would otherwise
    # produce a confidently wrong folder name).
    app_no = read_app_number(app_dir, review_seq)

    # Remove individual page TIFs
    for f in tifs:
        try:
            os.remove(os.path.join(app_dir, f))
        except Exception:
            pass

    return {
        "pages": len(tifs),
        "categories": cat_sizes,
        "signature_crops": len(sig_crops),
        "photo_saved": photo_ok,
        "app_number": app_no,
    }


# ---------------------------------------------------------------- main
def run(apps_dir: str, workers: int | None = None) -> None:
    if workers is None:
        if cuda_available():
            # One GPU is the shared bottleneck, so extra processes only queue up
            # against it and waste VRAM on duplicate model copies. A few workers
            # are enough to keep it fed while the previous batch finishes.
            workers = 4
        else:
            # Leave 2 logical cores for the OS; cap at 16 so we don't
            # over-saturate memory bandwidth on machines with many E-cores.
            workers = max(4, min(16, (os.cpu_count() or 4) - 2))

    app_names = sorted(
        d for d in os.listdir(apps_dir)
        if os.path.isdir(os.path.join(apps_dir, d)) and d.startswith("APP_")
    )
    if not app_names:
        logging.error(f"No APP_XX folders found in {apps_dir}. Run enhance.exe first.")
        return

    # Read original scan folder from manifest for photo crop
    scan_dir: str | None = None
    manifest_path = os.path.join(apps_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                scan_dir = json.load(fh).get("input")
            if scan_dir and not os.path.isdir(scan_dir):
                logging.warning(f"manifest input dir not found: {scan_dir}")
                scan_dir = None
        except Exception:
            pass

    if scan_dir:
        logging.info(f"Photo crop: using original scan folder {scan_dir}")
    else:
        logging.warning("manifest.json not found or input dir missing — photo.jpg will be skipped")

    # Collect all pages across all apps into one flat list
    app_tifs: dict[str, list[str]] = {}
    all_jobs: list[tuple[str, str]] = []

    for app in app_names:
        app_dir = os.path.join(apps_dir, app)
        tifs = sorted(f for f in os.listdir(app_dir)
                      if f.lower().endswith((".tif", ".tiff"))
                      and not any(f == f"{cat}.tif" for cat in CATEGORY_FILE.values())
                      and f not in ("signature.tif",))
        app_tifs[app] = tifs
        for f in tifs:
            all_jobs.append((app_dir, f))

    total_pages = len(all_jobs)
    logging.info(f"Indexing {len(app_names)} apps | {total_pages} pages | {workers} workers")
    t0 = time.time()

    # True parallel OCR via ProcessPoolExecutor (bypasses the GIL — each
    # worker process runs its own ONNX session independently).
    # On Windows the 'spawn' start method is used; freeze_support() in the
    # entry point ensures frozen-EXE subprocesses are handled correctly.
    page_cats: dict[tuple[str, str], str] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for app_dir, f, cat, _ in ex.map(_classify_job, all_jobs):
            page_cats[(app_dir, f)] = cat
            done += 1
            if done % 50 == 0 or done == total_pages:
                el = time.time() - t0
                logging.info(f"  [{done}/{total_pages}] {el:.0f}s  {el/done*1000:.0f} ms/page")

    # Assemble results per app
    summary = []
    rows = []
    for app in app_names:
        app_dir = os.path.join(apps_dir, app)
        tifs = app_tifs[app]
        per_page = {f: page_cats.get((app_dir, f), "unidentified") for f in tifs}
        result = _save_app(app_dir, per_page, tifs, scan_dir)
        summary.append({"app": app, **result})
        app_no = result.get("app_number", "")
        for cat_name, n in result["categories"].items():
            rows.append([app, app_no, cat_name, n])
        photo_str = "photo=yes" if result["photo_saved"] else "photo=no"
        logging.info(f"  {app}: {result['pages']} pages -> "
                     f"{list(result['categories'])} sigs={result['signature_crops']} "
                     f"{photo_str} app_no={app_no or '-'}")

    el = time.time() - t0
    filed = sum(s["pages"] for s in summary)

    with open(os.path.join(apps_dir, "index_report.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["application", "app_number", "category", "pages"])
        w.writerows(rows)

    with open(os.path.join(apps_dir, "index_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"applications": len(app_names),
                   "seconds": round(el, 1),
                   "detail": summary}, fh, indent=1)

    logging.info("=" * 64)
    logging.info(f"applications={len(app_names)}  total_pages_filed={filed}")
    logging.info(f"time={el:.0f}s  ({el/max(1,filed)*1000:.0f} ms/page)")
    logging.info(f"output: {apps_dir}")
    logging.info("=" * 64)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    import argparse
    ap = argparse.ArgumentParser(
        description="Index APP_XX folders into per-category multi-page TIFFs")
    ap.add_argument("--apps", required=True,
                    help="folder containing APP_01/, APP_02/... (output of enhance.exe)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    run(a.apps, a.workers)
