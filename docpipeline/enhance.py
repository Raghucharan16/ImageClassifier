"""
Stage 1 - Image enhancement for scanned LIC application batches.

Input: a scan folder where each PHYSICAL page produced two files by a
dual-stream scanner:
    00000001.jpg  <- colour, ~150 dpi  (kept for the applicant-photo step)
    00000002.tif  <- bitonal G4, ~200 dpi (the document image we enhance)
so odd files are JPGs and even files are the matching TIFs of the SAME page.

Output: a clean, upright, black-&-white Group4 TIFF per page.

Pipeline (order matters, and each step is here because the sample batches
actually contain the artefact):
  1. one connected-component analysis, shared by steps 2-4 (5 separate passes
     over a 2000x2500 page was the single biggest cost)
  2. tractor-feed holes  - periodic solid ovals punched down a margin
  3. scanner background  - solid black wedges/bars where a skewed page missed
                           the platen
  4. dark patches        - solid toner/ink blotches anywhere on the page
  5. blank detection     - skip empty page-backs (measured from component
                           sizes, so leftover dust cannot fake content)
  6. page orientation    - 0/90/180/270 via a bundled ~7MB PP-LCNet ONNX
                           classifier (rapid-orientation): 40/40 correct on
                           injected-rotation tests at ~25ms, no OCR needed.
                           Runs BEFORE despeckling, which the classifier is
                           sensitive to.
  7. outside-content     - dots and black lines in the white margin beyond the
                           document's own bounding box
  8. roller streaks      - the dashed vertical lines a dirty roller leaves
  9. margin lines        - thin horizontal streaks in top/bottom 12% of page
 10. despeckle           - isolated dots and thin dashes (scanner dust)
 11. fine deskew         - residual tilt, projection-profile search
                           (0.034 deg mean error, hierarchical for speed); pads
                           with white, so it never cuts content
 12. punch-hole crop     - trims ONLY the left/right strips that held tractor-feed
                           holes, and only when such a column was detected. This
                           is the pipeline's only crop: a page with clean margins
                           keeps its scanned dimensions.
 13. Group4 save

Every noise filter runs on every page. Nothing here is skipped by page type:
these artefacts are all scanner output, never document content. The streak
remover is safe even on monospaced dot-matrix print thanks to its
alone_on_its_rows test -- see remove_streaks.

No OCR and no heavy filters, so a page stays far inside the 1s/page budget.
"""
from __future__ import annotations

import os
import logging
import threading

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DOC_EXT = (".tif", ".tiff")
PHOTO_EXT = (".jpg", ".jpeg")
OUT_DPI = 200


# ------------------------------------------------------------------- io
def load_bitonal(path: str) -> np.ndarray | None:
    """Load a page as uint8 grayscale (0=ink, 255=paper)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        try:
            img = np.array(Image.open(path).convert("L"))
        except Exception as e:
            logging.error(f"cannot read {path}: {e}")
            return None
    return img


def save_group4(gray: np.ndarray, out_path: str, dpi: int = OUT_DPI) -> int:
    """Write a bitonal (black & white) Group4-compressed TIFF."""
    bw = gray if gray.dtype == bool else (gray > 127)
    Image.fromarray((bw * 255).astype(np.uint8)).convert("1").save(
        out_path, format="TIFF", compression="group4", dpi=(dpi, dpi))
    return os.path.getsize(out_path)


def _ink(gray: np.ndarray) -> np.ndarray:
    return (gray < 128).astype(np.uint8) * 255


def _erase(gray: np.ndarray, lab: np.ndarray, ids, n: int) -> np.ndarray:
    """Erase a set of component ids in one vectorised pass (LUT over labels)."""
    if not len(ids):
        return gray
    lut = np.zeros(n, dtype=bool)
    lut[np.fromiter(ids, dtype=np.int64, count=len(ids))] = True
    out = gray.copy()
    out[lut[lab]] = 255
    return out


# --------------------------------------------------------- page orientation
_ori_lock = threading.Lock()
_ori_tls = threading.local()


_ORI_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "orientation_config.yaml")


def _orientation_model():
    """One model instance per worker thread, each capped to a single ONNX
    thread (see orientation_config.yaml) so our worker pool -- not onnxruntime
    -- owns the parallelism."""
    m = getattr(_ori_tls, "m", None)
    if m is None:
        from rapid_orientation import RapidOrientation
        with _ori_lock:                    # model file read is not thread-safe
            if os.path.exists(_ORI_CFG):
                m = RapidOrientation(cfg_path=_ORI_CFG)
            else:
                m = RapidOrientation()
            _ori_tls.m = m
    return m


def text_runs_vertically(gray: np.ndarray, max_side: int = 600) -> bool:
    """True when the text lines run down the page, i.e. the page is sideways.

    Text lines create periodic banding perpendicular to themselves, so we
    compare how 'peaky' the row profile is against the column profile. This is
    an independent, purely geometric check and was 100% correct on sideways
    pages in testing -- used to confirm the model's 90/270 calls.
    """
    h, w = gray.shape[:2]
    s = max_side / max(h, w)
    sm = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) \
        if s < 1 else gray
    bw = (sm < 128).astype(np.float32)
    row_var = float(np.var(np.diff(bw.sum(axis=1))))
    col_var = float(np.var(np.diff(bw.sum(axis=0))))
    return col_var > row_var


def _orientation_probs(gray: np.ndarray, max_side: int = 1000):
    """(label, probability, margin) from the orientation classifier."""
    m = _orientation_model()
    h, w = gray.shape[:2]
    s = max_side / max(h, w)
    sm = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) \
        if s < 1 else gray
    bgr = cv2.cvtColor(sm, cv2.COLOR_GRAY2BGR)
    out = m.session(m.preprocess(m.load_img(bgr)))[0]
    mean = out.mean(axis=0)
    order = np.argsort(-mean)
    return int(m.labels[order[0]]), float(mean[order[0]]), float(mean[order[0]] - mean[order[1]])


def detect_page_rotation(gray: np.ndarray, flip_prob: float = 0.85,
                         flip_margin: float = 0.70) -> int:
    """Rotation (0/90/180/270) to undo so the page reads upright.

    Deliberately asymmetric, because the two cases are not equally hard and a
    wrong rotation is worse than leaving a page alone:

      * 90/270 (sideways) - accepted whenever an independent geometric check
        agrees the text really does run vertically. The model is only trusted
        to say WHICH way to turn, not WHETHER the page is sideways. Confidence
        here is often modest (0.4-0.9) yet correct, so gating on probability
        alone would wrongly skip most genuinely sideways pages.
      * 180 (upside-down) - no geometric cross-check exists, so this requires
        high confidence. Observed: a true flip scored p=0.89/margin=0.85 while
        the two pages it got WRONG scored 0.75/0.56 and 0.46/0.09.

    Anything else, or any low-confidence call, is left unrotated.
    """
    try:
        lab, prob, margin = _orientation_probs(gray)
    except Exception as e:                   # never fail a page over orientation
        logging.debug(f"orientation model failed: {e}")
        return 0

    if lab in (90, 270):
        return lab if text_runs_vertically(gray) else 0
    if lab == 180:
        return 180 if (prob >= flip_prob and margin >= flip_margin) else 0
    return 0


def apply_rotation(gray: np.ndarray, deg: int) -> np.ndarray:
    """Lossless 90-degree-family rotation (transpose/flip, no interpolation)."""
    if deg == 90:
        return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if deg == 180:
        return cv2.rotate(gray, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    return gray


# ------------------------------------------- artefact removal (shared CC pass)
def clean_artefacts(gray: np.ndarray,
                    edge_frac: float = 0.08, hole_min: int = 15, hole_max: int = 70,
                    hole_count: int = 5, hole_gap_cv: float = 0.6,
                    border_area: float = 0.0012, patch_area: float = 0.0008,
                    patch_area_max: float = 0.0015, patch_dim_max: int = 90
                    ) -> tuple[np.ndarray, dict]:
    """Remove feed holes, scanner-background bars/wedges and solid dark patches
    using a single connected-component analysis.

    Discriminators (all calibrated on the sample scans):
      * feed holes  - filled ovals >=15px, tightly x-aligned in an outer margin,
        with a REGULAR vertical pitch and at least 5 of them. Periodicity plus
        alignment is what stops this eating text: an earlier size-only version
        deleted the first letter of every line in the left margin.
      * background  - touches the image border and survives erosion, i.e. it is
        a solid mass. Glyph strokes and table rules are only a few px wide and
        erode away, so content is safe (measured fill: wedge 0.34 vs grid 0.03).
      * dark patches - the same solid-mass test applied away from the border,
        but ONLY for SMALL blobs (<= patch_area_max of the page and
        <= patch_dim_max px per side). That size cap is essential: passport
        photos (280x352 px, ~1.9% of the page) and Aadhaar QR blocks
        (~0.3-0.7%) are solid dark masses too, and an uncapped version deleted
        an applicant's photograph. Only genuine specks/blotches are removed;
        photos and QR codes are preserved.
    """
    h, w = gray.shape[:2]
    page = h * w
    ink = _ink(gray)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(ink, connectivity=8)
    info = {"holes_removed": 0, "patches_removed": 0}
    if n <= 1:
        return gray, info

    areas = stats[:, cv2.CC_STAT_AREA]
    ws = stats[:, cv2.CC_STAT_WIDTH]
    hs = stats[:, cv2.CC_STAT_HEIGHT]
    xs = stats[:, cv2.CC_STAT_LEFT]
    ys = stats[:, cv2.CC_STAT_TOP]
    kill: set[int] = set()

    # ---- feed holes: periodic filled ovals in an outer vertical margin
    band = max(1, int(w * edge_frac))
    oval = ((areas >= 0.60 * ws * hs) & (ws >= hole_min) & (hs >= hole_min)
            & (ws <= hole_max) & (hs <= hole_max))
    oval[0] = False
    for lo, hi in ((0, band), (w - band, w)):
        sel = np.where(oval & (cents[:, 0] >= lo) & (cents[:, 0] < hi))[0]
        if sel.size < hole_count:
            continue
        cx, cy = cents[sel, 0], cents[sel, 1]
        if cx.std() > 0.025 * w:                    # not a straight punched column
            continue
        gaps = np.diff(np.sort(cy))
        if gaps.size and gaps.mean() > 0 and gaps.std() / gaps.mean() > hole_gap_cv:
            continue                                # irregular -> text, not holes
        kill.update(sel.tolist())
        info["holes_removed"] += int(sel.size)
        # Record where the punched column sits. This is the ONLY thing that
        # licenses a crop later (see crop_hole_bands): pages without punch holes
        # are left at their scanned size.
        info.setdefault("hole_bands", []).append({
            "side": "left" if lo == 0 else "right",
            "x0": int(xs[sel].min()),
            "x1": int((xs[sel] + ws[sel]).max()),
        })

    # ---- solid masses: scanner background (border-touching) and dark patches
    kb = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    kp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    big = np.where(areas >= min(border_area, patch_area) * page)[0]
    for i in big:
        if i == 0 or i in kill:
            continue
        touches = (xs[i] <= 2 or ys[i] <= 2
                   or xs[i] + ws[i] >= w - 2 or ys[i] + hs[i] >= h - 2)
        if touches:
            if areas[i] < border_area * page:
                continue
            k, keep = kb, 0.15
        else:
            # interior blobs: only SMALL solid specks, never photo/QR-sized ones
            if areas[i] < patch_area * page or areas[i] > patch_area_max * page:
                continue
            if ws[i] > patch_dim_max or hs[i] > patch_dim_max:
                continue
            if areas[i] < 0.55 * ws[i] * hs[i]:
                continue
            k, keep = kp, 0.30
        sub = (lab[ys[i]:ys[i] + hs[i], xs[i]:xs[i] + ws[i]] == i).astype(np.uint8) * 255
        if cv2.erode(sub, k).sum() >= keep * sub.sum():
            kill.add(int(i))
            if not touches:
                info["patches_removed"] += 1

    out = _erase(gray, lab, kill, n)
    # content ink = ink in components big enough to be real (dust-proof blank test)
    live = np.ones(n, dtype=bool)
    live[0] = False
    for i in kill:
        live[i] = False
    info["content_ink_pct"] = float(areas[live & (areas >= 15)].sum()) / page * 100
    return out, info


# ------------------------------------------- noise outside the content region
def content_bbox(gray: np.ndarray, frac: float = 0.010,
                 pad_frac: float = 0.015):
    """Bounding box of the page's real content, ignoring stray noise.

    A row belonging to the document carries ink across a meaningful part of the
    width, while a row holding only scanner dirt carries a handful of pixels.
    Requiring `frac` of the dimension (1%, i.e. ~17 px on a 1700 px-wide page)
    therefore brackets the text block and excludes speckled margins -- the
    largest confirmed noise dot is ~10 px across, so a row of them cannot reach
    the threshold. Deliberately stricter than crop_to_content's 0.2%, whose job
    is to keep everything, whereas this box is used to decide what is OUTSIDE
    the document and safe to erase.

    Returns (x0, y0, x1, y1) padded outward by pad_frac, or None when the page
    has no discernible content.
    """
    h, w = gray.shape[:2]
    ink = gray < 128
    rows = np.where(ink.sum(axis=1) > frac * w)[0]
    cols = np.where(ink.sum(axis=0) > frac * h)[0]
    if rows.size == 0 or cols.size == 0:
        return None
    py, px = int(h * pad_frac), int(w * pad_frac)
    return (max(0, int(cols[0]) - px), max(0, int(rows[0]) - py),
            min(w, int(cols[-1]) + px + 1), min(h, int(rows[-1]) + py + 1))


def remove_outside_content(gray: np.ndarray, dot_area: int = 400,
                           thin_px: int = 16, keep_area: int = 12000
                           ) -> tuple[np.ndarray, int]:
    """Erase dots and black lines lying in the white margin outside the content.

    Anything wholly outside the content box is, by construction, not part of the
    document -- so this pass can be far more aggressive than despeckle(), which
    has to work amid text and is therefore limited to isolated specks of <=80 px.
    Out here the two artefacts the scanner leaves are handled directly:

      * dots / blobs  - up to `dot_area` px (5x despeckle's limit), since there
        is no neighbouring text to confuse them with.
      * black lines   - any component thinner than `thin_px` in one dimension,
        at any length. Length has to be unbounded because the streaks and edge
        bars that survive into the margin run for hundreds of pixels; thinness
        is what marks them as a line rather than an object.

    `keep_area` protects a genuinely large, solid object that happens to sit in
    the margin (a photo or stamp overhanging the text block): a chunky mass that
    big is content, not dirt, so it is left alone even out here.

    Components straddling the boundary are never touched -- only those entirely
    outside it -- so a descender or table rule reaching into the margin is safe.
    """
    box = content_bbox(gray)
    if box is None:
        return gray, 0
    x0, y0, x1, y1 = box

    ink = _ink(gray)
    n, lab, stats, _cents = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return gray, 0

    xs = stats[:, cv2.CC_STAT_LEFT]
    ys = stats[:, cv2.CC_STAT_TOP]
    ws = stats[:, cv2.CC_STAT_WIDTH]
    hs = stats[:, cv2.CC_STAT_HEIGHT]
    areas = stats[:, cv2.CC_STAT_AREA]

    # entirely outside the content box
    outside = ((xs + ws <= x0) | (xs >= x1) | (ys + hs <= y0) | (ys >= y1))
    is_dot = areas <= dot_area
    is_line = np.minimum(ws, hs) <= thin_px
    chunky = (areas > keep_area) & (np.minimum(ws, hs) > thin_px)

    kill = np.where(outside & (is_dot | is_line) & ~chunky)[0]
    kill = kill[kill != 0]
    if not kill.size:
        return gray, 0
    return _erase(gray, lab, kill.tolist(), n), int(kill.size)


# ------------------------------------------------- page type (gentle vs full)
def component_density(gray: np.ndarray, min_area: int = 15,
                      min_h: int = 6) -> float:
    """Real ink components per megapixel -- how densely printed the page is."""
    h, w = gray.shape[:2]
    mp = (h * w) / 1e6
    if mp <= 0:
        return 0.0
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(_ink(gray), connectivity=8)
    if n <= 1:
        return 0.0
    areas = stats[:, cv2.CC_STAT_AREA]
    hs = stats[:, cv2.CC_STAT_HEIGHT]
    real = np.where((areas >= min_area) & (hs >= min_h))[0]
    real = real[real != 0]
    return float(real.size) / mp


# Measured on a labelled sample of the real batches (components per megapixel):
#     proposal forms   474 - 873
#     review slips     536 - 559
#     enclosures       441 - 537
#     bank sheets      583 - 644
#     LIC policy slip  421
#     ID card scans    136 - 146
# Printed document pages cluster far above the ID cards, with an empty gap from
# 146 to 421, so this threshold sits in the middle of that gap.
DOC_DENSITY = 280.0


def is_document_page(gray: np.ndarray) -> bool:
    """True for a densely printed document page (form, review slip, enclosure).

    These pages get the GENTLE treatment: their content is the whole point of
    the scan and every aggressive filter risks eating it -- the streak remover
    once deleted 201 real glyphs from a single review slip, because dot-matrix
    print is monospaced and its thin characters line up into columns that look
    exactly like a dashed roller streak. Sparse pages (photocopied ID cards) are
    the opposite case: little content, lots of scanner noise, and they genuinely
    need the full clean-up plus the rebuild-from-JPG path.
    """
    return component_density(gray) >= DOC_DENSITY


def crop_hole_bands(gray: np.ndarray, bands, pad: int = 6,
                    max_frac: float = 0.14) -> tuple[np.ndarray, bool]:
    """Trim off the left/right strips that held tractor-feed punch holes.

    This is the ONLY cropping the pipeline does. Pages with no punched column
    come back untouched at their scanned size -- earlier versions ran
    crop-to-content on everything, which reflowed pages that had nothing wrong
    with their margins.

    The punch holes themselves are already erased by clean_artefacts; what is
    left is an empty strip down the edge, and this removes it. Only the strip is
    taken: the cut lands just inside the hole column (pad px past it), never into
    the text block. `max_frac` caps each side at 14% of the width so a bad hole
    detection cannot eat a real margin -- a genuine punched column sits within a
    few percent of the edge.

    Returns (image, cropped?).
    """
    if not bands:
        return gray, False
    h, w = gray.shape[:2]
    x0, x1 = 0, w
    limit = int(w * max_frac)
    for b in bands:
        if b["side"] == "left":
            cut = min(int(b["x1"]) + pad, limit)
            x0 = max(x0, cut)
        else:
            cut = max(int(b["x0"]) - pad, w - limit)
            x1 = min(x1, cut)
    if x1 - x0 < 0.5 * w:      # refuse an implausible cut
        return gray, False
    return gray[:, x0:x1], True


# ---------------------------------------------------------- scanner streaks
def remove_streaks(gray: np.ndarray, max_width: int = 14, max_h_frac: float = 0.15,
                   x_tol: float = 15.0, min_span: float = 0.10, min_blobs: int = 4,
                   bin_w: int = 28, solo_span: float = 0.45,
                   row_neighbour: float = 0.0004) -> tuple[np.ndarray, int]:
    """Remove the dashed vertical lines a dirty scanner roller leaves down a page.

    These survive despeckling because the dashes sit close together, so each one
    has neighbours and none looks isolated. What identifies them is geometry:
    they are THIN (a few px wide) fragments stacked in a column that shares
    almost the same x (measured x_std ~3.5 px) and spans much of the page.
    Selecting on thin-ness rather than area matters -- the dashes run from a few
    px up to ~700 px in area and ~200 px tall, so an area cap missed the longer
    ones. Text is laid out in horizontal lines and cannot form a tight column
    that tall, and a real (solid) table rule is a single component rather than
    `min_blobs` separate fragments, so both stay safe.
    """
    h, w = gray.shape[:2]
    ink = _ink(gray)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return gray, 0
    widths = stats[:, cv2.CC_STAT_WIDTH]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    tops = stats[:, cv2.CC_STAT_TOP]
    areas = stats[:, cv2.CC_STAT_AREA]
    row_ink = (ink > 0).sum(axis=1).astype(np.float64)

    def alone_on_its_rows(i: int) -> bool:
        """True when the component's own rows carry almost no other ink.

        This is the test that separates a roller streak from text, and it does
        the heavy lifting: a thin glyph always sits in a text line, so its rows
        are full of other ink, whereas a streak fragment lies on otherwise empty
        rows. It is essential because this print is MONOSPACED -- thin characters
        line up in vertical columns across many rows and look exactly like a
        dashed streak geometrically (an earlier version deleted 201 real glyphs
        from one review slip). Measured separation is wide and clean: streak
        fragments score 0.0000 here while the lowest real glyph scored 0.0005,
        which is why the column tests below can then be kept loose.
        """
        y0, y1 = tops[i], tops[i] + heights[i]
        other = float(row_ink[y0:y1].sum()) - float(areas[i])
        return other <= row_neighbour * w * max(1, y1 - y0)

    kill: set[int] = set()

    # a single thin sliver running down most of the page is a streak by itself
    for i in np.where((widths <= max_width) & (heights >= solo_span * h))[0]:
        if i != 0:
            kill.add(int(i))

    thin = np.where((widths <= max_width) & (heights <= max_h_frac * h)
                    & (heights > 0))[0]
    thin = np.array([i for i in thin if i != 0 and alone_on_its_rows(i)])
    if thin.size >= min_blobs:
        buckets: dict[int, list[int]] = {}
        for idx in thin:
            buckets.setdefault(int(cents[idx, 0]) // bin_w, []).append(int(idx))
        for b, ids in buckets.items():
            group = ids + buckets.get(b + 1, [])
            if len(group) < min_blobs:
                continue
            gx, gy = cents[group, 0], cents[group, 1]
            if gx.std() > x_tol:
                continue
            if (gy.max() - gy.min()) < min_span * h:
                continue
            kill.update(group)

    if not kill:
        return gray, 0
    return _erase(gray, lab, sorted(kill), n), len(kill)


# -------------------------------------------- remove horizontal margin lines
def remove_margin_lines(gray: np.ndarray,
                        margin_frac: float = 0.12,
                        min_span: float = 0.35,
                        max_height: int = 6) -> tuple[np.ndarray, int]:
    """Remove thin horizontal lines in the top/bottom margins from scanner feed.

    These appear as 1–5 px tall horizontal streaks spanning 35–100% of the
    page width, located only in the first/last 12% of the page height. Real
    content (ruled lines, boxes) is located in the central 76% of the page,
    so restricting to the outer margins makes this very safe.
    """
    h, w = gray.shape[:2]
    margin_rows = int(h * margin_frac)
    ink = _ink(gray)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return gray, 0

    tops    = stats[:, cv2.CC_STAT_TOP]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    widths  = stats[:, cv2.CC_STAT_WIDTH]

    in_margin = (
        ((tops < margin_rows) | (tops + heights > h - margin_rows))
        & (heights <= max_height)
        & (widths >= min_span * w)
    )
    in_margin[0] = False
    kill = np.where(in_margin)[0].tolist()
    if not kill:
        return gray, 0
    return _erase(gray, lab, kill, n), len(kill)


# --------------------------------------------------------------- despeckle
def despeckle(gray: np.ndarray, win: int = 50,
              min_neighbour: float = 0.020) -> np.ndarray:
    """Remove isolated scanner dust while keeping all text and handwriting.

    Two candidate shapes are targeted (measured on real scans):
      • Dots / blobs  : max(w, h) <= 12, area <= 80
      • Thin dashes   : min(w, h) <= 3, max(w, h) <= 45, area <= 120
        (short streaks from the scanner transport mechanism)

    A candidate is only erased when its neighbourhood (win × win window) has
    fewer than min_neighbour × win² ink pixels excluding itself.  That means
    a speck surrounded by empty paper is erased; a dot that is part of text
    or a signature is left alone (it always has companion ink nearby).

    Thresholds calibrated on this batch:
      - Smallest real glyph stroke:  3 × 14 px, area 40  → nearby ink ~150
      - Largest confirmed noise dot:  7 × 10 px, area  39 → nearby ink   0
      - Largest confirmed noise dash: 5 × 13 px, area  52 → nearby ink  18
      - Safe nearby-ink threshold:   min_neighbour × win² = 0.020 × 2500 = 50
    """
    ink = _ink(gray)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return gray

    areas = stats[:, cv2.CC_STAT_AREA]
    Ws    = stats[:, cv2.CC_STAT_WIDTH]
    Hs    = stats[:, cv2.CC_STAT_HEIGHT]
    maxWH = np.maximum(Ws, Hs)
    minWH = np.minimum(Ws, Hs)

    is_dot  = (areas <= 80)  & (maxWH <= 12)
    is_dash = (areas <= 120) & (minWH <= 3) & (maxWH <= 45)
    cand = is_dot | is_dash
    cand[0] = False   # never touch background label

    sel = np.where(cand)[0]
    if sel.size == 0:
        return gray

    dens = cv2.boxFilter((ink > 0).astype(np.float32), -1, (win, win),
                         normalize=True)
    cy = np.clip(cents[sel, 1].astype(int), 0, gray.shape[0] - 1)
    cx = np.clip(cents[sel, 0].astype(int), 0, gray.shape[1] - 1)
    nearby = dens[cy, cx] * (win * win) - areas[sel]
    kill = sel[nearby < min_neighbour * win * win].tolist()
    return _erase(gray, lab, kill, n)


# ------------------------------------------------------------------ deskew
def find_skew(gray: np.ndarray, max_side: int = 700, limit: float = 15.0,
              min_ink_pct: float = 0.25, min_gain: float = 1.15) -> float:
    """Angle (deg) to apply to straighten the page.

    Maximises the variance of the derivative of the horizontal ink profile:
    with text lines exactly horizontal the profile alternates sharply between
    line and gap. Searched hierarchically (2 deg -> 0.5 -> 0.1) which cuts the
    number of rotations by ~3x versus a flat sweep for the same 0.03 deg
    resolution.

    Returns 0.0 when no reliable estimate is possible: too little ink (a blank
    page has no text lines and the search then runs away to an arbitrary large
    angle -- it once "corrected" a blank page by -17.5 deg), or when the best
    angle fails to beat upright by `min_gain`.
    """
    if (gray < 128).mean() * 100 < min_ink_pct:
        return 0.0
    h, w = gray.shape[:2]
    s = max_side / max(h, w)
    small = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) \
        if s < 1 else gray
    bw = (small < 128).astype(np.float32)
    hh, ww = bw.shape
    cx, cy = ww / 2, hh / 2

    def score(a: float) -> float:
        if a == 0:
            r = bw
        else:
            M = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
            r = cv2.warpAffine(bw, M, (ww, hh), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(np.diff(r.sum(axis=1))))

    base = score(0.0)
    best = 0.0
    for step, span in ((2.0, limit), (0.5, 2.0), (0.1, 0.5)):
        lo, hi = best - span, best + span
        best = max(np.arange(lo, hi + 1e-9, step), key=score)
    return float(best) if score(best) >= min_gain * base else 0.0


def rotate_keep_page(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the centre, expanding the canvas, padding with paper white."""
    if abs(angle) < 0.05:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


# -------------------------------------- dark ID-card pages: rebuild from JPG
def paired_photo_path(doc_path: str, photo_dir: str | None = None) -> str | None:
    """The colour JPG of the same physical page.

    The dual-stream scanner writes the JPG first, so TIF `000000NN` pairs with
    JPG `000000NN-1` (verified across the sample batches).

    photo_dir: where to look for the JPG. Needed because scan.exe copies only
    TIFs into its APP_xx folders, leaving the JPGs behind in the original scan
    folder -- so once enhancement reads from scan.exe's output, the JPG is no
    longer beside its TIF. Defaults to the TIF's own directory.
    """
    d, name = os.path.split(doc_path)
    stem, _ = os.path.splitext(name)
    if not stem.isdigit():
        return None
    cand = f"{int(stem) - 1:0{len(stem)}d}"
    for base in (photo_dir, d):
        if not base:
            continue
        for ext in PHOTO_EXT:
            p = os.path.join(base, cand + ext)
            if os.path.exists(p):
                return p
    return None


def rebuild_from_photo(jpg_path: str) -> np.ndarray | None:
    """Re-binarise a page from its colour JPG.

    Needed for photocopied ID cards (Aadhaar/PAN): the scanner's own bitonal
    TIF of a dark photocopy comes out ~67% black and completely unreadable,
    because a single global threshold cannot cope with the uneven tone. The JPG
    still holds real grey levels, so local contrast equalisation plus an
    ADAPTIVE threshold recovers a legible card (measured 14% ink, ~58 KB and
    every field readable, versus an unreadable 491 KB from the TIF).
    """
    img = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    bw = cv2.adaptiveThreshold(cv2.GaussianBlur(eq, (3, 3), 0), 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11)
    return cv2.medianBlur(bw, 3)


# -------------------------------------------------------------------- crop
def crop_to_content(gray: np.ndarray, pad: int = 20,
                    min_ink_frac: float = 0.002) -> np.ndarray:
    """Trim surrounding whitespace, keeping a small margin."""
    ink = gray < 128
    h, w = ink.shape
    rows = np.where(ink.sum(axis=1) > min_ink_frac * w)[0]
    cols = np.where(ink.sum(axis=0) > min_ink_frac * h)[0]
    if rows.size == 0 or cols.size == 0:
        return gray
    y0, y1 = max(0, rows[0] - pad), min(h, rows[-1] + pad + 1)
    x0, x1 = max(0, cols[0] - pad), min(w, cols[-1] + pad + 1)
    if y1 - y0 < 20 or x1 - x0 < 20:
        return gray
    return gray[y0:y1, x0:x1]


# ---------------------------------------------------------------- pipeline
def enhance_page(path: str, blank_ink_pct: float = 0.35,
                 dark_pct: float = 25.0,
                 photo_dir: str | None = None) -> tuple[np.ndarray | None, dict]:
    """Full clean-up for one scanned page. Returns (image or None, info).

    photo_dir: folder holding the paired colour JPGs (the original scan folder),
    used to rebuild dark ID-card photocopies. See paired_photo_path.
    """
    info: dict = {"file": os.path.basename(path)}
    gray = load_bitonal(path)
    if gray is None:
        info["status"] = "error"
        return None, info
    info["in_shape"] = gray.shape

    g, cinfo = clean_artefacts(gray)
    info.update(cinfo)

    # A page still mostly black after artefact removal means the scanner's
    # bitonal threshold failed (typically a dark ID-card photocopy). Rebuild it
    # from the colour JPG of the same page, which still has real grey levels.
    if cinfo.get("content_ink_pct", 0.0) >= dark_pct:
        jpg = paired_photo_path(path, photo_dir)
        rebuilt = rebuild_from_photo(jpg) if jpg else None
        if rebuilt is not None:
            g, cinfo = clean_artefacts(rebuilt)
            info.update(cinfo)
            info["rebuilt_from"] = os.path.basename(jpg)

    # Blank test uses ink in real-sized components, so leftover dust cannot
    # fake content, and it runs before any rotation work (a blank page has no
    # text lines to orient or deskew by).
    if cinfo.get("content_ink_pct", 0.0) < blank_ink_pct:
        info["status"] = "blank"
        return None, info

    rot = detect_page_rotation(g)          # before despeckle: classifier is
    info["rotation"] = rot                 # sensitive to fine dot structure
    g = apply_rotation(g, rot)

    # --- noise removal: runs on EVERY page ---------------------------------
    # These are the black dots and black lines the scanner adds, and none of them
    # is ever content, so no page is exempt. remove_streaks is safe even on
    # monospaced dot-matrix print because of its alone_on_its_rows test: streak
    # fragments sit on otherwise-empty rows (measured 0.0000) while the thinnest
    # real glyph still shares its rows with the rest of its text line (0.0005).
    g, outside = remove_outside_content(g)
    info["outside_noise_removed"] = outside
    g, streak = remove_streaks(g)
    info["streak_blobs_removed"] = streak
    g, margin_lines = remove_margin_lines(g)
    info["margin_lines_removed"] = margin_lines
    g = despeckle(g)

    # --- straighten -------------------------------------------------------
    # Deskew pads with white rather than cutting, so it costs no content.
    angle = find_skew(g)
    info["skew"] = round(angle, 2)
    g = rotate_keep_page(g, angle)

    # --- crop: punch-hole strips ONLY -------------------------------------
    # Nothing else is cropped. A page with no punched column keeps its scanned
    # dimensions.
    g, cropped = crop_hole_bands(g, cinfo.get("hole_bands"))
    info["hole_crop"] = cropped

    info["status"] = "ok"
    info["out_shape"] = g.shape
    return g, info
