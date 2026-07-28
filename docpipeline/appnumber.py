"""
Find the handwritten 9-digit application number on a review slip.

The key idea (and the thing that makes this work) is to look ONLY at
handwriting and never at printed text. Printed dot-matrix characters on these
slips are tightly clustered around 19 px tall (p90 = 21, p95 = 27), while the
ballpoint digits measure 43-56 px -- roughly 2.5x taller -- and their strokes
are thicker. Selecting components that are both notably taller than the page's
typical print AND above the page's stroke-thickness distribution strips the
printed body text away almost completely and leaves the pen writing standing
alone.

That matters because every attempt to read the number *with* the print present
failed: printed-text OCR returns the surrounding address instead of the digits
(a tight crop of a slip whose number is plainly 765208417 came back as
'AKIHI HILAYAN 1H80 ATE'), and searching page text for a 9-digit run matches
PRINTED fields such as the Agency Code (027507631) or Policy No (000000000) --
which would name folders confidently wrong. The number also appears nowhere in
printed form on any page of the application, so isolating the handwriting is
the only route.

What survives alongside the digits is a small, predictable set: signatures,
the bold pre-printed LIC banner, and leftover scanner streaks. Those are
separated from the number by shape and grouping, not by position, since the
number's location varies from slip to slip.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_DIGITS = 6          # allow a couple of digits to merge or drop out
MAX_DIGITS = 13
WANT_DIGITS = 9


def handwriting_mask(gray: np.ndarray, height_mult: float = 1.5,
                     thick_pct: float = 45.0, min_area: int = 20):
    """Ink that is taller and thicker than the page's printed text.

    Returns (mask, labels, stats, centroids, selected_ids). Thresholds are
    derived from THIS page, so the method adapts to scan resolution and print
    density instead of relying on fixed pixel sizes.
    """
    ink = (gray < 128).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(ink * 255, connectivity=8)
    if n <= 1:
        return np.zeros_like(ink), lab, stats, cents, []

    H = stats[:, cv2.CC_STAT_HEIGHT]
    W = stats[:, cv2.CC_STAT_WIDTH]
    A = stats[:, cv2.CC_STAT_AREA]
    real = np.where((A >= min_area) & (H >= 6))[0]
    real = real[real != 0]
    if real.size == 0:
        return np.zeros_like(ink), lab, stats, cents, []

    med_h = float(np.median(H[real]))
    thickness = A[real] / np.maximum(1, W[real] + H[real])
    # Height is the strong signal (pen digits are ~2.5x the print height);
    # thickness is only a secondary filter and must stay generous, because a
    # handwritten '0' is a thin LOOP, not a solid stroke. At the 70th
    # percentile a real '0' scored 4.31 against a 4.33 cut-off and was dropped,
    # silently turning 765208418 into '7652 8418'.
    thr_thick = float(np.percentile(thickness, thick_pct))

    sel = [int(i) for i in real
           if H[i] >= height_mult * med_h
           and (A[i] / max(1, W[i] + H[i])) >= thr_thick]

    mask = np.zeros_like(ink)
    for i in sel:
        mask[lab == i] = 1
    return mask, lab, stats, cents, sel


def _digit_shaped(stats, i) -> bool:
    """A handwritten digit is roughly upright: taller than wide, or squarish."""
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    if h <= 0:
        return False
    ar = w / h
    return 0.10 <= ar <= 1.8


def _repair_chain(chain, stats, cents, n_all, st_all, ce_all):
    """Trim strays from a chain, then fill digit-sized gaps.

    Trim first: a run's digits share a height, so a member far from the run's
    median height (a checkmark, a signature flick) is dropped. Then fill: any
    remaining gap wide enough for a digit is re-examined against ALL ink -- not
    just the thick-stroke handwriting mask -- because thin-loop digits like '0'
    are exactly what that mask misses.

    Returns a list of (source, id) where source is 'hw' for handwriting-mask
    components and 'ink' for gap-filled ones.
    """
    H = stats[:, cv2.CC_STAT_HEIGHT]
    L = stats[:, cv2.CC_STAT_LEFT]
    W = stats[:, cv2.CC_STAT_WIDTH]
    if len(chain) < 3:
        return [("hw", i) for i in chain]

    med_h = float(np.median([H[i] for i in chain]))
    med_w = float(np.median([W[i] for i in chain]))
    kept = [i for i in chain if 0.55 * med_h <= H[i] <= 1.7 * med_h]
    if len(kept) < 3:
        kept = list(chain)
    kept.sort(key=lambda i: L[i])

    out = [("hw", i) for i in kept]
    for a, b in zip(kept, kept[1:]):
        gap_l = L[a] + W[a]
        gap_r = L[b]
        if gap_r - gap_l < 0.55 * med_w:
            continue                       # no room for a digit
        ys = [cents[a][1], cents[b][1]]
        y_mid = float(np.mean(ys))
        for j in range(1, n_all):
            cx, cy = ce_all[j]
            if not (gap_l - 2 <= cx <= gap_r + 2):
                continue
            if abs(cy - y_mid) > 0.9 * med_h:
                continue
            hj = st_all[j, cv2.CC_STAT_HEIGHT]
            wj = st_all[j, cv2.CC_STAT_WIDTH]
            if not (0.55 * med_h <= hj <= 1.7 * med_h):
                continue
            if wj > 2.0 * med_w:
                continue
            out.append(("ink", int(j)))
    out.sort(key=lambda t: (L[t[1]] if t[0] == "hw" else st_all[t[1], cv2.CC_STAT_LEFT]))
    return out


def find_number_runs(gray: np.ndarray):
    """Candidate handwritten-number runs, best first.

    Groups the handwriting components into horizontal runs (similar row, small
    gaps) and scores each by how much it looks like a written number: a count
    near nine, consistent digit heights, and digit-like shapes. Signatures lose
    because they are few, wide, connected strokes; the printed banner loses
    because its glyph heights are uniform but its shapes are wide and its run
    is long.
    """
    h_page, w_page = gray.shape[:2]
    _mask, lab, stats, cents, sel = handwriting_mask(gray)
    cand = [i for i in sel if _digit_shaped(stats, i)]
    if len(cand) < MIN_DIGITS:
        return []

    H = stats[:, cv2.CC_STAT_HEIGHT]
    L = stats[:, cv2.CC_STAT_LEFT]
    W = stats[:, cv2.CC_STAT_WIDTH]

    # Chain left-to-right, comparing each new digit to the PREVIOUS one rather
    # than to the first. Hand-written numbers are written at a slant -- on one
    # slip the baseline drifted 86 px (more than a digit height) across nine
    # digits -- so fixed-row grouping shattered the run into sub-threshold
    # pieces and lost it entirely. Chaining follows the slope naturally.
    cand.sort(key=lambda i: cents[i][0])
    used, runs = set(), []
    for start in cand:
        if start in used:
            continue
        chain = [start]
        used.add(start)
        while True:
            last = chain[-1]
            lx = L[last] + W[last]
            ly = cents[last][1]
            lh = max(10.0, float(H[last]))
            best, best_gap = None, None
            for j in cand:
                if j in used:
                    continue
                gap = L[j] - lx
                if gap < -0.5 * W[last] or gap > 2.2 * max(10.0, float(W[last])):
                    continue
                if abs(cents[j][1] - ly) > 0.9 * lh:      # allow a slanted line
                    continue
                if best_gap is None or gap < best_gap:
                    best, best_gap = j, gap
            if best is None:
                break
            chain.append(best)
            used.add(best)
        runs.append(chain)

    # ---- repair each chain before scoring -------------------------------
    # Two defects were costing accuracy, and every model tested (25KB CNN,
    # TrOCR-base, Florence-2) failed the same way because of them:
    #   * a missing digit -- a handwritten '0' is a thin LOOP and fails the
    #     stroke-thickness test that solid digits pass, so 765208418 reached the
    #     reader as '7652 8418'. No model can recover a digit that is absent.
    #   * stray strokes -- a tick/checkmark beside the number gets chained in,
    #     which is what produced the garbage prefixes ('YELH...', 'first # fees').
    all_ink = (gray < 128).astype(np.uint8)
    n_all, lab_all, st_all, ce_all = cv2.connectedComponentsWithStats(all_ink * 255, connectivity=8)
    runs = [_repair_chain(r, stats, cents, n_all, st_all, ce_all) for r in runs]

    scored = []
    for r in runs:
        if not (MIN_DIGITS <= len(r) <= MAX_DIGITS):
            continue

        def box_of(item):
            src, i = item
            s = stats if src == "hw" else st_all
            return (s[i, cv2.CC_STAT_LEFT], s[i, cv2.CC_STAT_TOP],
                    s[i, cv2.CC_STAT_WIDTH], s[i, cv2.CC_STAT_HEIGHT])

        boxes = [box_of(it) for it in r]
        hs = np.array([b[3] for b in boxes], dtype=float)
        if hs.mean() <= 0:
            continue
        consistency = 1.0 - min(1.0, hs.std() / hs.mean())   # 1.0 = uniform
        closeness = 1.0 - abs(len(r) - WANT_DIGITS) / float(WANT_DIGITS)
        score = 2.0 * closeness + consistency
        x0 = int(min(b[0] for b in boxes))
        y0 = int(min(b[1] for b in boxes))
        x1 = int(max(b[0] + b[2] for b in boxes))
        y1 = int(max(b[1] + b[3] for b in boxes))
        scored.append({"score": round(float(score), 3), "box": (x0, y0, x1, y1),
                       "digits": len(r), "ids": r,
                       "consistency": round(float(consistency), 2)})
    scored.sort(key=lambda d: -d["score"])
    return scored


def crop(gray: np.ndarray, box, pad: int = 14) -> np.ndarray:
    """Cut the region from the FULL-QUALITY page so strokes stay intact."""
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = box
    return gray[max(0, y0 - pad):min(h, y1 + pad),
                max(0, x0 - pad):min(w, x1 + pad)]


def crop_handwriting_only(gray: np.ndarray, run, pad: int = 14) -> np.ndarray:
    """Render just this run's strokes on white -- the printed text that overlaps
    the digits is dropped, which is what makes the crop readable.

    Handles both component sources: 'hw' ids index the handwriting-mask
    labelling, 'ink' ids are gap-filled digits taken from the full ink labelling.
    """
    _mask, lab, _stats, _c, _sel = handwriting_mask(gray)
    all_ink = (gray < 128).astype(np.uint8)
    _n, lab_all, _st, _ce = cv2.connectedComponentsWithStats(all_ink * 255, connectivity=8)
    out = np.full_like(gray, 255)
    for item in run["ids"]:
        if isinstance(item, tuple):
            src, i = item
            out[(lab_all if src == "ink" else lab) == i] = 0
        else:
            out[lab == item] = 0
    return crop(out, run["box"], pad)


# OCR trained on printed text maps handwritten digits onto visually similar
# letters. Observed on real slips: 765208417 came back as 'F6S2084 7'
# (F for 7, S for 5). Folding these back recovers most of the number.
LOOKALIKE = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "i": "1", "|": "1", "]": "1", "[": "1",
    "Z": "2", "z": "2",
    "E": "3", "e": "3",
    "A": "4", "h": "4",
    "S": "5", "s": "5",
    "G": "6", "b": "6", "C": "6",
    "T": "7", "F": "7", "f": "7", "t": "7", "?": "7",
    "B": "8", "R": "8",
    "g": "9", "q": "9", "y": "9",
})


def normalise_digits(text: str) -> str:
    """Best-effort digit string from an OCR reading of handwriting."""
    return "".join(c for c in text.translate(LOOKALIKE) if c.isdigit())


def _read_run(ocr, gray: np.ndarray, run) -> str:
    """OCR a run's isolated strokes (upscaled if small)."""
    hw = crop_handwriting_only(gray, run)
    if hw.size == 0:
        return ""
    if min(hw.shape[:2]) < 60:
        hw = cv2.resize(hw, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return ocr.text_of(np.stack([hw] * 3, axis=-1))


def extract(gray: np.ndarray, ocr=None):
    """Best handwritten-number candidate.

    Returns (crop_from_page, handwriting_only_crop, run). When an `ocr` engine
    is supplied the candidates are additionally filtered by reading them: the
    pre-printed bold LIC banner (the main false positive, since bold print is
    also tall and thick) comes back as LETTERS -- 'OF INDIA', '10F FIAPIA' --
    while the real handwritten number comes back as digits only. So a run whose
    reading contains letters is rejected, and runs that read as digits are
    preferred. `run['reading']` carries whatever OCR made of the digits, which
    is a hint for a human, not a trusted value.
    """
    runs = find_number_runs(gray)
    if not runs:
        return None, None, None

    if ocr is not None:
        scored = []
        for r in runs:
            txt = _read_run(ocr, gray, r)
            raw_digits = sum(c.isdigit() for c in txt)
            letters = sum(c.isalpha() for c in txt)
            guess = normalise_digits(txt)
            r["reading"] = txt.strip()
            r["guess"] = guess
            # Digit dominance, not a plain letter count, separates the two:
            # the printed banner read '10F FIAPIA 0' (7 letters vs 3 digits)
            # while the handwritten number read 'et F6S2084 7 G'
            # (6 digits vs 5 letters) -- rejecting on letters>=2 threw the real
            # number away along with the banner.
            if raw_digits < letters:
                continue
            r["score"] = (r["score"]
                          + 1.5 * min(len(guess), WANT_DIGITS) / WANT_DIGITS
                          + (1.0 if len(guess) == WANT_DIGITS else 0.0))
            scored.append(r)
        if scored:
            scored.sort(key=lambda d: -d["score"])
            runs = scored

    best = runs[0]
    return crop(gray, best["box"]), crop_handwriting_only(gray, best), best
