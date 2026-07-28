"""
Read the handwritten 9-digit application number from a review slip.

Pipeline (crop-then-recognise, which is what makes this work at all):
  1. appnumber.find_number_runs isolates HANDWRITING only -- printed dot-matrix
     text is ~19px tall while the pen digits are 43-56px, so height + stroke
     thickness separate them. Reading the number with the printed text still in
     frame never worked: OCR returned the surrounding address instead.
  2. Each candidate run is repaired (strays trimmed, digit-sized gaps re-filled
     from the full ink mask, because a handwritten '0' is a thin loop that the
     thickness test drops -- that alone turned 765208418 into '7652 8418').
  3. The isolated crop goes to TrOCR-base-handwritten, a model trained on real
     handwriting. Measured alternatives on the same crops: an MNIST CNN got the
     middle digits at confidence 1.0 but the ends wrong, and TrOCR-small
     returned prose.
  4. The reading is normalised to digits. The number is written WITH SPACES
     ('765208 417'), so spaces are simply dropped, and letters that printed-text
     tokenisers substitute for digits are folded back (F->7, S->5, B->8 ...).
  5. Every candidate view/run is read and the results VOTED. A number is only
     accepted when it is exactly 9 digits; otherwise the caller is told to fall
     back and the crop is kept for a human.

Nothing here assumes the numbers are sequential.
"""
from __future__ import annotations

import logging
import os
import re
from collections import Counter

import cv2
import numpy as np

import appnumber as AN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

WANT = 9

# Chosen by measurement on identical isolated crops from this batch
# (truths 765208417 / 765208418 / 765208419, read off the crops by eye):
#   MNIST CNN 25MB      '14520844'                     - ends wrong
#   TrOCR-small 60MB    'in size20th F'                - prose
#   TrOCR-base 1.3GB    abstains, and invented         - produced a WRONG
#                                                        785208418 at conf 0.67
#   GOT-OCR-2.0 1.1GB   '165208418' for 765208418      - WRONG but 9 digits,
#                                                        i.e. it would pass the
#                                                        length gate silently
#   TrOCR-large 2.4GB   'FGS 208417' -> 765208417      - EXACT, and abstained
#                                                        on the ones it could
#                                                        not resolve
# The deciding property is not raw accuracy but that trocr-large declines
# instead of guessing: a wrong folder name is worse than a blank one.
MODEL = "microsoft/trocr-large-handwritten"


class HandwrittenNumberReader:
    """Loads TrOCR once; call read_slip() per review slip, then free it."""

    def __init__(self, model_name: str = MODEL, threads: int | None = None):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).eval()
        logging.info(f"handwriting reader ready ({model_name})")

    # ------------------------------------------------------------- reading
    def _read(self, gray: np.ndarray, beams: int = 4) -> str:
        from PIL import Image
        if gray.size == 0:
            return ""
        # upscale small crops: the recogniser expects text-line sized input
        if min(gray.shape[:2]) < 96:
            f = 96.0 / min(gray.shape[:2])
            gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        img = Image.fromarray(gray).convert("RGB")
        px = self.processor(images=img, return_tensors="pt").pixel_values
        with self._torch.no_grad():
            ids = self.model.generate(px, max_new_tokens=24, num_beams=beams)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _to_digits(text: str) -> str:
        """Digits only. Spaces are expected (the number is written '765208 417')
        so they are dropped; digit-shaped letters are folded back.

        Guarded by MIN_RAW_DIGITS in the caller: folding letters to digits is
        only legitimate for a reading that is ALREADY mostly digits. Applied
        blindly it manufactures numbers out of prose -- the footer text
        'ION GENERALA.S' became '106338445', a perfectly plausible-looking
        9-digit application number that was completely invented.
        """
        return AN.normalise_digits(text.replace(" ", ""))

    # A reading must be mostly digits before lookalike folding is trusted.
    MIN_RAW_DIGITS = 5

    # ------------------------------------------------------------- per slip
    def read_slip(self, gray: np.ndarray, want: int = WANT) -> dict:
        """Return {'number', 'confidence', 'crop', 'candidates'}.

        'number' is None unless a `want`-digit result was obtained.
        """
        runs = AN.find_number_runs(gray)
        if not runs:
            return {"number": None, "confidence": 0.0, "crop": None,
                    "candidates": [], "reason": "no handwriting run found"}

        h = gray.shape[0]
        # the number is written in the body of the slip, not in the footer
        # signature area; prefer upper runs but keep all as fallback
        ordered = sorted(runs, key=lambda r: (r["box"][1] > 0.45 * h, -r["digits"]))

        votes: Counter = Counter()
        cands = []
        best_crop = None
        for r in ordered[:4]:
            iso = AN.crop_handwriting_only(gray, r, pad=20)
            if best_crop is None:
                best_crop = iso
            # Only the ISOLATED view is read. The page view still contains the
            # printed text the digits are written over, and it returned things
            # like '" See also known as the House' -- pure noise that then
            # normalised into long fake digit strings.
            txt = self._read(iso)
            raw_digits = sum(c.isdigit() for c in txt)
            digits = self._to_digits(txt)
            accepted = (raw_digits >= self.MIN_RAW_DIGITS and len(digits) == want)
            cands.append({"view": "isolated", "y": round(r["box"][1] / h, 3),
                          "raw": txt, "digits": digits,
                          "raw_digit_count": raw_digits, "accepted": accepted})
            if accepted:
                votes[digits] += 1

        if votes:
            number, score = votes.most_common(1)[0]
            # confidence reflects how much of the ORIGINAL reading was already
            # digits, not just vote agreement -- a single mostly-letters read
            # scoring 1.0 was how the invented number slipped through
            best = next(c for c in cands if c["digits"] == number)
            conf = round(min(1.0, best["raw_digit_count"] / float(want)), 2)
            return {"number": number, "confidence": conf,
                    "crop": best_crop, "candidates": cands}

        # nothing produced exactly `want` digits -- report the closest attempt
        closest = max(cands, key=lambda c: len(c["digits"]), default=None)
        return {"number": None, "confidence": 0.0, "crop": best_crop,
                "candidates": cands,
                "reason": f"no {want}-digit reading (best: "
                          f"{closest['digits'] if closest else ''!r})"}


def save_crop(crop: np.ndarray, path: str, scale: int = 3) -> None:
    """Write the number crop, upscaled so a human can read it at a glance."""
    if crop is None or crop.size == 0:
        return
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(path, big)
