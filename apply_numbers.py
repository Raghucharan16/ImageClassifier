"""
Rename APP_nn folders to their real 9-digit application numbers.

    # 1. index_batch.py writes application_numbers.csv with a suggestion +
    #    the cropped handwriting for every application
    # 2. open the CSV, check each 'application_number' cell against the crop
    #    (the crop is a clean, upscaled picture of just the digits)
    # 3. python apply_numbers.py --output <indexed folder>

Why a confirmation step: the number is handwritten, and every recogniser tried
(MNIST CNN, TrOCR-small, TrOCR-base, Florence-2) misread at least some digits.
One produced a wrong-but-plausible number at high confidence, which would have
named a folder incorrectly with nothing to flag it. Reading 9 digits off the
crop takes a couple of seconds and cannot fail silently.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CSV_NAME = "application_numbers.csv"


def run(out_dir: str, digits: int = 9, dry_run: bool = False):
    path = os.path.join(out_dir, CSV_NAME)
    if not os.path.exists(path):
        logging.error(f"{CSV_NAME} not found in {out_dir}; run index_batch.py first")
        return

    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    pat = re.compile(rf"^\d{{{digits}}}$")
    renamed = skipped = 0
    seen: dict[str, str] = {}

    for r in rows:
        folder = (r.get("folder") or "").strip()
        num = re.sub(r"\D", "", (r.get("application_number") or ""))
        src = os.path.join(out_dir, folder)
        if not folder or not os.path.isdir(src):
            continue
        if not pat.match(num):
            logging.warning(f"{folder}: no valid {digits}-digit number "
                            f"({r.get('application_number')!r}) -- left as is")
            skipped += 1
            continue
        if num in seen:
            logging.error(f"{folder}: number {num} already used by {seen[num]} -- skipped")
            skipped += 1
            continue
        dst = os.path.join(out_dir, num)
        if os.path.exists(dst):
            logging.error(f"{folder}: target {num} already exists -- skipped")
            skipped += 1
            continue
        logging.info(f"{folder} -> {num}")
        if not dry_run:
            os.rename(src, dst)
        seen[num] = folder
        renamed += 1

    logging.info(f"{'would rename' if dry_run else 'renamed'}={renamed}  skipped={skipped}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rename APP_nn folders to application numbers")
    ap.add_argument("--output", required=True, help="the indexed output folder")
    ap.add_argument("--digits", type=int, default=9)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run(a.output, a.digits, a.dry_run)
