"""
Detect signatures on each document, flag signed/unsigned, and save CROPS.

Decoupled from classification so it can run on its own. It augments the
classification results JSON in place with, per file:
    "signed": true|false,
    "signature_count": <n>,
    "signatures": [{"box": [x1,y1,x2,y2], "conf": 0.7, "crop": "<path>"}]

Each detected signature is cropped out of the page (not the whole page) and
saved as its own lossless PNG into <output_dir>/signatures/. A page with N
signatures produces N crop files: "<stem>_sig0.png", "<stem>_sig1.png", ...

Note: this is a detector, not a verifier. On forms with very dense cursive
handwriting it may occasionally box a handwritten field value; raise --conf
to favour precision. Real signatures typically score >=0.65.
"""
import os
import json
import argparse
import logging

import cv2

from signature_detector import SignatureDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

IMAGE_EXTS = (".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp")


def detect_signatures(input_dir, results_json, output_dir, conf=0.5, limit=None):
    det = SignatureDetector(conf_thresh=conf)

    results = {}
    if os.path.exists(results_json):
        with open(results_json, "r", encoding="utf-8") as f:
            results = json.load(f)

    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(IMAGE_EXTS))
    if limit:
        files = files[:limit]

    sig_dir = os.path.join(output_dir, "signatures")
    os.makedirs(sig_dir, exist_ok=True)

    signed_total = 0
    for i, filename in enumerate(files, 1):
        path = os.path.join(input_dir, filename)
        dets = det.detect(path)

        stem = os.path.splitext(filename)[0]
        sigs = []
        for j, d in enumerate(dets):
            crop_img = det.crop(path, d["box"])
            crop_name = f"{stem}_sig{j}.png"
            crop_path = os.path.join(sig_dir, crop_name)
            if crop_img is not None and crop_img.size:
                cv2.imwrite(crop_path, crop_img)
                sigs.append({"box": d["box"], "conf": d["conf"], "crop": crop_path})

        entry = results.get(filename, {})
        entry["signed"] = bool(sigs)
        entry["signature_count"] = len(sigs)
        entry["signatures"] = sigs
        results[filename] = entry
        if sigs:
            signed_total += 1

        if i % 20 == 0 or i == len(files):
            logging.info(f"[{i}/{len(files)}] {filename}: {len(sigs)} signature(s)")

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    logging.info(f"Done. {signed_total}/{len(files)} pages have signatures, "
                 f"crops saved into {sig_dir}. Metadata merged into {results_json}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Offline signature detection")
    ap.add_argument("--input-dir", default="Images")
    ap.add_argument("--results-json", default="offline_classification_results.json")
    ap.add_argument("--output-dir", default="offlineSegregate")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    detect_signatures(args.input_dir, args.results_json, args.output_dir, args.conf, args.limit)
