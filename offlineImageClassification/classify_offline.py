"""
Offline document classification.

OCRs each ORIGINAL scan with RapidOCR (ONNX) and applies the rule-based
classifier. Originals are used deliberately: the Gemini-oriented preprocessing
(median blur + deskew + recompress) destroys ~80% of recoverable text, while
RapidOCR reads the originals at ~0.9 confidence.

Results are written as:
    { "<filename>": {"content": "<short label>", "recommended_folder": "<cat>"} }
matching the shape produced by the Gemini pipeline (classify_batch.py).
"""
import os
import json
import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from ocr_engine import RapidOCREngine
from rules import RuleBasedClassifier
from enhance import _default_workers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

IMAGE_EXTS = (".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp")


OCR_CACHE_NAME = "_ocr_cache.json"
_tls = threading.local()


def _thread_ocr(enable_tamil, intra_threads):
    eng = getattr(_tls, "ocr", None)
    if eng is None:
        eng = _tls.ocr = RapidOCREngine(enable_tamil=enable_tamil, intra_threads=intra_threads)
    return eng


def classify_images_offline(input_dir, output_json, limit=None, enable_tamil=False,
                            ocr=None, workers=None):
    classifier = RuleBasedClassifier()

    # Reuse OCR text cached by the preprocessing stage, so we never OCR twice.
    text_cache = {}
    cache_path = os.path.join(input_dir, OCR_CACHE_NAME)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                text_cache = json.load(f)
            logging.info(f"Using cached OCR text for {len(text_cache)} images (no re-OCR).")
        except Exception:
            logging.warning("Could not read OCR cache; will OCR as needed.")

    results = {}
    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                results = json.load(f)
            logging.info(f"Loaded {len(results)} existing entries from {output_json}")
        except Exception:
            logging.warning("Could not read existing results, starting fresh.")

    all_files = sorted(f for f in os.listdir(input_dir)
                       if f.lower().endswith(IMAGE_EXTS))
    if not all_files:
        logging.error(f"No images found in {input_dir}.")
        return

    if limit:
        all_files = all_files[:limit]

    todo = [f for f in all_files if f not in results]
    logging.info(f"Total: {len(all_files)} | already done: {len(results)} | to do: {len(todo)}")
    if not todo:
        return

    # Files with cached text classify instantly (no OCR); the rest need OCR
    # and are run in parallel threads (same sizing as the enhance stage).
    cached_todo = [f for f in todo if f in text_cache]
    ocr_todo = [f for f in todo if f not in text_cache]

    cpu = os.cpu_count() or 4
    workers = workers or _default_workers(cpu)
    workers = min(workers, max(1, len(ocr_todo)))
    intra = max(1, cpu // workers) if ocr_todo else 1

    def get_text(filename):
        if ocr is not None:
            return ocr.extract_text(os.path.join(input_dir, filename))
        return _thread_ocr(enable_tamil, intra).extract_text(os.path.join(input_dir, filename))

    done = 0
    total = len(todo)

    for filename in cached_todo:
        category, label, _ = classifier.classify(text_cache[filename])
        results[filename] = {"content": label, "recommended_folder": category}
        done += 1
        logging.info(f"[{done}/{total}] {filename} -> {category}  ({label})  [cached]")

    if ocr_todo:
        logging.info(f"OCR'ing {len(ocr_todo)} image(s) with {workers} thread(s) "
                     f"(intra_threads={intra})")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for filename, text in zip(ocr_todo, ex.map(get_text, ocr_todo)):
                category, label, _ = classifier.classify(text)
                results[filename] = {"content": label, "recommended_folder": category}
                done += 1
                if done % 10 == 0 or done == total:
                    with open(output_json, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)
                logging.info(f"[{done}/{total}] {filename} -> {category}  ({label})")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logging.info(f"Offline classification complete. {len(results)} entries in {output_json}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline OCR Image Classification")
    parser.add_argument("--input-dir", default="Images", help="Folder of original document images")
    parser.add_argument("--output-json", default="offline_classification_results.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tamil", action="store_true", help="Enable an extra Tamil OCR pass")
    args = parser.parse_args()
    classify_images_offline(args.input_dir, args.output_json, args.limit, args.tamil)
