"""
Offline Image Classification - interactive pipeline.

Flow when run (double-click .exe / run.bat, or `python run_offline.py`):
    1. Ask for the INPUT folder (original scans) and OUTPUT folder.
    2. Enhance? [y/n]  -> blank-skip / auto-orient / clean / compress into
                          <output>/_enhanced
    3. Classify? [y/n] -> OCR + rule-based classify the images, then segregate
                          them into <output>/<category>/ subfolders.

Enhance+Classify together share a single OCR pass per image (reliable 4-way
rotation fix + text, no duplicated work). Enhance alone uses a fast classical
(no-OCR) rotation guess instead, to stay near real-time on modest hardware.
If enhancement runs, classification operates on the ENHANCED images.
Everything runs fully offline (RapidOCR ONNX models are bundled).
"""
import os
import sys
import argparse
import logging

from enhance import preprocess_images_fast, preprocess_and_ocr
from classify_offline import classify_images_offline
from segregate import segregate_images
from detect_signatures import detect_signatures

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def ask(prompt, default=None):
    val = input(prompt).strip()
    return val or default


def yesno(prompt, default_yes=True):
    d = "Y/n" if default_yes else "y/N"
    val = input(f"{prompt} [{d}]: ").strip().lower()
    if not val:
        return default_yes
    return val in ("y", "yes")


def main():
    ap = argparse.ArgumentParser(description="Offline Image Classification")
    ap.add_argument("-i", "--input-dir")
    ap.add_argument("-o", "--output-dir")
    ap.add_argument("--enhance", action="store_true")
    ap.add_argument("--classify", action="store_true")
    ap.add_argument("--signatures", action="store_true")
    ap.add_argument("--no-enhance", action="store_true")
    ap.add_argument("--no-classify", action="store_true")
    ap.add_argument("--no-signatures", action="store_true")
    ap.add_argument("--sig-conf", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    interactive = not args.input_dir

    print("=" * 52)
    print("        Offline Image Classification")
    print("=" * 52)

    # 1. Folders
    input_dir = args.input_dir or ask("Enter INPUT folder (scans) [Images]: ", "Images")
    if not os.path.isdir(input_dir):
        print(f"Error: input folder '{input_dir}' does not exist.")
        if interactive:
            input("Press Enter to exit...")
        sys.exit(1)
    output_dir = args.output_dir or ask("Enter OUTPUT folder [output]: ", "output")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Enhance?
    if args.enhance:
        do_enhance = True
    elif args.no_enhance:
        do_enhance = False
    else:
        do_enhance = yesno("Enhance images (blank-skip / auto-orient / clean / compress)?", True)

    # 3. Classify?
    if args.classify:
        do_classify = True
    elif args.no_classify:
        do_classify = False
    else:
        do_classify = yesno("Classify the images now?", True)

    classify_source = input_dir
    if do_enhance and do_classify:
        # Merged path: the OCR pass Classify needs anyway also gives a
        # reliable 4-way orientation fix, so rotation is only solved once.
        enhanced_dir = os.path.join(output_dir, "_enhanced")
        print(f"\n--- Preprocessing + OCR -> {enhanced_dir} ---")
        preprocess_and_ocr(input_dir, enhanced_dir)
        classify_source = enhanced_dir
    elif do_enhance:
        # Enhance-only: no OCR, so rotation uses a fast best-effort classical
        # guess (reliable for sideways 90/270, can miss upside-down 180 on
        # unusual layouts) instead of the slow/reliable OCR-based method.
        enhanced_dir = os.path.join(output_dir, "_enhanced")
        print(f"\n--- Enhancing (no OCR) -> {enhanced_dir} ---")
        preprocess_images_fast(input_dir, enhanced_dir)
        classify_source = enhanced_dir
    else:
        print("\nSkipping enhancement; classification will use the original images.")

    results_json = os.path.join(output_dir, "classification_results.json")

    if do_classify:
        print(f"\n--- Classifying images in {classify_source} ---")
        # Uses the cached OCR text from preprocessing when available (no re-OCR).
        classify_images_offline(classify_source, results_json, limit=args.limit)

    # 4. Signatures? (separate 'signatures' class alongside the categories)
    if args.signatures:
        do_sig = True
    elif args.no_signatures:
        do_sig = False
    else:
        do_sig = yesno("Detect signatures & collect signed pages into a 'signatures' folder?", True)

    if do_sig:
        print(f"\n--- Detecting signatures in {classify_source} ---")
        detect_signatures(classify_source, results_json, output_dir,
                          conf=args.sig_conf, limit=args.limit)

    if do_classify:
        print("\n--- Segregating into category subfolders ---")
        segregate_images(classify_source, results_json, output_dir)
        print(f"\nDone. Sorted images: {output_dir}  |  metadata: {results_json}")
    elif do_sig:
        print(f"\nDone. Signed pages collected in {os.path.join(output_dir, 'signatures')}.")
    else:
        print(f"\nDone. Enhanced images are in {classify_source} (classification skipped).")

    print("=" * 52)
    if interactive:
        input("Press Enter to exit...")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # required for parallel workers in the frozen .exe
    main()
