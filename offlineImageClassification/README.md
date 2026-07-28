# Offline Image Classification

Self-contained, fully offline LIC-proposal-packet document sorter. No internet,
no Python installation needed on the target machine (standalone `.exe`).

## Interactive flow

1. Ask for **input** folder (scans) and **output** folder.
2. **Enhance? [y/n]**
3. **Classify? [y/n]**
4. **Signatures? [y/n]** — detect signatures and save each one **cropped**
   (not the full page) into `<output>/signatures/`, a separate folder
   alongside the categories. A page with N signatures produces N crop files
   (`<stem>_sig0.png`, `<stem>_sig1.png`, ...).

Categories (from `rules.py`): `Proposal_form`, `Proposal_enclosures`,
`Proposal_review_slip`, `Medical_report`, `KYC_documents`, `Bank`,
`unidentified` (fallback for anything that matches none of the above).

## Enhance and Classify are linked, by design

Reliable rotation correction (telling upright from upside-down) requires
reading the text — i.e. OCR — which is inherently a multi-second-per-page
operation on CPU. There is no way around that; see "Speed, honestly" below.
So the two stages behave differently depending on what you pick:

- **Enhance + Classify together** (recommended): OCR runs **once** per image.
  That single pass both reliably fixes rotation (all 4 orientations) and
  produces the classification text — no duplicated work. The classified
  category then picks the size/colour budget: `KYC_documents`/`Bank` pages
  keep colour, capped at 200 KB (worth preserving an ID photo/logo); every
  other category becomes cleaned grayscale, capped at 50 KB.
- **Enhance only** (no Classify): there's no OCR, so rotation uses a fast
  **classical heuristic** (no OCR/ML, ~40ms) instead — reliable for sideways
  90°/270° pages, but it can occasionally guess wrong on upside-down 180°
  pages with an unusual layout (e.g. several ID cards stacked on one scan).
  Since no category is known yet, output defaults to grayscale, 50 KB.
- **Classify only** (no Enhance): OCR runs on the original scans directly,
  in parallel; no cleaned/resized images are produced, just the
  classification result.

Neither path does anything destructive (arbitrary-angle deskew, harsh
binarisation, auto-crop, forced sharpening) — those were measured to lower
classification accuracy. Contrast boost and denoising are **adaptive**: a
page is only touched if a quick measurement says it actually needs it.

## Speed, honestly

- **Enhance-only** (no OCR): pure image processing — blank-check, classical
  rotation guess, adaptive contrast/denoise, compress. Well under 1s/image on
  reasonable hardware.
- **Enhance+Classify / Classify**: OCR-bound. A dense multi-field form can
  take several seconds per page on CPU even on capable hardware — this is a
  property of running OCR without a GPU, not a bug. Worker threads default to
  4-8 (not scaled to the build machine's core count) to stay reasonable on
  modest deployment hardware (e.g. an 11th-gen i5-G7 @ 1.5GHz).

## Run it

- **Standalone** (no Python needed): `dist/run_offline.exe`, or double-click
  `run.bat`.
- **Command line:**
  ```
  run_offline.exe                                            # interactive prompts
  run_offline.exe -i scans -o output --enhance --classify --no-signatures
  run_offline.exe -i scans -o output --no-enhance --classify --no-signatures
  ```
- **From source:** `python run_offline.py`

## Files
| File | Purpose |
|------|---------|
| `run_offline.py` | Interactive pipeline (enhance → classify → signatures → segregate) |
| `enhance.py` | Preprocessing: blank-skip, rotation (classical or OCR-merged), adaptive clean, compress |
| `ocr_engine.py` | RapidOCR (ONNX) wrapper: single-pass-first orientation+read, resolution safety cap |
| `rules.py` | Rule-based classifier for the 7-category taxonomy |
| `classify_offline.py` | OCR (parallel) + classify → results JSON |
| `signature_detector.py` | YOLOv8 ONNX signature detector |
| `detect_signatures.py` | Flag signed pages + crop each signature into `signatures/` |
| `segregate.py` | Copy images into category subfolders (dynamic, from `rules.CATEGORIES`) |
| `paths.py` | Resolve bundled model paths (source or frozen `.exe`) |
| `run.bat` | Double-click launcher |
| `run_offline.spec` | PyInstaller build spec |
| `requirements.txt` | Python dependencies for running from source / rebuilding |
| `models/signature/signature.onnx` | Bundled signature-detection model |
| `dist/run_offline.exe` | The built standalone executable |

## Rebuild the .exe
```
pip install -r requirements.txt pyinstaller
pyinstaller run_offline.spec --noconfirm --distpath dist
```
Output: `dist/run_offline.exe` (single self-contained file, no internet needed).
