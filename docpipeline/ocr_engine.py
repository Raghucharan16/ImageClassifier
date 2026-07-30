"""
Offline OCR engine built on RapidOCR (ONNXRuntime).

Why RapidOCR: it runs the same PP-OCR models as PaddleOCR but through
ONNXRuntime, so there is no PaddlePaddle native layer (and none of the
oneDNN crashes we hit on this Windows CPU). The models are small, bundled
ONNX files, which makes the whole thing trivial to ship to an offline
deployment machine.

The engine OCRs *original* scans (not the degraded, Gemini-oriented PNGs)
and includes a rotation fallback so sideways full-page scans still read.
"""
import os
# Keep paddle's oneDNN path disabled in case any paddle import sneaks in.
os.environ.setdefault("FLAGS_use_mkldnn", "0")

import logging
import numpy as np
import cv2
from PIL import Image

from rapidocr_onnxruntime import RapidOCR

from paths import resource_path

logging.getLogger("RapidOCR").setLevel(logging.WARNING)


_cuda_available: bool | None = None


def cuda_available() -> bool:
    """True when onnxruntime on THIS machine can actually run on CUDA.

    Asks onnxruntime what providers it was built with rather than looking for a
    GPU directly: a CUDA-capable card is useless if the installed wheel is the
    CPU-only or DirectML one, and that mismatch is the normal state of affairs
    (the three onnxruntime variants cannot be installed side by side). Probing
    the runtime therefore answers the only question that matters -- will
    use_cuda=True actually work here -- and lets one build serve both GPU and
    CPU machines. Run setup_runtime.py once per machine to install the matching
    wheel; see RAPIDOCR_CUDA to override.
    """
    global _cuda_available
    if _cuda_available is None:
        override = os.environ.get("RAPIDOCR_CUDA")
        if override is not None:
            _cuda_available = override == "1"
        else:
            try:
                import onnxruntime as ort
                _cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
            except Exception:
                _cuda_available = False
        logging.info("OCR execution provider: %s",
                     "CUDA (GPU)" if _cuda_available else "CPU")
    return _cuda_available


def _load_bgr(img_path: str):
    """Load any image (jpg/tif/png) as a BGR ndarray, robust to odd TIFFs."""
    img = cv2.imread(img_path)
    if img is not None:
        return img
    # Fallback via PIL for TIFFs/formats cv2 chokes on
    try:
        pil = Image.open(img_path)
        pil = pil.convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"Could not load image {img_path}: {e}")
        return None


def _rotate(img, angle: int):
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


_TAMIL_REC = resource_path(os.path.join("models", "tamil", "ta_rec.onnx"))
_TAMIL_DICT = resource_path(os.path.join("models", "tamil", "ta_dict.txt"))


class RapidOCREngine:
    """RapidOCR-based extractor with a rotation fallback.

    Returns the concatenated recognized text. A combined score
    (text length x mean confidence) is used to pick the best orientation.

    enable_tamil: run an additional Tamil recognition pass and append its text
    so the rules see both Latin and Tamil keywords. Requires a converted Tamil
    ONNX recogniser at models/tamil/ (see tools/convert_tamil_model.py). If the
    model is absent the engine logs a warning and continues English-only --
    classification still works because Indian KYC/insurance docs carry English
    markers, so Tamil is an accuracy add-on, not a hard dependency.
    """

    def __init__(self, sparse_char_threshold: int = 40, low_conf: float = 0.55,
                 enable_tamil: bool = False, intra_threads: int | None = None,
                 use_cuda: bool | None = None):
        # Bundled ch+en PP-OCRv4 models. The detector is language-agnostic;
        # the recognizer handles Latin/English well. intra_threads caps the
        # onnxruntime thread pool -- set it low when running many parallel
        # worker processes so they don't oversubscribe the CPU.
        #
        # use_cuda defaults to None = auto-detect: GPU when the installed
        # onnxruntime actually offers the CUDA provider, CPU otherwise. One
        # build therefore runs correctly on both a 4090 workstation and a
        # CPU-only client machine with no flags to remember. rec_batch_num is
        # raised to 32 on GPU so more crops go per ONNX call, amortising CUDA
        # kernel-launch overhead.
        if use_cuda is None:
            use_cuda = cuda_available()
        cuda_kwargs: dict = (
            {"det_use_cuda": True, "rec_use_cuda": True, "cls_use_cuda": True,
             "rec_batch_num": 32}
            if use_cuda else {}
        )
        base_kwargs: dict = (
            {"intra_op_num_threads": intra_threads} if intra_threads else {}
        )
        self.engine = RapidOCR(**base_kwargs, **cuda_kwargs)
        self.sparse_char_threshold = sparse_char_threshold
        self.low_conf = low_conf
        logging.info("RapidOCR (ONNX) engine initialized%s.",
                     " [CUDA]" if use_cuda else "")

        self.tamil = None
        if enable_tamil:
            if os.path.exists(_TAMIL_REC) and os.path.exists(_TAMIL_DICT):
                try:
                    self.tamil = RapidOCR(
                        Rec_model_path=_TAMIL_REC, Rec_keys_path=_TAMIL_DICT,
                    )
                    logging.info("Tamil recognition pass enabled.")
                except Exception as e:
                    logging.warning(f"Could not load Tamil model, continuing English-only: {e}")
            else:
                logging.warning(
                    f"Tamil model not found at {_TAMIL_REC}; continuing English-only. "
                    "Run tools/convert_tamil_model.py to generate it."
                )

    def _ocr_array(self, img, max_side=2200):
        """Run OCR on a BGR ndarray. Returns (text, mean_conf, n_boxes).

        Caps the longest side at max_side before running RapidOCR: bounds the
        worst-case cost for unusually high-DPI scans without affecting normal
        scans (most inputs are already well under this) or text legibility.
        """
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            f = max_side / max(h, w)
            img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        try:
            result, _ = self.engine(img)
        except Exception as e:
            logging.error(f"RapidOCR failed on array: {e}")
            return "", 0.0, 0
        if not result:
            return "", 0.0, 0
        texts = [box[1] for box in result]
        confs = [box[2] for box in result]
        mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
        return " ".join(texts), mean_conf, len(result)

    @staticmethod
    def _score(text: str, conf: float) -> float:
        return len(text) * conf

    def text_of(self, img):
        """Single OCR pass over an ndarray (already upright). Returns text."""
        return self._ocr_array(img)[0]

    def orient_and_read(self, img, min_chars=60, detect_max=500):
        """Orient the image AND return its OCR text, in as few FULL-RESOLUTION
        passes as possible -- this is the main speed lever for batch runs.

        Always OCRs the image upright at full resolution once; that pass is
        needed anyway to get the classification text, so it's never wasted
        work. Real (non-blank) pages fed in their normal orientation read at
        least a few dozen characters, so this single pass is the answer for
        the overwhelming common case: office scans are almost always fed
        upright, so most images cost exactly ONE full-res OCR call.

        Only when that first pass is sparse (< min_chars, i.e. the page is
        likely sideways/upside-down) do we search the other three rotations --
        and we do that on a small, cheap thumbnail rather than full-res, then
        run ONE more full-res pass on the winning rotation. Worst case is
        therefore 1 full pass + 3 cheap thumbnail passes + 1 more full pass,
        instead of up to 4 full-resolution passes.

        Returns (upright_image, text).
        """
        text0 = self.text_of(img)
        if len(text0) >= min_chars:
            return img, text0

        h, w = img.shape[:2]
        scale = detect_max / max(h, w) if max(h, w) > detect_max else 1.0
        small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img

        best_angle, best = 0, len(text0)
        for angle in (90, 180, 270):
            t = self.text_of(_rotate(small, angle))
            if len(t) > best:
                best, best_angle = len(t), angle

        if best_angle == 0:
            return img, text0
        rotated = _rotate(img, best_angle)
        return rotated, self.text_of(rotated)

    def orient_upright(self, img, detect_max=800, confident=250):
        """Rotate a scan to the correct upright orientation, losslessly.

        Handles ALL four orientations (0/90/180/270), including upside-down.
        Orientation is detected on a downscaled thumbnail; the winning rotation
        is applied to the full-resolution image with cv2.rotate (transpose/flip,
        no interpolation), so OCR quality is never degraded.

        Fast path: if the upright thumbnail already reads confidently
        (len(text) x conf >= `confident`), we accept it without testing the
        other three rotations -- this is the common case for dense pages.
        """
        h, w = img.shape[:2]
        scale = detect_max / max(h, w) if max(h, w) > detect_max else 1.0
        small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img

        best = self._score(*self._ocr_array(small)[:2])
        if best >= confident:
            return img  # already upright and reading well
        best_angle = 0
        for angle in (90, 180, 270):
            s = self._score(*self._ocr_array(_rotate(small, angle))[:2])
            if s > best:
                best, best_angle = s, angle
        return _rotate(img, best_angle) if best_angle else img

    def extract_text(self, img_path: str) -> str:
        img = _load_bgr(img_path)
        if img is None:
            return ""

        best_text, best_conf, _ = self._ocr_array(img)
        best_score = self._score(best_text, best_conf)

        upright = img
        # If the upright pass looks sparse or low-confidence, the page may be
        # rotated. Try the three other orientations and keep the best.
        if len(best_text.strip()) < self.sparse_char_threshold or best_conf < self.low_conf:
            for angle in (90, 270, 180):
                rot = _rotate(img, angle)
                t, c, _ = self._ocr_array(rot)
                if self._score(t, c) > best_score:
                    best_text, best_conf, best_score, upright = t, c, self._score(t, c), rot
                    logging.info(f"Rotation {angle} improved OCR for {os.path.basename(img_path)}.")

        # Optional Tamil pass on the best orientation; append so the rules see
        # both scripts' keywords.
        if self.tamil is not None:
            try:
                res, _ = self.tamil(upright)
                if res:
                    ta_text = " ".join(b[1] for b in res)
                    if ta_text.strip():
                        best_text = (best_text + " " + ta_text).strip()
            except Exception as e:
                logging.debug(f"Tamil pass failed for {os.path.basename(img_path)}: {e}")

        return best_text


if __name__ == "__main__":
    import sys
    eng = RapidOCREngine()
    for p in sys.argv[1:]:
        txt = eng.extract_text(p)
        print(f"\n=== {p} ({len(txt)} chars) ===")
        print(txt[:500])
