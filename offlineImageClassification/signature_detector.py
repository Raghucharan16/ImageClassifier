"""
Offline signature detection using a YOLOv8 single-class ONNX model
(StabRise/signature_detection), run via onnxruntime -- no torch, no internet.

Returns signature bounding boxes (in original-image pixel coordinates) with
confidence, and can crop them out. Pairs with the document classifier so each
form/declaration can be flagged signed/unsigned and its signatures saved.

Model: input [1,3,960,768] (NCHW), output [1,5,N] = (cx,cy,w,h,score) per anchor
in letterboxed input pixels; single class "signature".
"""
import os
import logging
import numpy as np
import cv2
import onnxruntime as ort

from paths import resource_path

logging.getLogger(__name__)

_MODEL = resource_path(os.path.join("models", "signature", "signature.onnx"))


def _letterbox(img, new_shape):
    """Resize keeping aspect ratio, pad to (H, W). Returns padded img, ratio, (dw, dh)."""
    h0, w0 = img.shape[:2]
    new_h, new_w = new_shape
    r = min(new_h / h0, new_w / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new_w - nw) / 2, (new_h - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, (left, top)


def _nms(boxes, scores, iou_thresh):
    """Pure-numpy NMS. boxes: xyxy. Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


class SignatureDetector:
    def __init__(self, model_path=_MODEL, conf_thresh=0.5, iou_thresh=0.45):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Signature model not found at {model_path}. See README_offline.md."
            )
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # NCHW -> (H, W)
        _, _, self.in_h, self.in_w = inp.shape
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        logging.info(f"SignatureDetector ready ({self.in_w}x{self.in_h}).")

    def detect(self, img_path):
        """Return list of dicts: {box:[x1,y1,x2,y2], conf:float} in original pixels."""
        img = cv2.imread(img_path)
        if img is None:
            from PIL import Image
            try:
                img = cv2.cvtColor(np.array(Image.open(img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception:
                return []
        h0, w0 = img.shape[:2]

        padded, r, (dw, dh) = _letterbox(img, (self.in_h, self.in_w))
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]  # NCHW

        out = self.session.run(None, {self.input_name: blob})[0]  # [1,5,N]
        preds = out[0].T  # [N,5]
        scores = preds[:, 4]
        mask = scores >= self.conf_thresh
        preds, scores = preds[mask], scores[mask]
        if len(preds) == 0:
            return []

        # xywh (center) in input px -> xyxy
        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        keep = _nms(xyxy, scores, self.iou_thresh)

        results = []
        for i in keep:
            x1, y1, x2, y2 = xyxy[i]
            # undo letterbox -> original coords
            x1 = (x1 - dw) / r; x2 = (x2 - dw) / r
            y1 = (y1 - dh) / r; y2 = (y2 - dh) / r
            x1 = max(0, min(w0, x1)); x2 = max(0, min(w0, x2))
            y1 = max(0, min(h0, y1)); y2 = max(0, min(h0, y2))
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            results.append({"box": [int(x1), int(y1), int(x2), int(y2)],
                            "conf": round(float(scores[i]), 3)})
        return results

    def crop(self, img_path, box, pad=4):
        img = cv2.imread(img_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        return img[y1:y2, x1:x2]


if __name__ == "__main__":
    import sys
    det = SignatureDetector()
    for p in sys.argv[1:]:
        dets = det.detect(p)
        print(f"{p}: {len(dets)} signature(s) -> {dets}")
