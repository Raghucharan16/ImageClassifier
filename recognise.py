"""
Read a handwritten number crop with TrOCR (quantised ONNX, CPU only).

Used for the application number, where two cheaper approaches were measured and
rejected:
  * printed-text OCR on the crop returns the surrounding printed words, or
    lookalike letters for the digits ('F6S2084 7' for 765208417)
  * an MNIST CNN on the individually segmented digits got the middle of the
    number right at confidence 1.0 but the ends wrong at 0.64-0.71
    ('14520844' vs 765208417) -- MNIST is trained on clean, thick, centred
    digits and ballpoint form-writing is out of domain, so per-digit accuracy
    around 65% leaves almost no chance of a correct nine-digit string.

TrOCR is a transformer trained on real handwriting, which is the class of model
suited to slanted, connected pen strokes. Encoder + merged decoder are run
directly through onnxruntime (no torch), with plain greedy decoding: the strings
we need are ~9 tokens, so the KV cache is not worth the complexity.
"""
from __future__ import annotations

import json
import os
import re

import cv2
import numpy as np
import onnxruntime as ort

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "trocr")
_ENC = os.path.join(_DIR, "encoder_model_quantized.onnx")
_DEC = os.path.join(_DIR, "decoder_model_merged_quantized.onnx")
_TOK = os.path.join(_DIR, "tokenizer.json")

N_LAYERS = 6
N_HEADS = 8
HEAD_DIM = 32
START_ID = 2
EOS_ID = 2
IMG = 384


class TrOCR:
    def __init__(self, threads: int = 1):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.enc = ort.InferenceSession(_ENC, so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(_DEC, so, providers=["CPUExecutionProvider"])
        tok = json.load(open(_TOK, encoding="utf-8"))
        vocab = tok["model"]["vocab"]
        # Unigram vocab is a list of [token, score]; BPE would be a dict
        if isinstance(vocab, list):
            self.id2tok = {i: (e[0] if isinstance(e, (list, tuple)) else e)
                           for i, e in enumerate(vocab)}
        else:
            self.id2tok = {int(v): k for k, v in vocab.items()}

    # ---------------------------------------------------------------- image
    @staticmethod
    def _preprocess(gray: np.ndarray) -> np.ndarray:
        """TrOCR expects 384x384 RGB normalised to [-1, 1]."""
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        rgb = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_CUBIC)
        x = rgb.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        return np.transpose(x, (2, 0, 1))[None]

    # ---------------------------------------------------------------- decode
    def _empty_past(self, enc_len: int):
        past = {}
        for l in range(N_LAYERS):
            past[f"past_key_values.{l}.decoder.key"] = np.zeros((1, N_HEADS, 0, HEAD_DIM), np.float32)
            past[f"past_key_values.{l}.decoder.value"] = np.zeros((1, N_HEADS, 0, HEAD_DIM), np.float32)
            past[f"past_key_values.{l}.encoder.key"] = np.zeros((1, N_HEADS, enc_len, HEAD_DIM), np.float32)
            past[f"past_key_values.{l}.encoder.value"] = np.zeros((1, N_HEADS, enc_len, HEAD_DIM), np.float32)
        return past

    def read(self, gray: np.ndarray, max_len: int = 20):
        """Return (text, mean_token_probability)."""
        hidden = self.enc.run(None, {"pixel_values": self._preprocess(gray)})[0]
        enc_len = hidden.shape[1]
        ids = [START_ID]
        probs = []
        for _ in range(max_len):
            feed = {"input_ids": np.array([ids], np.int64),
                    "encoder_hidden_states": hidden,
                    "use_cache_branch": np.array([False])}
            feed.update(self._empty_past(enc_len))
            logits = self.dec.run(["logits"], feed)[0][0, -1]
            e = np.exp(logits - logits.max())
            p = e / e.sum()
            nid = int(p.argmax())
            if nid == EOS_ID and len(ids) > 1:
                break
            probs.append(float(p[nid]))
            ids.append(nid)
        toks = [self.id2tok.get(i, "") for i in ids[1:]]
        text = "".join(toks).replace("▁", " ").replace("Ġ", " ").strip()
        return text, (float(np.mean(probs)) if probs else 0.0)

    def read_digits(self, gray: np.ndarray, want: int = 9):
        """Read and keep only digits. Returns (digits, confidence, raw_text)."""
        text, conf = self.read(gray)
        digits = re.sub(r"\D", "", text)
        return digits, conf, text
