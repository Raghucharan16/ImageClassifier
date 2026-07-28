# -*- mode: python ; coding: utf-8 -*-
# Self-contained offline Image Classification .exe.
# Bundles RapidOCR (English ONNX models + configs), onnxruntime, OpenCV.
# No internet / Python / venv needed on the target machine.
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ["rapidocr_onnxruntime", "onnxruntime", "shapely", "pyclipper", "cv2"]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

datas += [("models/signature/signature.onnx", "models/signature")]

hiddenimports += ["paths", "ocr_engine", "rules", "classify_offline",
                  "segregate", "enhance", "signature_detector", "detect_signatures"]

a = Analysis(
    ["run_offline.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["paddle", "paddleocr", "paddlex", "paddle2onnx", "torch",
              "tensorflow", "matplotlib", "pytesseract"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="run_offline",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
