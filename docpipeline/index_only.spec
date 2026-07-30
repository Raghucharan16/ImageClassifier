# PyInstaller spec for the standalone index.exe.
# Build from the docpipeline folder:
#   pyinstaller index_only.spec

import os
from pathlib import Path

VENV = Path(r"C:\Users\venka\OneDrive\Desktop\ImageClassifier\.venv")
PIPE = Path(r"C:\Users\venka\OneDrive\Desktop\ImageClassifier\docpipeline")

datas = [
    # rapidocr models (shipped inside its wheel)
    (str(VENV / "Lib/site-packages/rapidocr_onnxruntime"), "rapidocr_onnxruntime"),
    # signature ONNX model
    (str(PIPE / "models/signature/signature.onnx"), "models/signature"),
    # shared modules (moved here from offlineImageClassification)
    (str(PIPE / "ocr_engine.py"),         "."),
    (str(PIPE / "rules.py"),              "."),
    (str(PIPE / "signature_detector.py"), "."),
    (str(PIPE / "paths.py"),              "."),
    (str(PIPE / "appnumber.py"),          "."),
]

block_cipher = None

a = Analysis(
    [str(PIPE / "index_runner.py")],
    pathex=[str(PIPE)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "onnxruntime",
        "onnxruntime.capi._pybind_state",
        "rapidocr_onnxruntime",
        "PIL",
        "PIL.Image",
        "PIL.TiffImagePlugin",   # AppendingTiffWriter: photo page + Group4 pages
        "cv2",
        "numpy",
        "appnumber",
        "tkinter",
        "tkinter.filedialog",
        "tkinter.messagebox",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "torchvision", "tensorflow", "rapid_orientation"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="index",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True,
)
