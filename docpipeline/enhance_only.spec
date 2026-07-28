# PyInstaller spec for the standalone enhance-only EXE.
# Build from the docpipeline folder:
#   pyinstaller enhance_only.spec

import os
from pathlib import Path

VENV = Path(r"C:\Users\venka\OneDrive\Desktop\ImageClassifier\.venv")
PIPE = Path(r"C:\Users\venka\OneDrive\Desktop\ImageClassifier\docpipeline")

# Collect data files that must be present at runtime
datas = [
    # rapid-orientation ONNX model + config
    (str(VENV / "Lib/site-packages/rapid_orientation/models/rapid_orientation.onnx"),
     "rapid_orientation/models"),
    (str(VENV / "Lib/site-packages/rapid_orientation/config.yaml"),
     "rapid_orientation"),
    # thread-capping config used by enhance.py
    (str(PIPE / "orientation_config.yaml"), "."),
    # rapidocr models (shipped inside its wheel)
    (str(VENV / "Lib/site-packages/rapidocr_onnxruntime"), "rapidocr_onnxruntime"),
]

block_cipher = None

a = Analysis(
    [str(PIPE / "enhance_runner.py")],
    pathex=[str(PIPE)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "onnxruntime",
        "onnxruntime.capi._pybind_state",
        "rapidocr_onnxruntime",
        "rapid_orientation",
        "PIL",
        "PIL.Image",
        "cv2",
        "numpy",
        "yaml",
        "tkinter",
        "tkinter.filedialog",
        "tkinter.messagebox",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "torchvision", "tensorflow"],
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
    name="enhance",
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
