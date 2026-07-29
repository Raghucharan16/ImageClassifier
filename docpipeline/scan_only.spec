# PyInstaller spec for the standalone scan.exe.
# Build from the docpipeline folder:
#   pyinstaller scan_only.spec

from pathlib import Path

VENV = Path(r"C:\Users\venka\OneDrive\Desktop\ImageClassifier\.venv")
PIPE = Path(r"C:\Users\venka\OneDrive\Desktop\ImageClassifier\docpipeline")

datas = [
    (str(VENV / "Lib/site-packages/zxingcpp"), "zxingcpp"),
]

block_cipher = None

a = Analysis(
    [str(PIPE / "scan_runner.py")],
    pathex=[str(PIPE)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "zxingcpp",
        "cv2",
        "numpy",
        "PIL",
        "PIL.Image",
        "tkinter",
        "tkinter.filedialog",
        "tkinter.messagebox",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch", "torchvision", "tensorflow",
        "rapid_orientation", "rapidocr_onnxruntime", "onnxruntime",
    ],
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
    name="scan",
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
