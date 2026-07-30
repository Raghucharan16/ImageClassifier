"""
One-time per-machine setup: install the onnxruntime build that matches this
machine's hardware and remove the ones that do not.

Why this is a separate script rather than something the pipeline does itself:
the three onnxruntime distributions (onnxruntime, onnxruntime-gpu,
onnxruntime-directml) all install the SAME importable package name. Having two
of them present leaves a half-overwritten install whose available providers
depend on which was unpacked last, so exactly one must be installed -- and
swapping them means a pip install/uninstall, which cannot happen from inside a
frozen EXE (there is no pip in the bundle) and must not happen mid-run in any
case. So the environment is fixed once, here, and the pipeline itself only
DETECTS what it was given (see ocr_engine.cuda_available).

Run once after copying the project to a new machine:

    python setup_runtime.py            # detect, then install the right wheel
    python setup_runtime.py --dry-run  # report what it would change
    python setup_runtime.py --cpu      # force the CPU wheel
    python setup_runtime.py --gpu      # force the CUDA wheel

An NVIDIA GPU is detected via nvidia-smi, which ships with the driver. If one
is found, onnxruntime-gpu is installed; otherwise plain onnxruntime. DirectML is
never selected automatically: it was measured on this project's models against
an Intel Arc integrated GPU at 8.47s/page versus 8.57s on CPU, i.e. no gain for
the extra moving part.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

# All distributions that provide the `onnxruntime` module. Only one may be
# installed at a time.
VARIANTS = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml")

GPU_PKG = "onnxruntime-gpu"
CPU_PKG = "onnxruntime"


def _pip(*args: str) -> int:
    """Run pip in THIS interpreter's environment."""
    cmd = [sys.executable, "-m", "pip", *args]
    print(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd)


def has_nvidia_gpu() -> tuple[bool, str]:
    """(found, description) for an NVIDIA GPU, via the driver's nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False, "nvidia-smi not found (no NVIDIA driver installed)"
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except Exception as exc:
        return False, f"nvidia-smi failed: {exc}"
    if out.returncode != 0:
        return False, f"nvidia-smi returned {out.returncode}"
    name = out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ""
    return (True, name) if name else (False, "nvidia-smi listed no GPU")


def installed_variants() -> list[str]:
    """Which onnxruntime distributions are currently installed."""
    try:
        from importlib.metadata import distribution
    except ImportError:
        return []
    found = []
    for v in VARIANTS:
        try:
            distribution(v)
            found.append(v)
        except Exception:
            pass
    return found


def runtime_healthy() -> bool:
    """True when onnxruntime imports AND exposes a usable provider list.

    Checked in a SUBPROCESS because a broken onnxruntime can only be diagnosed
    by importing it, and a failed import may leave this interpreter's module
    cache poisoned -- which would then make the post-repair verification lie.

    Presence in pip's list is not sufficient evidence of health: these
    distributions share file paths, so uninstalling a sibling deletes binaries
    the survivor also owns, leaving a package that imports but has lost
    get_available_providers entirely.
    """
    code = (
        "import onnxruntime as o; "
        "assert o.get_available_providers(); "
        "print('ok')"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=120)
        return r.returncode == 0 and "ok" in r.stdout
    except Exception:
        return False


def report_providers() -> None:
    """Print the execution providers onnxruntime actually offers now."""
    try:
        import onnxruntime as ort
    except Exception as exc:
        print(f"  ! onnxruntime does not import: {exc}")
        return
    provs = ort.get_available_providers()
    print(f"  onnxruntime {ort.__version__}")
    print(f"  providers: {', '.join(provs)}")
    if "CUDAExecutionProvider" in provs:
        print("  -> pipeline will run OCR on the GPU")
    else:
        print("  -> pipeline will run OCR on the CPU")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Install the onnxruntime build matching this machine")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gpu", action="store_true", help="force the CUDA build")
    g.add_argument("--cpu", action="store_true", help="force the CPU build")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the plan without changing anything")
    a = ap.parse_args()

    print("=" * 64)
    print("Hardware")
    print("=" * 64)
    found, desc = has_nvidia_gpu()
    print(f"  NVIDIA GPU: {'yes -- ' + desc if found else 'no  -- ' + desc}")

    if a.gpu:
        want, why = GPU_PKG, "forced by --gpu"
    elif a.cpu:
        want, why = CPU_PKG, "forced by --cpu"
    else:
        want = GPU_PKG if found else CPU_PKG
        why = "NVIDIA GPU detected" if found else "no NVIDIA GPU detected"

    if want == GPU_PKG and not found:
        print("\n  ! Requesting the CUDA build on a machine with no detected NVIDIA")
        print("    GPU. It will install, but CUDA will be unavailable at runtime")
        print("    and the pipeline will silently fall back to the CPU.")

    have = installed_variants()
    remove = [v for v in have if v != want]
    healthy = runtime_healthy() if want in have else False
    # A target that is installed but broken must be rewritten, not left alone.
    repair = want in have and not healthy

    print()
    print("=" * 64)
    print("Plan")
    print("=" * 64)
    print(f"  installed now : {', '.join(have) if have else '(none)'}")
    print(f"  target        : {want}  ({why})")
    print(f"  to uninstall  : {', '.join(remove) if remove else '(none)'}")
    print(f"  target health : {'ok' if healthy else 'BROKEN -- will reinstall'}")

    if want in have and not remove and healthy:
        print("\n  Already correct -- nothing to change.")
        print()
        print("=" * 64)
        print("Result")
        print("=" * 64)
        report_providers()
        return 0

    if a.dry_run:
        print("\n  --dry-run: no changes made.")
        return 0

    print()
    print("=" * 64)
    print("Applying")
    print("=" * 64)

    # Uninstall first: these distributions overwrite each other's files, so
    # installing over one leaves a mixture whose providers depend on unpack order.
    for v in remove:
        if _pip("uninstall", "-y", v) != 0:
            print(f"  ! failed to uninstall {v}")
            return 1

    # Reinstall the target whenever a sibling was removed, even if pip still
    # lists the target as installed. Because these distributions share file
    # paths, uninstalling one DELETES binaries the survivor also owns: removing
    # onnxruntime-directml alongside onnxruntime left the latter importable but
    # gutted, with get_available_providers() missing entirely. --force-reinstall
    # rewrites the full file set and repairs that.
    if remove or repair:
        if _pip("install", "--force-reinstall", "--no-cache-dir", want) != 0:
            print(f"  ! failed to reinstall {want}")
            return 1
    elif want not in have:
        if _pip("install", want) != 0:
            print(f"  ! failed to install {want}")
            return 1

    print()
    print("=" * 64)
    print("Result")
    print("=" * 64)
    report_providers()
    print()
    print("  Done. The pipeline auto-detects this at startup -- no flags needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
