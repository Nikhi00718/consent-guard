"""Kaggle GPU entry point for one ConsentGuard model component.

Edit COMPONENT to baseline, face, plate, handwriting, or all. Separate Kaggle
sessions are recommended because every run checkpoints independently.
"""

from __future__ import annotations

import os
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image, ImageOps


PACKAGED_COMPONENT = "baseline"
COMPONENT_FILE = Path(__file__).with_name("component.txt")
DEFAULT_COMPONENT = (
    COMPONENT_FILE.read_text(encoding="utf-8").strip()
    if COMPONENT_FILE.is_file()
    else PACKAGED_COMPONENT
)
COMPONENT = os.environ.get("CONSENTGUARD_COMPONENT", DEFAULT_COMPONENT)
SEEDS = os.environ.get("CONSENTGUARD_SEEDS", "1337")
REPO = Path("/kaggle/working/consentguard")
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
PYTORCH_CU118_INDEX = "https://download.pytorch.org/whl/cu118"


def find_mount(fragment: str) -> Path:
    matches = sorted(
        (path for path in INPUT.rglob("*") if path.is_dir() and fragment in path.name.lower()),
        key=lambda path: (len(path.parts), path.as_posix()),
    )
    if not matches:
        available = sorted(path.name for path in INPUT.iterdir() if path.is_dir())
        raise FileNotFoundError(
            f"Attach a Kaggle Dataset whose mount contains {fragment!r}; top-level inputs: {available}"
        )
    return matches[0]


def find_content_root(mount: Path, relative: Path) -> Path:
    direct = mount / relative
    if direct.exists():
        return mount
    matches = sorted(
        path.parent.parent
        for path in mount.rglob(relative.name)
        if path.is_dir() and path.parent.name == relative.parent.name
    )
    if not matches:
        raise FileNotFoundError(f"Could not find {relative.as_posix()} below {mount}")
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_image_identity(path: Path) -> tuple[str, int, int]:
    """Hash canonical RGB pixels so metadata-only mirror changes remain safe."""

    with Image.open(path) as image:
        canonical = ImageOps.exif_transpose(image).convert("RGB")
        digest = hashlib.sha256(canonical.tobytes()).hexdigest()
        width, height = canonical.size
    return digest, width, height


def prepare_baseline_transport() -> Path:
    """Link train/validation images and overlay the private mirror patch, then verify."""

    records_mount = find_mount("consentguard-v2-records")
    records_root = find_content_root(records_mount, Path("data/processed"))
    mirror_mount = find_mount("visual-redactions")
    # The public mirror is immutable but has 48 files whose decoded pixels no
    # longer match the frozen V2 manifest.  A tiny private patch dataset carries
    # the approved bytes for those files; when absent, the public mirror remains
    # the source and the verifier fails closed as before.
    try:
        patch_mount = find_mount("consentguard-v2-image-patch")
    except FileNotFoundError:
        patch_mount = None
    train_candidates = sorted(
        path for path in mirror_mount.rglob("train") if path.is_dir() and path.parent.name == "images"
    )
    val_candidates = sorted(
        path for path in mirror_mount.rglob("val") if path.is_dir() and path.parent.name == "images"
    )
    if len(train_candidates) != 1 or len(val_candidates) != 1:
        raise RuntimeError(
            f"Expected one train and one val image directory; got train={train_candidates}, val={val_candidates}"
        )
    image_root = REPO / "data/raw/visual_redactions/images"
    image_root.mkdir(parents=True, exist_ok=True)
    for split, source in (("train2017", train_candidates[0]), ("val2017", val_candidates[0])):
        target = image_root / split
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"Refusing to replace baseline image path: {target}")
        target.mkdir(parents=True, exist_ok=False)
        for source_image in sorted(source.iterdir()):
            if not source_image.is_file():
                continue
            # Kaggle datasets strip the local ``data/`` prefix when they are
            # mounted.  Accept both layouts so the private patch actually
            # overlays the 48 approved files instead of silently falling back
            # to the public mirror.
            patch_relatives = (
                Path("data/raw/visual_redactions/images") / split / source_image.name,
                Path("raw/visual_redactions/images") / split / source_image.name,
            )
            patched = None
            if patch_mount is not None:
                for patch_relative in patch_relatives:
                    candidate = patch_mount / patch_relative
                    if candidate.is_file():
                        patched = candidate
                        break
            destination = target / source_image.name
            if patched is not None and patched.is_file():
                shutil.copy2(patched, destination)
            else:
                destination.symlink_to(source_image)

    manifests = sorted(records_mount.rglob("kaggle_trainval_data_manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one train/validation manifest, found: {manifests}")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("test_split_used") is not False:
        raise RuntimeError("Baseline transport manifest does not keep the test split locked")
    images = manifest.get("images", [])
    mismatches = []
    byte_matches = 0
    pixel_matches = 0
    for index, item in enumerate(images, start=1):
        relative = Path(item["path"])
        if "test" in relative.as_posix().lower():
            raise RuntimeError(f"Test path present in baseline transport manifest: {relative}")
        image = REPO / relative
        actual = sha256(image)
        if actual.lower() == item["sha256"].lower():
            byte_matches += 1
        else:
            expected_pixels = item.get("pixel_sha256")
            if not expected_pixels:
                mismatches.append(
                    {"path": relative.as_posix(), "reason": "manifest_missing_pixel_sha256", "sha256": actual}
                )
            else:
                pixel_sha256, width, height = decoded_image_identity(image)
                if (
                    pixel_sha256.lower() == str(expected_pixels).lower()
                    and width == int(item["width"])
                    and height == int(item["height"])
                ):
                    pixel_matches += 1
                else:
                    mismatches.append(
                        {
                            "path": relative.as_posix(),
                            "reason": "decoded_pixels_or_geometry_changed",
                            "sha256": actual,
                            "pixel_sha256": pixel_sha256,
                            "width": width,
                            "height": height,
                        }
                    )
        if index % 500 == 0 or index == len(images):
            print(f"Verified baseline transport images: {index}/{len(images)}", flush=True)
    verification = {
        "schema_version": "consentguard-baseline-transport-verification-v2",
        "images": len(images),
        "byte_matches": byte_matches,
        "pixel_matches": pixel_matches,
        "mismatches": mismatches,
        "test_split_used": False,
    }
    report_path = WORK / "baseline_transport_verification.json"
    report_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: verification[key] for key in ("images", "byte_matches", "pixel_matches")}), flush=True)
    if mismatches:
        raise RuntimeError(
            f"Baseline mirror changed decoded content for {len(mismatches)} images; "
            f"see {report_path} for the complete fail-closed report"
        )
    return records_root


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        return
    print(f"Downloading {url} -> {destination}", flush=True)
    urllib.request.urlretrieve(url, destination)


def prepare_hiertext() -> Path:
    root = WORK / "hiertext"
    gt = root / "gt"
    gt.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        download(
            f"https://raw.githubusercontent.com/google-research-datasets/hiertext/main/gt/{split}.jsonl.gz",
            gt / f"{split}.jsonl.gz",
        )
        archive = root / f"{split}.tgz"
        download(f"https://open-images-dataset.s3.amazonaws.com/ocr/{split}.tgz", archive)
        marker = root / f".{split}-extracted"
        if not marker.exists():
            with tarfile.open(archive, "r:gz") as handle:
                handle.extractall(root, filter="data")
            marker.write_text("ok\n", encoding="utf-8")
    return root


def ensure_torch_gpu_compatibility() -> None:
    """Pin an official Pascal-compatible build only when Kaggle's build omits the assigned GPU."""

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "cap=torch.cuda.get_device_capability(); "
                "arch=f'sm_{cap[0]}{cap[1]}'; "
                "raise SystemExit(0 if arch in torch.cuda.get_arch_list() else 3)"
            ),
        ],
        check=False,
    )
    if probe.returncode == 0:
        return
    print(
        "Kaggle's preinstalled PyTorch does not contain kernels for the assigned GPU; "
        "installing the official CUDA 11.8 compatibility pair.",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            "--force-reinstall",
            "torch==2.7.1+cu118",
            "torchvision==0.22.1+cu118",
            "--index-url",
            PYTORCH_CU118_INDEX,
            "--extra-index-url",
            "https://pypi.org/simple",
        ],
        check=True,
    )


if not REPO.is_dir():
    code_archives = sorted(INPUT.rglob("consentguard-training-code.zip"))
    unpacked_roots = sorted(
        path.parent
        for path in INPUT.rglob("pyproject.toml")
        if (path.parent / "main_project").is_dir()
    )
    if code_archives:
        REPO.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(code_archives[0]) as archive:
            archive.extractall(REPO)
    elif unpacked_roots:
        shutil.copytree(unpacked_roots[0], REPO)
    else:
        raise FileNotFoundError(
            "Attach the private consentguard-training-code Kaggle Dataset; neither its archive nor unpacked tree was found"
        )

ensure_torch_gpu_compatibility()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO)], check=True)
command = [
    sys.executable,
    str(REPO / "main_project/scripts/stage_02_baseline_model/run_kaggle_training.py"),
    "--component",
    COMPONENT,
    "--seeds",
    *SEEDS.split(),
]
if COMPONENT in {"baseline", "all"}:
    command.extend(("--data-root", str(prepare_baseline_transport())))
if COMPONENT in {"face", "all"}:
    command.extend(("--face-root", str(find_mount("wider-face"))))
if COMPONENT in {"plate", "all"}:
    command.extend(("--plate-root", str(find_mount("indian-license-plates"))))
if COMPONENT in {"handwriting", "all"}:
    command.extend(("--handwriting-root", str(prepare_hiertext())))

print("Running:", " ".join(command), flush=True)
subprocess.run(command, cwd=REPO, check=True)
