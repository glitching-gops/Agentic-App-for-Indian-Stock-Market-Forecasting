"""
tools/upload_model_store.py

Packages trained model artifacts into models_store.zip and pushes them to
the model-store orphan branch on GitHub so Render can download them on deploy.

Run this AFTER a full pipeline training completes (python main.py):

    python tools/upload_model_store.py

What it does:
  1. Zips models/joblib/, models/meta/, models/tft/
  2. Checks out (or creates) an orphan model-store branch
  3. Commits and force-pushes the zip
  4. Returns to the original branch and removes the local zip
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path

ROOT   = Path(__file__).parent.parent
AUTHOR = "glitching-gops"
EMAIL  = "venuworkspace@outlook.com"


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if result.stdout.strip():
        print(f"    {result.stdout.strip()}")
    if result.returncode != 0 and result.stderr.strip():
        print(f"    {result.stderr.strip()}", file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")
    return result


def create_zip() -> Path:
    zip_path = ROOT / "models_store.zip"
    model_dirs = [
        ROOT / "models" / "joblib",
        ROOT / "models" / "meta",
        ROOT / "models" / "tft",
    ]

    total = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for model_dir in model_dirs:
            if not model_dir.exists():
                print(f"  Warning: {model_dir.relative_to(ROOT)} not found — skipping")
                continue
            for f in sorted(model_dir.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(ROOT))
                    total += 1

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  Created models_store.zip — {total} files, {size_mb:.1f} MB")
    return zip_path


def push_to_model_store(zip_path: Path) -> None:
    # Remember where we are
    current = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    print(f"\n  Current branch: {current}")

    # Does model-store exist on remote already?
    remote_exists = bool(
        _run(["git", "ls-remote", "--heads", "origin", "model-store"], check=False)
        .stdout.strip()
    )

    if remote_exists:
        print("  model-store branch exists — checking out to update...")
        _run(["git", "fetch", "origin", "model-store"])
        _run(["git", "checkout", "model-store"])
        _run(["git", "rm", "-f", "models_store.zip"], check=False)
    else:
        print("  Creating new orphan model-store branch...")
        _run(["git", "checkout", "--orphan", "model-store"])
        _run(["git", "reset", "--hard"])

    _run(["git", "add", "models_store.zip"])
    _run([
        "git", "-c", f"user.name={AUTHOR}", "-c", f"user.email={EMAIL}",
        "commit", "-m", "update trained model artifacts",
    ])
    _run(["git", "push", "origin", "model-store", "--force"])

    # Return to original branch and clean up
    _run(["git", "checkout", current])
    zip_path.unlink()
    print("  models_store.zip removed from working directory.")


if __name__ == "__main__":
    print("=== Model Store Upload ===\n")

    joblib_dir = ROOT / "models" / "joblib"
    trained = list(joblib_dir.glob("*.joblib")) if joblib_dir.exists() else []
    if not trained:
        print("ERROR: No trained models found in models/joblib/")
        print("Run the full pipeline first:\n  python main.py")
        sys.exit(1)

    print(f"Found {len(trained)} trained model(s) in models/joblib/\n")
    print("[1/2] Packaging model artifacts...")
    zip_path = create_zip()

    print("\n[2/2] Pushing to model-store branch on GitHub...")
    push_to_model_store(zip_path)

    print("\nDone. Render will download these models on next deploy.")
    print("Trigger a new Render deploy once you have the service set up.")
