# PURPOSE: Set up the MatchAnything third-party dependency for the switchable feature matcher backend.
#          Clones the MatchAnything HuggingFace space and downloads Eloftr model weights.
# INPUTS: None (uses current working directory's third_party/).
# OUTPUTS: MatchAnything code at third_party/MatchAnything (HF space), weights downloaded inside.
#          Prints the config.py snippet to activate the Eloftr backend.
# KEYWORDS: setup, matchanything, eloftr, download, model, weights, dependency, huggingface
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
SPACE_DIR = THIRD_PARTY_DIR / "MatchAnything"
# After cloning the HF space, the actual MatchAnything source lives at:
#   {SPACE_DIR}/imcui/third_party/MatchAnything/
MA_SRC_DIR = SPACE_DIR / "imcui" / "third_party" / "MatchAnything"
WEIGHTS_DIR = MA_SRC_DIR / "weights"
REPO_URL = "https://huggingface.co/spaces/LittleFrog/MatchAnything"
WEIGHTS_REPO = "LittleFrog/MatchAnything_checkpoints"
WEIGHTS_FILE = "matchanything_eloftr.ckpt"


def run_cmd(cmd, cwd=None):
    """Run a command and return True on success."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        return False
    return True


def clone_repo():
    """Clone the MatchAnything HF space into third_party/."""
    if SPACE_DIR.exists():
        print(f"[SKIP] Space already exists at {SPACE_DIR}")
        return True

    print(f"Cloning MatchAnything HF space into {SPACE_DIR} ...")
    THIRD_PARTY_DIR.mkdir(parents=True, exist_ok=True)

    for attempt, args in enumerate([
        (["git", "clone", REPO_URL, str(SPACE_DIR)]),
        (["git", "clone", "--depth", "1", REPO_URL, str(SPACE_DIR)]),
    ]):
        success = run_cmd(args, cwd=str(THIRD_PARTY_DIR))
        if success:
            break
        if attempt == 0:
            print("WARNING: Full clone failed. Trying shallow clone...")

    if not MA_SRC_DIR.exists():
        print("ERROR: Cloned repo but MatchAnything source not found at expected path:")
        print(f"  {MA_SRC_DIR}")
        print("The HF space structure may have changed. Please check manually.")
        return False
    return True


def download_weights():
    """Download model weights from HuggingFace."""
    weights_path = WEIGHTS_DIR / WEIGHTS_FILE
    if weights_path.exists():
        print(f"[SKIP] Weights already exist at {weights_path}")
        return True

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Try huggingface_hub first
    try:
        from huggingface_hub import hf_hub_download
        print(f"Downloading {WEIGHTS_FILE} from {WEIGHTS_REPO} ...")
        hf_hub_download(
            repo_id=WEIGHTS_REPO,
            filename=WEIGHTS_FILE,
            local_dir=str(WEIGHTS_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"  Downloaded to {weights_path}")
        return True
    except ImportError:
        print("  huggingface_hub not installed. Trying wget...")
    except Exception as e:
        print(f"  huggingface_hub download failed: {e}")

    # Fallback: wget
    try:
        import urllib.request
        url = f"https://huggingface.co/{WEIGHTS_REPO}/resolve/main/{WEIGHTS_FILE}"
        print(f"  Downloading from {url}")
        urllib.request.urlretrieve(url, str(weights_path))
        print(f"  Downloaded to {weights_path}")
        return True
    except Exception:
        pass

    # Fallback: wget CLI
    try:
        url = f"https://huggingface.co/{WEIGHTS_REPO}/resolve/main/{WEIGHTS_FILE}"
        success = run_cmd(["wget", "-O", str(weights_path), url])
        if success:
            return True
    except Exception:
        pass

    print("ERROR: Could not download model weights automatically.")
    print(f"Please manually download {WEIGHTS_FILE} from https://huggingface.co/{WEIGHTS_REPO}")
    print(f"and place it at {weights_path}")
    return False


def install_deps():
    """Install additional Python dependencies required by MatchAnything."""
    deps_to_check = [
        ("einops", "einops"),
        ("omegaconf", "omegaconf"),
        ("yacs", "yacs"),
        ("huggingface_hub", "huggingface_hub"),
    ]
    for import_name, pip_name in deps_to_check:
        try:
            __import__(import_name)
            print(f"[OK] {pip_name} already installed")
        except ImportError:
            print(f"Installing {pip_name}...")
            run_cmd([sys.executable, "-m", "pip", "install", pip_name])

    try:
        import torch
        print("[OK] torch already installed")
    except ImportError:
        print("WARNING: PyTorch not found. MatchAnything requires PyTorch.")


def main():
    print("=" * 60)
    print("MatchAnything Eloftr Setup")
    print("=" * 60)

    if not clone_repo():
        print("\nFailed to clone repository. Aborting.")
        return 1

    print("\n--- Checking Dependencies ---")
    install_deps()

    print("\n--- Downloading Model Weights ---")
    download_weights()

    weights_path = WEIGHTS_DIR / WEIGHTS_FILE
    if weights_path.exists():
        # The third_party_dir that feature_matchers.py needs is:
        #   {SPACE_DIR}/imcui/third_party
        tp_dir = SPACE_DIR / "imcui" / "third_party"
        print(f"\n[SUCCESS] MatchAnything Eloftr is ready!")
        print(f"  Space: {SPACE_DIR}")
        print(f"  Source: {MA_SRC_DIR}")
        print(f"  Weights: {weights_path}")
        print(f"\nUpdate config.py with:")
        print(f"  MATCHER_BACKEND = 'matchanything_eloftr'")
        print(f'  MATCHER_BACKEND_KWARGS = {{"third_party_dir": "{tp_dir}"}}')
        return 0
    else:
        print(f"\n[WARNING] Setup incomplete: weights not found at {weights_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
