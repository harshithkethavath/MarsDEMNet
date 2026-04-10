#!/bin/bash

# ─────────────────────────────────────────────
# download_data.sh
# Downloads MCTED dataset from HuggingFace
#
# Usage:
#   bash download_data.sh --full
#   bash download_data.sh --sample 100
# ─────────────────────────────────────────────

set -e

REPO_ID="ESA-Datalabs/MCTED"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR"

MODE=""
N_PATCHES=""

# ── Parse arguments ───────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)
            MODE="full"
            shift
            ;;
        --sample)
            MODE="sample"
            N_PATCHES="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash download_data.sh --full"
            echo "       bash download_data.sh --sample <N>"
            exit 1
            ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "Error: must specify --full or --sample <N>"
    echo "Usage: bash download_data.sh --full"
    echo "       bash download_data.sh --sample <N>"
    exit 1
fi

# ── HuggingFace token ─────────────────────────
if [[ -z "$HF_TOKEN" ]]; then
    echo "Warning: HF_TOKEN is not set. Attempting unauthenticated download."
    echo "If this fails, run: export HF_TOKEN=your_token_here"
else
    echo "HF_TOKEN found."
fi

# ── Full download ─────────────────────────────
if [[ "$MODE" == "full" ]]; then
    echo "Downloading full MCTED dataset to $DATA_DIR ..."
    python3 - <<EOF
from huggingface_hub import hf_hub_download, list_repo_files
import tarfile, os
from pathlib import Path

repo_id   = "$REPO_ID"
local_dir = Path("$DATA_DIR")
token     = os.environ.get("HF_TOKEN")

all_files = list(list_repo_files(repo_id, repo_type="dataset", token=token))
tar_files = sorted([f for f in all_files if f.endswith(".tar")])

print(f"Found {len(tar_files)} tar files. Downloading and extracting...")

for i, tar_file in enumerate(tar_files):
    split = "train" if "train" in tar_file else "val"
    extract_dir = local_dir / split
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{i+1}/{len(tar_files)}] {tar_file}")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=tar_file,
        repo_type="dataset",
        local_dir=str(local_dir / "tars"),
        token=token,
    )
    with tarfile.open(path) as tar:
        tar.extractall(path=extract_dir)
    os.remove(path)

print("Full download and extraction complete.")
EOF

# ── Sample download ───────────────────────────
elif [[ "$MODE" == "sample" ]]; then
    echo "Downloading $N_PATCHES sample patches from MCTED to $DATA_DIR ..."
    python3 - <<EOF
from huggingface_hub import hf_hub_download
import tarfile, os
from pathlib import Path
from collections import defaultdict

repo_id   = "$REPO_ID"
local_dir = Path("$DATA_DIR")
token     = os.environ.get("HF_TOKEN")
n_patches = int("$N_PATCHES")

# Sample pulls from train-000.tar only
tar_filename = "data/train-000.tar"
extract_dir  = local_dir / "sample"
extract_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading {tar_filename}...")
tar_path = hf_hub_download(
    repo_id=repo_id,
    filename=tar_filename,
    repo_type="dataset",
    local_dir=str(local_dir / "tars"),
    token=token,
)

print(f"Extracting first {n_patches} patches...")
with tarfile.open(tar_path) as tar:
    members = tar.getmembers()

    # Group members by patch base name (strip the file extension suffix)
    patch_groups = defaultdict(list)
    for m in members:
        base = m.name.split(".", 1)[0]   # e.g. <scene>_00003
        patch_groups[base].append(m)

    # Select first N patch groups
    patch_bases  = sorted(patch_groups.keys())
    selected     = patch_bases[:n_patches]
    to_extract   = [m for base in selected for m in patch_groups[base]]

    print(f"  Total patches in tar : {len(patch_bases)}")
    print(f"  Patches to extract   : {len(selected)}")
    print(f"  Files to extract     : {len(to_extract)}")

    tar.extractall(path=extract_dir, members=to_extract)

# Clean up tar
os.remove(tar_path)
# Remove tars folder entirely
import shutil
tars_dir = local_dir / "tars"
if tars_dir.exists():
    shutil.rmtree(tars_dir)

print(f"Sample extraction complete → {extract_dir}")
EOF
fi