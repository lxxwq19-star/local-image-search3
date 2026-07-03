#!/usr/bin/env python3
"""
Download CLIP models for offline use.
Supports Chinese mirror (hf-mirror.com) for faster download.

Usage:
    python download_models.py              # download both models
    python download_models.py --siglip2-only  # only SigLIP2
    python download_models.py --clipl-only    # only CLIP-L/14
"""

import os
import sys
import argparse

def download_model(repo_id, local_dir, mirror=True):
    """Download model from Hugging Face."""
    print(f"\n{'='*60}")
    print(f"Downloading: {repo_id}")
    print(f"To: {local_dir}")
    print(f"{'='*60}")

    try:
        from huggingface_hub import snapshot_download, configure_http_backend
        import requests

        # Use hf-mirror.com if mirror=True
        if mirror:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            print(f"[Mirror] Using https://hf-mirror.com")

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # copy files, don't symlink
            resume_download=True,
        )
        print(f"✅ Downloaded {repo_id} to {local_dir}")
        return True
    except ImportError:
        print(f"❌ huggingface_hub not installed. Install with: pip install huggingface-hub")
        return False
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download CLIP models for offline use")
    parser.add_argument("--siglip2-only", action="store_true", help="Only download SigLIP2")
    parser.add_argument("--clipl-only", action="store_true", help="Only download CLIP-L/14")
    parser.add_argument("--no-mirror", action="store_true", help="Don't use HF mirror")
    args = parser.parse_args()

    # Detect if running in China (try to connect to hf-mirror.com)
    use_mirror = not args.no_mirror

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    success = True

    # Download SigLIP2-Large-Patch16-256
    if not args.clipl_only:
        siglip2_dir = os.path.join(models_dir, "siglip2-large")
        if not download_model(
            repo_id="google/siglip2-large-patch16-256",
            local_dir=siglip2_dir,
            mirror=use_mirror
        ):
            success = False

    # Download CLIP-ViT-Large-Patch14
    if not args.siglip2_only:
        clipl_dir = os.path.join(models_dir, "clip-large")
        if not download_model(
            repo_id="openai/clip-vit-large-patch14",
            local_dir=clipl_dir,
            mirror=use_mirror
        ):
            success = False

    print(f"\n{'='*60}")
    if success:
        print("✅ All models downloaded successfully!")
        print(f"Models directory: {models_dir}")
        print(f"Total size: ", end="")
        # Calculate total size
        total_size = 0
        for root, dirs, files in os.walk(models_dir):
            for f in files:
                fp = os.path.join(root, f)
                total_size += os.path.getsize(fp)
        print(f"{total_size / (1024**3):.2f} GB")
    else:
        print("⚠ Some models failed to download. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
