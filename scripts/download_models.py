#!/usr/bin/env python3
"""Download ML models for offline/airgapped deployments.

This script downloads the following models:
- FastEmbed embedding model (BAAI/bge-small-en-v1.5)
- Tiktoken encodings (cl100k_base)
- LLMLingua-2 ONNX model

Models are architecture-independent (ONNX format) and can be pre-downloaded
once and used across different CPU architectures (amd64, arm64).
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import tiktoken
from fastembed import TextEmbedding

# Default paths matching config.py defaults and Dockerfile expectations
DEFAULT_MODELS_DIR = Path(__file__).parent.parent / "models"
DEFAULT_FASTEMBED_PATH = DEFAULT_MODELS_DIR / "fastembed"
DEFAULT_TIKTOKEN_PATH = DEFAULT_MODELS_DIR / "tiktoken"
DEFAULT_LLMLINGUA_PATH = DEFAULT_MODELS_DIR / "llmlingua"

# Model identifiers
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
TIKTOKEN_ENCODING = "cl100k_base"
LLMLINGUA_HF_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
LLMLINGUA_ONNX_FOLDER = "llmlingua2-onnx"


def check_fastembed_model(cache_path: Path) -> bool:
    """Check if fastembed model is already downloaded."""
    # FastEmbed creates a subdirectory based on model name
    # The model files are in a subdirectory like "models--BAAI--bge-small-en-v1.5"
    if not cache_path.exists():
        return False

    # Check for ONNX model file in any subdirectory
    for subdir in cache_path.iterdir():
        if subdir.is_dir():
            onnx_files = list(subdir.rglob("*.onnx"))
            if onnx_files:
                return True
    return False


def check_tiktoken_encoding(cache_path: Path) -> bool:
    """Check if tiktoken encoding is already downloaded."""
    if not cache_path.exists():
        return False

    # Tiktoken caches files with hash-based names
    # Check if any files exist in the cache directory
    files = list(cache_path.iterdir())
    return len(files) > 0


def check_llmlingua_model(cache_path: Path) -> bool:
    """Check if LLMLingua ONNX model is already exported."""
    onnx_dir = cache_path / LLMLINGUA_ONNX_FOLDER
    if not onnx_dir.exists():
        return False

    # Check for the main model.onnx file
    model_file = onnx_dir / "model.onnx"
    return model_file.exists()


def download_fastembed_model(cache_path: Path) -> bool:
    """Download FastEmbed model."""
    print(f"Downloading FastEmbed model '{FASTEMBED_MODEL}'...")
    cache_path.mkdir(parents=True, exist_ok=True)

    try:
        # Import fastembed and download the model
        os.environ["FASTEMBED_CACHE_PATH"] = str(cache_path)

        # Instantiating TextEmbedding triggers the download
        _ = TextEmbedding(model_name=FASTEMBED_MODEL, cache_dir=str(cache_path))
        print(f"FastEmbed model downloaded to {cache_path}")
        return True
    except ImportError:
        print("ERROR: fastembed not installed. Run: uv sync --dev")
        return False
    except Exception as e:
        print(f"ERROR: Failed to download FastEmbed model: {e}")
        return False


def download_tiktoken_encoding(cache_path: Path) -> bool:
    """Download tiktoken encoding."""
    print(f"Downloading tiktoken encoding '{TIKTOKEN_ENCODING}'...")
    cache_path.mkdir(parents=True, exist_ok=True)

    try:
        os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_path)

        # Getting the encoding triggers the download
        _ = tiktoken.get_encoding(TIKTOKEN_ENCODING)
        print(f"Tiktoken encoding downloaded to {cache_path}")
        return True
    except ImportError:
        print("ERROR: tiktoken not installed. Run: uv sync --dev")
        return False
    except Exception as e:
        print(f"ERROR: Failed to download tiktoken encoding: {e}")
        return False


def export_llmlingua_model(cache_path: Path) -> bool:
    """Export LLMLingua model to ONNX format using optimum exporter."""
    print(f"Exporting LLMLingua model '{LLMLINGUA_HF_MODEL}' to ONNX...")
    cache_path.mkdir(parents=True, exist_ok=True)

    output_dir = cache_path / LLMLINGUA_ONNX_FOLDER

    try:
        # Use optimum.exporters.onnx module to export the model
        # Command: python -m optimum.exporters.onnx -m MODEL output_dir --task TASK
        cmd = [
            sys.executable,
            "-m",
            "optimum.exporters.onnx",
            "-m",
            LLMLINGUA_HF_MODEL,
            "--task",
            "token-classification",
            str(output_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            print(f"ERROR: optimum export failed: {result.stderr}")
            return False

        print(f"LLMLingua ONNX model exported to {output_dir}")
        return True
    except FileNotFoundError:
        print("ERROR: optimum not installed. Run: uv sync --dev")
        return False
    except Exception as e:
        print(f"ERROR: Failed to export LLMLingua model: {e}")
        return False


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Download ML models for offline/airgapped deployments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models downloaded:
  - FastEmbed: BAAI/bge-small-en-v1.5 (embedding model)
  - Tiktoken: cl100k_base (tokenizer encoding)
  - LLMLingua: microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank (ONNX)

These models are architecture-independent and can be used across amd64/arm64.
        """,
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Base directory for all models (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--fastembed-path",
        type=Path,
        default=None,
        help="Override path for FastEmbed models",
    )
    parser.add_argument(
        "--tiktoken-path",
        type=Path,
        default=None,
        help="Override path for tiktoken encodings",
    )
    parser.add_argument(
        "--llmlingua-path",
        type=Path,
        default=None,
        help="Override path for LLMLingua ONNX model",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if models exist",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing models before downloading",
    )
    return parser


def _clean_model_directories(paths: list[Path]) -> None:
    """Remove existing model directories."""
    print("Cleaning existing models...")
    for path in paths:
        if path.exists():
            shutil.rmtree(path)
            print(f"  Removed {path}")


def _download_all_models(
    fastembed_path: Path, tiktoken_path: Path, llmlingua_path: Path, force: bool
) -> bool:
    """Download all models and return success status."""
    all_success = True

    # Download FastEmbed model
    if force or not check_fastembed_model(fastembed_path):
        if not download_fastembed_model(fastembed_path):
            all_success = False
    else:
        print(f"FastEmbed model already exists at {fastembed_path}")

    # Download tiktoken encoding
    if force or not check_tiktoken_encoding(tiktoken_path):
        if not download_tiktoken_encoding(tiktoken_path):
            all_success = False
    else:
        print(f"Tiktoken encoding already exists at {tiktoken_path}")

    # Export LLMLingua model
    if force or not check_llmlingua_model(llmlingua_path):
        if not export_llmlingua_model(llmlingua_path):
            all_success = False
    else:
        print(f"LLMLingua ONNX model already exists at {llmlingua_path}")

    return all_success


def main() -> int:
    """Main entry point."""
    parser = _create_argument_parser()
    args = parser.parse_args()

    # Determine paths
    fastembed_path = args.fastembed_path or args.models_dir / "fastembed"
    tiktoken_path = args.tiktoken_path or args.models_dir / "tiktoken"
    llmlingua_path = args.llmlingua_path or args.models_dir / "llmlingua"

    # Clean if requested
    if args.clean:
        _clean_model_directories([fastembed_path, tiktoken_path, llmlingua_path])

    all_success = _download_all_models(fastembed_path, tiktoken_path, llmlingua_path, args.force)

    if all_success:
        print("\nAll models downloaded successfully!")
        print("\nModel locations:")
        print(f"  FastEmbed:  {fastembed_path}")
        print(f"  Tiktoken:   {tiktoken_path}")
        print(f"  LLMLingua:  {llmlingua_path}")
        return 0

    print("\nSome models failed to download. Check errors above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
