"""
Download ConceptNet Numberbatch Mini
====================================

Downloads the ConceptNet Numberbatch mini embeddings (~150MB).
This provides offline semantic similarity without requiring the full 9GB database.

Usage:
    python download_numberbatch.py

Or from code:
    from sid.download_numberbatch import download_mini
    download_mini()
"""

import os
import sys
import urllib.request
import hashlib
from pathlib import Path


# Download URLs
NUMBERBATCH_URLS = {
    "mini": "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/19.08/mini.h5",
    "en_full": "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz",
    "multilingual": "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-19.08.txt.gz"
}

# File sizes (approximate)
FILE_SIZES = {
    "mini": "~50MB (downloads to ~150MB HDF5)",
    "en_full": "~100MB compressed",
    "multilingual": "~1GB compressed"
}

# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent / "data"


def get_download_progress():
    """Create a download progress reporter."""
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = min(int(count * block_size * 100 / total_size), 100)
            downloaded = count * block_size / (1024 * 1024)  # MB
            total = total_size / (1024 * 1024)
            sys.stdout.write(f"\rDownloading: {percent}% ({downloaded:.1f}/{total:.1f} MB)")
            sys.stdout.flush()
        else:
            downloaded = count * block_size / (1024 * 1024)
            sys.stdout.write(f"\rDownloading: {downloaded:.1f} MB")
            sys.stdout.flush()
    return reporthook


def download_file(url: str, destination: Path, force: bool = False) -> bool:
    """
    Download a file from URL with progress.
    
    Args:
        url: The URL to download from
        destination: The local file path to save to
        force: If True, download even if file exists
    
    Returns:
        True if download successful
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    if destination.exists() and not force:
        print(f"File already exists: {destination}")
        print(f"Use force=True to re-download")
        return True
    
    print(f"Downloading from: {url}")
    print(f"Saving to: {destination}")
    
    try:
        urllib.request.urlretrieve(url, destination, get_download_progress())
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        if destination.exists():
            destination.unlink()
        return False


def download_mini(data_dir: Path = None, force: bool = False) -> Path:
    """
    Download the ConceptNet Numberbatch mini.h5 file.
    
    This is the smallest option (~50MB download, ~150MB on disk).
    It contains semantic embeddings for ~500K concepts.
    
    Args:
        data_dir: Directory to save the file (default: ./sid/data/)
        force: If True, re-download even if file exists
    
    Returns:
        Path to the downloaded file
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    destination = data_dir / "mini.h5"
    
    print("=" * 60)
    print("ConceptNet Numberbatch Mini Downloader")
    print("=" * 60)
    print(f"File size: {FILE_SIZES['mini']}")
    print(f"Contains: Semantic embeddings for ~500K concepts")
    print()
    
    success = download_file(NUMBERBATCH_URLS["mini"], destination, force)
    
    if success:
        size_mb = destination.stat().st_size / (1024 * 1024)
        print(f"\n✓ Successfully downloaded mini.h5 ({size_mb:.1f} MB)")
        print(f"  Location: {destination}")
        return destination
    
    return None


def download_english(data_dir: Path = None, force: bool = False) -> Path:
    """
    Download the English-only Numberbatch text file.
    
    This is a compressed text format (~100MB download).
    Requires gensim or manual parsing to load.
    
    Args:
        data_dir: Directory to save the file
        force: If True, re-download even if file exists
    
    Returns:
        Path to the downloaded file
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    destination = data_dir / "numberbatch-en-19.08.txt.gz"
    
    print("=" * 60)
    print("ConceptNet Numberbatch English Downloader")
    print("=" * 60)
    print(f"File size: {FILE_SIZES['en_full']}")
    print(f"Format: Gzipped text (word2vec format)")
    print()
    
    success = download_file(NUMBERBATCH_URLS["en_full"], destination, force)
    
    if success:
        size_mb = destination.stat().st_size / (1024 * 1024)
        print(f"\n✓ Successfully downloaded ({size_mb:.1f} MB)")
        print(f"  Location: {destination}")
        return destination
    
    return None


def verify_installation(data_dir: Path = None) -> dict:
    """
    Check what Numberbatch files are installed.
    
    Returns:
        Dict with file paths and sizes
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    
    result = {
        "data_dir": str(data_dir),
        "files": {},
        "ready": False
    }
    
    files_to_check = [
        ("mini.h5", "mini"),
        ("numberbatch-en-19.08.txt.gz", "en_full"),
        ("numberbatch-19.08.txt.gz", "multilingual")
    ]
    
    for filename, key in files_to_check:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            result["files"][key] = {
                "path": str(filepath),
                "size_mb": round(size_mb, 2),
                "exists": True
            }
            if key == "mini":
                result["ready"] = True
    
    return result


def print_installation_status():
    """Print current installation status."""
    status = verify_installation()
    
    print("=" * 60)
    print("ConceptNet Numberbatch Installation Status")
    print("=" * 60)
    print(f"Data directory: {status['data_dir']}")
    print()
    
    if status["files"]:
        print("Installed files:")
        for key, info in status["files"].items():
            print(f"  ✓ {key}: {info['size_mb']} MB")
    else:
        print("No Numberbatch files installed.")
    
    print()
    
    if status["ready"]:
        print("✓ Ready for offline use!")
    else:
        print("✗ Not ready. Run download_mini() to install.")
    
    return status


def main():
    """Main entry point for CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download ConceptNet Numberbatch embeddings"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["mini", "en", "full", "status"],
        default="mini",
        help="Type of download: mini (50MB), en (100MB), full (1GB), status (check installation)"
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=None,
        help="Directory to save files (default: ./sid/data/)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if file exists"
    )
    
    args = parser.parse_args()
    
    if args.type == "status":
        print_installation_status()
    elif args.type == "mini":
        download_mini(args.data_dir, args.force)
    elif args.type == "en":
        download_english(args.data_dir, args.force)
    else:
        print(f"Type '{args.type}' not implemented yet.")
        print("Use --type mini for the recommended lightweight option.")


if __name__ == "__main__":
    main()
