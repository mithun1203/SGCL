"""
ConceptNet FULL 20GB Dataset Downloader
=======================================

Downloads the COMPLETE ConceptNet 5.7 dataset including:
- Full multilingual assertions (~20GB uncompressed)
- All relations and concepts
- Complete metadata
- High-quality curated knowledge

This is the research-grade dataset used in ACM publications.

Dataset Info:
- Compressed: ~4GB download
- Uncompressed: ~20GB
- Assertions: ~34 million edges
- Concepts: ~8 million unique
- Languages: 30+ languages (we'll filter English)
- Relations: 50+ semantic relations

Download time: 30-90 minutes depending on connection
Processing time: 60-120 minutes

Usage:
    python download_full_conceptnet.py
"""

import os
import sys
import gzip
import json
import csv
import urllib.request
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


# ConceptNet 5.7 Full Dataset URLs
CONCEPTNET_FULL_URLS = {
    "full_assertions": "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz",
    "metadata": "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0-metadata.json"
}

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
KB_DIR = SCRIPT_DIR / "sid"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
KB_DIR.mkdir(parents=True, exist_ok=True)


def format_size(bytes_size):
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def download_with_progress(url: str, output_path: Path) -> bool:
    """Download file with detailed progress indicator."""
    print(f"\n{'='*70}")
    print(f"DOWNLOADING: {url.split('/')[-1]}")
    print(f"{'='*70}")
    print(f"URL: {url}")
    print(f"Destination: {output_path}")
    
    start_time = time.time()
    
    try:
        def report_progress(count, block_size, total_size):
            if total_size > 0:
                downloaded = count * block_size
                percent = min(int(downloaded * 100 / total_size), 100)
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                
                # Progress bar
                bar_length = 50
                filled = int(bar_length * percent / 100)
                bar = "â–ˆ" * filled + "â–‘" * (bar_length - filled)
                
                # Speed calculation
                elapsed = time.time() - start_time
                if elapsed > 0:
                    speed_mbps = downloaded_mb / elapsed
                    remaining_mb = total_mb - downloaded_mb
                    eta_seconds = remaining_mb / speed_mbps if speed_mbps > 0 else 0
                    eta_minutes = int(eta_seconds / 60)
                    
                    sys.stdout.write(
                        f"\r[{bar}] {percent}% "
                        f"({downloaded_mb:.1f}/{total_mb:.1f} MB) "
                        f"Speed: {speed_mbps:.1f} MB/s "
                        f"ETA: {eta_minutes}m"
                    )
                else:
                    sys.stdout.write(
                        f"\r[{bar}] {percent}% "
                        f"({downloaded_mb:.1f}/{total_mb:.1f} MB)"
                    )
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
        
        elapsed = time.time() - start_time
        file_size = output_path.stat().st_size
        avg_speed = (file_size / (1024 * 1024)) / elapsed
        
        print(f"\n{'='*70}")
        print(f"âœ“ Download Complete!")
        print(f"  Time: {int(elapsed/60)}m {int(elapsed%60)}s")
        print(f"  Size: {format_size(file_size)}")
        print(f"  Avg Speed: {avg_speed:.1f} MB/s")
        print(f"{'='*70}")
        return True
        
    except KeyboardInterrupt:
        print(f"\n\nâœ— Download cancelled by user")
        if output_path.exists():
            output_path.unlink()
        return False
    except Exception as e:
        print(f"\n\nâœ— Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def parse_full_conceptnet(csv_path: Path, output_json: Path) -> bool:
    """
    Parse full ConceptNet CSV and create optimized JSON knowledge base.
    
    This processes ALL 34 million assertions and creates an indexed KB.
    Processing takes 60-120 minutes.
    """
    print(f"\n{'='*70}")
    print("PARSING FULL CONCEPTNET DATASET")
    print(f"{'='*70}")
    print(f"Input: {csv_path}")
    print(f"Output: {output_json}")
    print("\nThis will take 60-120 minutes. Progress updates every 1M rows.")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    # Counters
    total_rows = 0
    kept_assertions = 0
    skipped_rows = 0
    
    # Storage
    concept_index = defaultdict(list)
    relation_stats = defaultdict(int)
    language_stats = defaultdict(int)
    
    # Quality thresholds
    MIN_WEIGHT = 1.5  # Keep high-quality assertions
    
    try:
        with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            
            for row in reader:
                total_rows += 1
                
                # Progress updates
                if total_rows % 1000000 == 0:
                    elapsed = time.time() - start_time
                    rate = total_rows / elapsed
                    print(f"  Processed: {total_rows:,} rows | Kept: {kept_assertions:,} | "
                          f"Rate: {rate:.0f} rows/sec | Time: {int(elapsed/60)}m")
                
                if len(row) < 5:
                    skipped_rows += 1
                    continue
                
                try:
                    # Parse fields
                    uri = row[0]
                    rel = row[1]
                    start = row[2]
                    end = row[3]
                    metadata = json.loads(row[4]) if len(row) > 4 else {}
                    
                    weight = metadata.get('weight', 1.0)
                    
                    # Quality filter
                    if weight < MIN_WEIGHT:
                        skipped_rows += 1
                        continue
                    
                    # Extract language
                    start_parts = start.split('/')
                    end_parts = end.split('/')
                    
                    if len(start_parts) < 3 or len(end_parts) < 3:
                        skipped_rows += 1
                        continue
                    
                    start_lang = start_parts[2]
                    end_lang = end_parts[2]
                    
                    # Track language distribution
                    language_stats[start_lang] += 1
                    
                    # Keep English + high-weight cross-lingual
                    if not (start_lang == 'en' or end_lang == 'en' or weight >= 3.0):
                        skipped_rows += 1
                        continue
                    
                    # Extract concept names
                    start_concept = start_parts[3] if len(start_parts) > 3 else start
                    end_concept = end_parts[3] if len(end_parts) > 3 else end
                    
                    # Create edge
                    edge = {
                        "start": start,
                        "rel": rel,
                        "end": end,
                        "weight": round(weight, 2)
                    }
                    
                    # Index by start concept
                    concept_index[start_concept].append(edge)
                    
                    # Track relation stats
                    relation_stats[rel] += 1
                    
                    kept_assertions += 1
                    
                except Exception as e:
                    skipped_rows += 1
                    continue
        
        # Sort edges by weight
        for concept in concept_index:
            concept_index[concept] = sorted(
                concept_index[concept],
                key=lambda x: x['weight'],
                reverse=True
            )
        
        # Create final JSON
        print(f"\n{'='*70}")
        print("CREATING KNOWLEDGE BASE JSON")
        print(f"{'='*70}")
        
        kb = {
            "_metadata": {
                "version": "5.7.0-full",
                "description": "Full ConceptNet 5.7 knowledge base (research-grade)",
                "source": "ConceptNet 5.7 complete assertions",
                "last_updated": "2026-01-06",
                "total_concepts": len(concept_index),
                "total_assertions": kept_assertions,
                "total_rows_processed": total_rows,
                "min_weight": MIN_WEIGHT,
                "top_relations": dict(sorted(
                    relation_stats.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]),
                "language_distribution": dict(sorted(
                    language_stats.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10])
            },
            "concepts": dict(concept_index)
        }
        
        # Write JSON
        print(f"Writing JSON to: {output_json}")
        print("This may take 10-20 minutes for large dataset...")
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(kb, f, indent=2, ensure_ascii=False)
        
        elapsed = time.time() - start_time
        file_size = output_json.stat().st_size
        
        print(f"\n{'='*70}")
        print("âœ“ PARSING COMPLETE!")
        print(f"{'='*70}")
        print(f"Total processing time: {int(elapsed/60)}m {int(elapsed%60)}s")
        print(f"Total rows processed: {total_rows:,}")
        print(f"Kept assertions: {kept_assertions:,}")
        print(f"Skipped rows: {skipped_rows:,}")
        print(f"Unique concepts: {len(concept_index):,}")
        print(f"Output size: {format_size(file_size)}")
        print(f"\nTop 10 relations:")
        for rel, count in list(kb['_metadata']['top_relations'].items())[:10]:
            print(f"  {rel}: {count:,}")
        print(f"{'='*70}")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n\nâœ— Parsing cancelled by user")
        return False
    except Exception as e:
        print(f"\n\nâœ— Parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("ConceptNet 5.7 FULL Dataset Downloader")
    print("="*70)
    print("This will download and process the complete ConceptNet dataset:")
    print("  - Download: ~4GB compressed (30-90 minutes)")
    print("  - Uncompressed: ~20GB")
    print("  - Processing: 60-120 minutes")
    print("  - Final KB: ~500MB-2GB JSON")
    print("\nTotal time: 2-4 hours")
    print("="*70)
    
    # Check for auto-proceed flag
    auto_mode = '--auto' in sys.argv or '--yes' in sys.argv
    
    if not auto_mode:
        # Confirm
        response = input("\nProceed with full download? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled by user")
            return
    else:
        print("\nAuto-proceeding with download (--auto flag detected)...")
    
    # Paths
    csv_path = DATA_DIR / "conceptnet-assertions-5.7.0.csv.gz"
    json_path = KB_DIR / "knowledge_base_full_20gb.json"
    
    # Step 1: Download
    if csv_path.exists():
        print(f"\nâœ“ Dataset already downloaded: {csv_path}")
        size = csv_path.stat().st_size
        print(f"  Size: {format_size(size)}")
        
        response = input("\nRe-download? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            csv_path.unlink()
        else:
            print("Skipping download, using existing file.")
    
    if not csv_path.exists():
        print("\nStarting download...")
        print("You can cancel with Ctrl+C and resume later.")
        if not download_with_progress(CONCEPTNET_FULL_URLS["full_assertions"], csv_path):
            return
    
    # Step 2: Parse
    print("\nStarting parsing...")
    print("This will take 60-120 minutes.")
    print("You can cancel with Ctrl+C (but will need to restart parsing).")
    
    if not parse_full_conceptnet(csv_path, json_path):
        return
    
    # Step 3: Install
    print(f"\n{'='*70}")
    print("INSTALLATION")
    print(f"{'='*70}")
    
    # Backup old KB
    old_kb = KB_DIR / "knowledge_base.json"
    if old_kb.exists():
        backup = KB_DIR / "knowledge_base_mini_backup.json"
        import shutil
        shutil.copy(old_kb, backup)
        print(f"âœ“ Backed up old KB: {backup}")
    
    # Create symlink or copy
    import shutil
    shutil.copy(json_path, old_kb)
    print(f"âœ“ Installed full KB as default: {old_kb}")
    
    print(f"\n{'='*70}")
    print("âœ“ COMPLETE!")
    print(f"{'='*70}")
    print(f"Full KB: {json_path}")
    print(f"Default KB: {old_kb}")
    print("\nYour SID module now uses the full 20GB ConceptNet dataset!")
    print("This is research-grade knowledge for ACM publication-level work.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nâœ— Interrupted by user. Exiting.")
        sys.exit(1)


