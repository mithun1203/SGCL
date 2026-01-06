"""
ConceptNet Full Dataset Downloader & Converter
==============================================

Downloads the FULL ConceptNet 5.7 assertions dataset and converts it to an optimized
local knowledge base for offline use.

ConceptNet Versions:
- Current (mini): 55 concepts, 278 edges (~50KB)
- Full Dataset: ~8 million assertions (~1.5GB compressed, ~9GB uncompressed)
- Optimized Subset: 100K most relevant assertions (~50MB JSON)

This script:
1. Downloads ConceptNet 5.7 assertions (English only for speed)
2. Filters for high-quality, relevant facts (weight > 2.0)
3. Converts to optimized JSON format
4. Creates indexed lookup structure

Usage:
    python upgrade_conceptnet_kb.py --mode full      # Download full dataset
    python upgrade_conceptnet_kb.py --mode optimize  # Create optimized subset
    python upgrade_conceptnet_kb.py --mode both      # Do both (recommended)

Data Sources:
- Full dataset: https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
- English only: Filter by language during processing
- Size: ~1.5GB compressed -> ~9GB uncompressed -> ~50MB optimized
"""

import os
import sys
import gzip
import json
import csv
import urllib.request
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Optional
import argparse


# ConceptNet URLs (version 5.7 - latest stable)
CONCEPTNET_URLS = {
    "full_assertions": "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz",
    "assertions_metadata": "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0-metadata.json",
}

# File sizes
FILE_INFO = {
    "full_assertions": "~1.5GB compressed, ~9GB uncompressed, 8M+ assertions",
    "optimized": "~50MB JSON, 100K high-quality assertions",
    "mini": "~50KB JSON, 278 assertions (current)"
}

# Default paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
KB_DIR = SCRIPT_DIR / "sid"

# Quality thresholds
MIN_WEIGHT = 2.0  # Only keep edges with weight >= 2.0 (higher quality)
MIN_FREQUENCY = 1  # Minimum times a fact appears in sources

# Relations to keep (filter out less useful ones)
IMPORTANT_RELATIONS = {
    "/r/IsA",
    "/r/CapableOf",
    "/r/NotCapableOf",
    "/r/HasProperty",
    "/r/HasA",
    "/r/PartOf",
    "/r/UsedFor",
    "/r/AtLocation",
    "/r/Causes",
    "/r/CausesDesire",
    "/r/HasPrerequisite",
    "/r/HasSubevent",
    "/r/DefinedAs",
    "/r/DerivedFrom",
    "/r/RelatedTo",
    "/r/Antonym",
    "/r/Synonym",
    "/r/DistinctFrom",
    "/r/SimilarTo",
    "/r/MadeOf",
    "/r/LocatedNear",
    "/r/ReceivesAction",
}


def download_with_progress(url: str, output_path: Path) -> bool:
    """Download file with progress indicator."""
    print(f"\nDownloading: {url}")
    print(f"Destination: {output_path}")
    
    try:
        def report_progress(count, block_size, total_size):
            if total_size > 0:
                percent = min(int(count * block_size * 100 / total_size), 100)
                downloaded_mb = count * block_size / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                bar_length = 50
                filled = int(bar_length * percent / 100)
                bar = "=" * filled + "-" * (bar_length - filled)
                sys.stdout.write(f"\r[{bar}] {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
        print("\n✓ Download complete!")
        return True
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False


def parse_conceptnet_csv(csv_path: Path, language: str = "en") -> List[Dict]:
    """
    Parse ConceptNet CSV file and extract relevant assertions.
    
    CSV Format:
    /a/[/r/Relation/,/c/lang/start/,/c/lang/end/],/d/source/,weight
    
    Example:
    /a/[/r/IsA/,/c/en/dog/,/c/en/animal/],/d/wiktionary/en,5.292
    """
    print(f"\nParsing ConceptNet CSV: {csv_path}")
    print("This may take 5-10 minutes for full dataset...")
    
    assertions = []
    skipped = 0
    
    with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        
        for i, row in enumerate(reader):
            if i % 100000 == 0:
                print(f"  Processed {i:,} rows, kept {len(assertions):,} assertions...")
            
            if len(row) < 5:
                continue
            
            try:
                # Parse row
                uri = row[0]
                rel = row[1]
                start = row[2]
                end = row[3]
                metadata = json.loads(row[4]) if len(row) > 4 else {}
                
                weight = metadata.get('weight', 1.0)
                
                # Filter criteria
                if weight < MIN_WEIGHT:
                    skipped += 1
                    continue
                
                # Only keep English concepts
                if not (start.startswith(f'/c/{language}/') and end.startswith(f'/c/{language}/')):
                    skipped += 1
                    continue
                
                # Only keep important relations
                if rel not in IMPORTANT_RELATIONS:
                    skipped += 1
                    continue
                
                # Extract concept names
                start_concept = start.split('/')[3] if len(start.split('/')) > 3 else start
                end_concept = end.split('/')[3] if len(end.split('/')) > 3 else end
                
                assertions.append({
                    "start": start,
                    "rel": rel,
                    "end": end,
                    "weight": weight,
                    "start_concept": start_concept,
                    "end_concept": end_concept,
                    "sources": metadata.get('sources', [])
                })
                
            except Exception as e:
                skipped += 1
                continue
    
    print(f"\n✓ Parsed {len(assertions):,} assertions (skipped {skipped:,})")
    return assertions


def build_concept_index(assertions: List[Dict]) -> Dict[str, List[Dict]]:
    """Build concept-indexed knowledge base."""
    print("\nBuilding concept index...")
    
    concept_index = defaultdict(list)
    
    for assertion in assertions:
        start_concept = assertion['start_concept']
        end_concept = assertion['end_concept']
        
        # Index by start concept
        concept_index[start_concept].append({
            "start": assertion['start'],
            "rel": assertion['rel'],
            "end": assertion['end'],
            "weight": assertion['weight']
        })
        
        # For IsA relations, also index reverse (for finding subclasses)
        if assertion['rel'] == '/r/IsA':
            concept_index[end_concept].append({
                "start": assertion['end'],
                "rel": '/r/HasSubclass',  # Inverse of IsA
                "end": assertion['start'],
                "weight": assertion['weight']
            })
    
    # Convert defaultdict to regular dict and sort by weight
    concept_index = {
        concept: sorted(edges, key=lambda x: x['weight'], reverse=True)
        for concept, edges in concept_index.items()
    }
    
    print(f"✓ Indexed {len(concept_index):,} concepts")
    return concept_index


def create_optimized_kb(concept_index: Dict, output_path: Path, max_concepts: int = 100000):
    """Create optimized knowledge base JSON."""
    print(f"\nCreating optimized KB: {output_path}")
    
    # Sort concepts by number of edges (more connected = more important)
    concept_importance = [
        (concept, len(edges))
        for concept, edges in concept_index.items()
    ]
    concept_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Keep top N concepts
    top_concepts = [c for c, _ in concept_importance[:max_concepts]]
    
    optimized_kb = {
        "_metadata": {
            "version": "2.0.0",
            "description": "Full ConceptNet 5.7 knowledge base (optimized)",
            "source": "ConceptNet 5.7 assertions",
            "last_updated": "2026-01-06",
            "total_concepts": len(top_concepts),
            "total_edges": sum(len(concept_index[c]) for c in top_concepts),
            "min_weight": MIN_WEIGHT,
            "relations": list(IMPORTANT_RELATIONS)
        },
        "concepts": {
            concept: concept_index[concept]
            for concept in top_concepts
        }
    }
    
    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(optimized_kb, f, indent=2, ensure_ascii=False)
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Created optimized KB: {size_mb:.1f} MB")
    print(f"  - {len(top_concepts):,} concepts")
    print(f"  - {optimized_kb['_metadata']['total_edges']:,} edges")
    
    return optimized_kb


def create_statistics(concept_index: Dict) -> Dict:
    """Generate statistics about the knowledge base."""
    stats = {
        "total_concepts": len(concept_index),
        "total_edges": sum(len(edges) for edges in concept_index.values()),
        "relation_distribution": defaultdict(int),
        "top_concepts": [],
        "average_edges_per_concept": 0
    }
    
    # Relation distribution
    for edges in concept_index.values():
        for edge in edges:
            stats["relation_distribution"][edge['rel']] += 1
    
    # Top concepts by edge count
    concept_sizes = [(c, len(edges)) for c, edges in concept_index.items()]
    concept_sizes.sort(key=lambda x: x[1], reverse=True)
    stats["top_concepts"] = concept_sizes[:20]
    
    # Average
    if concept_index:
        stats["average_edges_per_concept"] = stats["total_edges"] / stats["total_concepts"]
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Upgrade ConceptNet knowledge base")
    parser.add_argument(
        "--mode",
        choices=["full", "optimize", "both"],
        default="both",
        help="Download full dataset, optimize existing, or both"
    )
    parser.add_argument(
        "--max-concepts",
        type=int,
        default=100000,
        help="Maximum concepts in optimized KB (default: 100K)"
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language to extract (default: en)"
    )
    
    args = parser.parse_args()
    
    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("ConceptNet Knowledge Base Upgrade")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Language: {args.language}")
    print(f"Max concepts: {args.max_concepts:,}")
    print("="*70)
    
    # Paths
    full_csv_path = DATA_DIR / "conceptnet-assertions-5.7.0.csv.gz"
    optimized_kb_path = KB_DIR / "knowledge_base_full.json"
    stats_path = KB_DIR / "knowledge_base_stats.json"
    
    # Step 1: Download full dataset (if needed)
    if args.mode in ["full", "both"]:
        if full_csv_path.exists():
            print(f"\n✓ Full dataset already exists: {full_csv_path}")
            size_mb = full_csv_path.stat().st_size / (1024 * 1024)
            print(f"  Size: {size_mb:.1f} MB")
        else:
            print("\nDownloading full ConceptNet dataset...")
            print(f"Expected size: {FILE_INFO['full_assertions']}")
            print("This will take 10-30 minutes depending on connection speed.")
            
            if not download_with_progress(CONCEPTNET_URLS["full_assertions"], full_csv_path):
                print("✗ Download failed. Exiting.")
                return
    
    # Step 2: Parse and optimize
    if args.mode in ["optimize", "both"]:
        if not full_csv_path.exists():
            print(f"✗ Full dataset not found: {full_csv_path}")
            print("  Run with --mode full first")
            return
        
        # Parse CSV
        assertions = parse_conceptnet_csv(full_csv_path, language=args.language)
        
        if not assertions:
            print("✗ No assertions found. Exiting.")
            return
        
        # Build index
        concept_index = build_concept_index(assertions)
        
        # Generate statistics
        print("\nGenerating statistics...")
        stats = create_statistics(concept_index)
        
        print("\nKnowledge Base Statistics:")
        print(f"  Total concepts: {stats['total_concepts']:,}")
        print(f"  Total edges: {stats['total_edges']:,}")
        print(f"  Avg edges/concept: {stats['average_edges_per_concept']:.1f}")
        print(f"\n  Top relations:")
        for rel, count in sorted(stats['relation_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {rel}: {count:,}")
        print(f"\n  Top concepts:")
        for concept, edge_count in stats['top_concepts'][:10]:
            print(f"    {concept}: {edge_count:,} edges")
        
        # Save statistics
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n✓ Statistics saved: {stats_path}")
        
        # Create optimized KB
        optimized_kb = create_optimized_kb(concept_index, optimized_kb_path, args.max_concepts)
        
        # Backup old KB
        old_kb_path = KB_DIR / "knowledge_base.json"
        if old_kb_path.exists():
            backup_path = KB_DIR / "knowledge_base_mini_backup.json"
            old_kb_path.rename(backup_path)
            print(f"\n✓ Backed up old KB: {backup_path}")
        
        # Use optimized KB as new default
        import shutil
        shutil.copy(optimized_kb_path, old_kb_path)
        print(f"✓ Installed new KB: {old_kb_path}")
    
    print("\n" + "="*70)
    print("UPGRADE COMPLETE!")
    print("="*70)
    print(f"New KB: {optimized_kb_path}")
    print(f"Statistics: {stats_path}")
    print("\nYour SID module will now use the full ConceptNet knowledge base!")
    print("="*70)


if __name__ == "__main__":
    main()
