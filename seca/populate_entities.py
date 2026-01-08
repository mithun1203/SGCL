"""
Populate entities field for all samples in SeCA 10K dataset.

Uses simple pattern matching for common entity types since the dataset
already has semantic structure. This is faster than full NER.
"""

import json
import re
from pathlib import Path
from collections import Counter

def extract_simple_entities(text: str):
    """
    Extract entities using simple pattern matching.
    Focuses on:
    - Capitalized words (likely proper nouns)
    - Common nouns that appear to be subjects/objects
    - Numbers
    """
    entities = []
    
    # Find capitalized words (potential proper nouns)
    # But exclude common sentence-starting words
    common_starts = {'The', 'A', 'An', 'All', 'Some', 'Many', 'Most', 'Every', 'No'}
    
    words = text.split()
    for i, word in enumerate(words):
        # Remove punctuation for checking
        clean_word = word.strip('.,!?";:')
        
        # Capitalized word that's not at sentence start
        if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
            if i > 0 or clean_word not in common_starts:
                # Check if it's a multi-word entity
                entity_text = clean_word
                entities.append({
                    "text": entity_text,
                    "label": "ENTITY"
                })
    
    # Find numbers
    numbers = re.findall(r'\b\d+\b', text)
    for num in numbers:
        entities.append({
            "text": num,
            "label": "NUMBER"
        })
    
    # Remove duplicates while preserving order
    seen = set()
    unique_entities = []
    for ent in entities:
        key = (ent['text'], ent['label'])
        if key not in seen:
            seen.add(key)
            unique_entities.append(ent)
    
    return unique_entities


def populate_entities(input_path: str, output_path: str):
    """
    Populate entities field using fast pattern matching.
    """
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process all tasks
    total_samples = 0
    entities_added = 0
    
    print("\nProcessing samples...")
    for task_idx, task in enumerate(data['tasks']):
        samples = task.get('samples', [])
        total_samples += len(samples)
        
        for sample in samples:
            # Extract text
            text = sample.get('sentence', '')
            
            # Extract entities
            entities = extract_simple_entities(text)
            
            # Update sample
            sample['entities'] = entities
            
            if entities:
                entities_added += 1
        
        print(f"  Task {task_idx + 1}/{len(data['tasks'])}: {task['name']} - {len(samples)} samples")
    
    # Save enhanced dataset
    print(f"\nSaving enhanced dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    file_size = Path(output_path).stat().st_size / (1024*1024)
    
    print("\n" + "="*60)
    print("✅ ENTITY POPULATION COMPLETE")
    print("="*60)
    print(f"Total samples processed: {total_samples:,}")
    print(f"Samples with entities:   {entities_added:,} ({entities_added/total_samples*100:.1f}%)")
    print(f"Output saved to:         {output_path}")
    print(f"File size:               {file_size:.2f} MB")
    print("="*60)
    
    return output_path


if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "sid" / "seca_10k_dataset.json"
    output_file = base_dir / "sid" / "seca_10k_final.json"
    
    # Check if input exists
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        exit(1)
    
    # Run
    populate_entities(str(input_file), str(output_file))
