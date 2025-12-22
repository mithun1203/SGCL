"""
ConceptNet Numberbatch Integration
===================================

Uses ConceptNet Numberbatch word embeddings for semantic similarity.
The mini.h5 file is ~150MB and provides semantic vectors for ~500K+ concepts.

This provides an alternative to the ConceptNet API when:
1. API is down/unreliable
2. Offline usage is required
3. Fast similarity lookups are needed

Download mini.h5 from:
https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/19.08/mini.h5

Or English-only text file (~100MB compressed):
https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import h5py for HDF5 support
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logger.warning("h5py not available. Install with: pip install h5py")

# Try to import gensim for text format support
try:
    from gensim.models import KeyedVectors
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False


@dataclass
class NumberbatchConfig:
    """Configuration for Numberbatch embeddings."""
    embeddings_path: Optional[str] = None  # Path to mini.h5 or .txt.gz file
    auto_download: bool = False  # Whether to auto-download if missing
    language: str = "en"
    similarity_threshold: float = 0.5  # Minimum similarity to consider related
    cache_lookups: bool = True


class NumberbatchKB:
    """
    Knowledge base using ConceptNet Numberbatch word embeddings.
    
    Uses semantic similarity to infer relationships:
    - High similarity between "penguin" and "bird" -> IsA relation likely
    - Low similarity between "penguin" and "fly" + high with "swim" -> capability inference
    
    Example:
        >>> kb = NumberbatchKB("path/to/mini.h5")
        >>> similarity = kb.get_similarity("cat", "dog")
        >>> print(f"Similarity: {similarity:.3f}")
        
        >>> related = kb.get_related_concepts("penguin", top_k=10)
        >>> for concept, score in related:
        ...     print(f"  {concept}: {score:.3f}")
    """
    
    def __init__(self, config: Optional[NumberbatchConfig] = None):
        self.config = config or NumberbatchConfig()
        self.embeddings: Optional[Dict[str, np.ndarray]] = None
        self.vectors: Optional[np.ndarray] = None
        self.index_to_key: Optional[List[str]] = None
        self.key_to_index: Optional[Dict[str, int]] = None
        self._loaded = False
        self._cache: Dict[str, Any] = {}
        
        if self.config.embeddings_path:
            self.load(self.config.embeddings_path)
    
    def load(self, path: str) -> bool:
        """
        Load embeddings from file.
        
        Supports:
        - .h5 files (HDF5 format from ConceptNet)
        - .txt or .txt.gz files (word2vec text format)
        """
        path = Path(path)
        
        if not path.exists():
            logger.error(f"Embeddings file not found: {path}")
            return False
        
        try:
            if path.suffix == '.h5':
                return self._load_h5(path)
            elif path.suffix in ['.txt', '.gz']:
                return self._load_text(path)
            else:
                logger.error(f"Unsupported file format: {path.suffix}")
                return False
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return False
    
    def _load_h5(self, path: Path) -> bool:
        """Load from HDF5 format."""
        if not HAS_H5PY:
            logger.error("h5py required for .h5 files. Install: pip install h5py")
            return False
        
        logger.info(f"Loading Numberbatch from {path}...")
        
        with h5py.File(path, 'r') as f:
            # The mini.h5 format has 'mat' for vectors and 'label' for terms
            if 'mat' in f and 'label' in f:
                self.vectors = f['mat'][:]
                labels = f['label'][:]
                # Labels might be bytes
                self.index_to_key = [
                    l.decode('utf-8') if isinstance(l, bytes) else l 
                    for l in labels
                ]
            else:
                # Try alternative format
                keys = list(f.keys())
                logger.info(f"H5 keys: {keys}")
                # Assume first key is the data
                data = f[keys[0]]
                if hasattr(data, 'keys'):
                    self.index_to_key = list(data.keys())
                    self.vectors = np.array([data[k][:] for k in self.index_to_key])
        
        self.key_to_index = {k: i for i, k in enumerate(self.index_to_key)}
        self._loaded = True
        
        logger.info(f"Loaded {len(self.index_to_key)} concept vectors")
        return True
    
    def _load_text(self, path: Path) -> bool:
        """Load from word2vec text format."""
        if HAS_GENSIM:
            logger.info(f"Loading Numberbatch from {path} using gensim...")
            kv = KeyedVectors.load_word2vec_format(str(path), binary=False)
            self.vectors = kv.vectors
            self.index_to_key = kv.index_to_key
            self.key_to_index = kv.key_to_index
            self._loaded = True
            logger.info(f"Loaded {len(self.index_to_key)} concept vectors")
            return True
        else:
            # Manual loading
            logger.info(f"Loading Numberbatch from {path}...")
            
            import gzip
            opener = gzip.open if str(path).endswith('.gz') else open
            
            vectors = []
            keys = []
            
            with opener(path, 'rt', encoding='utf-8') as f:
                # First line is dimensions
                header = f.readline().strip().split()
                num_vectors, dim = int(header[0]), int(header[1])
                
                for line in f:
                    parts = line.strip().split(' ')
                    key = parts[0]
                    vector = np.array([float(x) for x in parts[1:]])
                    keys.append(key)
                    vectors.append(vector)
            
            self.vectors = np.array(vectors)
            self.index_to_key = keys
            self.key_to_index = {k: i for i, k in enumerate(keys)}
            self._loaded = True
            
            logger.info(f"Loaded {len(keys)} concept vectors")
            return True
    
    def _normalize_concept(self, concept: str) -> str:
        """Normalize concept to ConceptNet URI format."""
        # ConceptNet uses /c/en/concept format
        concept = concept.lower().strip().replace(' ', '_')
        
        lang = self.config.language
        if not concept.startswith('/c/'):
            concept = f"/c/{lang}/{concept}"
        
        return concept
    
    def get_vector(self, concept: str) -> Optional[np.ndarray]:
        """Get embedding vector for a concept."""
        if not self._loaded:
            return None
        
        # Try different formats
        normalized = self._normalize_concept(concept)
        
        for key in [normalized, concept.lower(), f"/c/en/{concept.lower()}"]:
            if key in self.key_to_index:
                return self.vectors[self.key_to_index[key]]
        
        return None
    
    def get_similarity(self, concept1: str, concept2: str) -> float:
        """
        Get cosine similarity between two concepts.
        
        Returns:
            Similarity score in [-1, 1], or 0.0 if either concept not found
        """
        vec1 = self.get_vector(concept1)
        vec2 = self.get_vector(concept2)
        
        if vec1 is None or vec2 is None:
            return 0.0
        
        # Cosine similarity
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))
    
    def get_related_concepts(
        self, 
        concept: str, 
        top_k: int = 10,
        threshold: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Find concepts most similar to the given concept.
        
        Args:
            concept: The concept to find related concepts for
            top_k: Number of results to return
            threshold: Minimum similarity threshold
        
        Returns:
            List of (concept, similarity) tuples, sorted by similarity
        """
        if not self._loaded:
            return []
        
        vec = self.get_vector(concept)
        if vec is None:
            return []
        
        threshold = threshold or self.config.similarity_threshold
        
        # Compute similarities to all concepts
        # Normalize for cosine similarity
        vec_norm = vec / np.linalg.norm(vec)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_vectors = self.vectors / norms
        
        similarities = np.dot(normalized_vectors, vec_norm)
        
        # Get top-k indices (excluding the concept itself)
        top_indices = np.argsort(similarities)[::-1][:top_k + 5]
        
        results = []
        normalized_input = self._normalize_concept(concept)
        
        for idx in top_indices:
            if len(results) >= top_k:
                break
            
            key = self.index_to_key[idx]
            sim = float(similarities[idx])
            
            # Skip self and low similarity
            if key == normalized_input or sim < threshold:
                continue
            
            # Clean up the key for display
            display_key = key.split('/')[-1] if '/' in key else key
            results.append((display_key, sim))
        
        return results
    
    def has_concept(self, concept: str) -> bool:
        """Check if a concept exists in the embeddings."""
        return self.get_vector(concept) is not None
    
    def infer_capability(
        self, 
        subject: str, 
        action: str,
        positive_examples: Optional[List[str]] = None,
        negative_examples: Optional[List[str]] = None
    ) -> Tuple[bool, float]:
        """
        Infer if a subject is capable of an action based on semantic similarity.
        
        Uses comparison with known positive and negative examples.
        
        Args:
            subject: The subject (e.g., "penguin")
            action: The action (e.g., "fly")
            positive_examples: Known subjects that CAN do the action
            negative_examples: Known subjects that CANNOT do the action
        
        Returns:
            Tuple of (is_capable, confidence)
        """
        # Default examples for common actions
        default_examples = {
            "fly": {
                "positive": ["bird", "airplane", "bat", "eagle", "sparrow"],
                "negative": ["fish", "dog", "cat", "human", "elephant", "snake"]
            },
            "swim": {
                "positive": ["fish", "dolphin", "whale", "penguin", "duck"],
                "negative": ["bird", "cat", "lion", "elephant"]
            },
            "walk": {
                "positive": ["human", "dog", "cat", "elephant", "bird"],
                "negative": ["fish", "whale", "snake", "worm"]
            },
            "bark": {
                "positive": ["dog", "seal"],
                "negative": ["cat", "fish", "bird", "human"]
            }
        }
        
        # Get examples
        action_lower = action.lower()
        if action_lower in default_examples:
            positive_examples = positive_examples or default_examples[action_lower]["positive"]
            negative_examples = negative_examples or default_examples[action_lower]["negative"]
        else:
            positive_examples = positive_examples or []
            negative_examples = negative_examples or []
        
        if not positive_examples and not negative_examples:
            # No examples, use direct similarity
            sim = self.get_similarity(subject, action)
            return sim > 0.3, abs(sim)
        
        # Calculate average similarity to positive and negative examples
        pos_similarities = [
            self.get_similarity(subject, ex) 
            for ex in positive_examples
        ]
        neg_similarities = [
            self.get_similarity(subject, ex)
            for ex in negative_examples
        ]
        
        avg_pos = np.mean(pos_similarities) if pos_similarities else 0
        avg_neg = np.mean(neg_similarities) if neg_similarities else 0
        
        # Subject is more similar to positive examples -> likely capable
        is_capable = avg_pos > avg_neg
        confidence = abs(avg_pos - avg_neg)
        
        return is_capable, float(confidence)
    
    @property
    def vocab_size(self) -> int:
        """Number of concepts in the embeddings."""
        return len(self.index_to_key) if self.index_to_key else 0
    
    @property
    def is_loaded(self) -> bool:
        """Whether embeddings are loaded."""
        return self._loaded


def download_numberbatch_mini(destination: str = "./data/mini.h5") -> bool:
    """
    Download the ConceptNet Numberbatch mini.h5 file.
    
    The file is ~50MB compressed, ~150MB uncompressed.
    """
    import urllib.request
    import zipfile
    import io
    
    url = "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/19.08/mini.h5"
    
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists():
        logger.info(f"File already exists: {dest_path}")
        return True
    
    logger.info(f"Downloading Numberbatch mini from {url}...")
    
    try:
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"Downloaded to {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


# Convenience function
def create_numberbatch_kb(path: Optional[str] = None) -> NumberbatchKB:
    """
    Create a NumberbatchKB instance.
    
    If no path provided, looks for mini.h5 in common locations.
    """
    if path:
        config = NumberbatchConfig(embeddings_path=path)
        return NumberbatchKB(config)
    
    # Try common locations
    search_paths = [
        "./data/mini.h5",
        "./mini.h5",
        "../data/mini.h5",
        Path(__file__).parent / "data" / "mini.h5",
        Path.home() / ".conceptnet" / "mini.h5"
    ]
    
    for p in search_paths:
        if Path(p).exists():
            config = NumberbatchConfig(embeddings_path=str(p))
            return NumberbatchKB(config)
    
    logger.warning("No Numberbatch embeddings found. KB will be empty.")
    return NumberbatchKB()


if __name__ == "__main__":
    # Demo
    print("ConceptNet Numberbatch KB Demo")
    print("=" * 50)
    
    # Check if file exists
    if not Path("./data/mini.h5").exists():
        print("\nNumberbatch mini.h5 not found.")
        print("Download from: https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/19.08/mini.h5")
        print("Save to: ./data/mini.h5")
    else:
        kb = create_numberbatch_kb("./data/mini.h5")
        
        if kb.is_loaded:
            print(f"Loaded {kb.vocab_size} concepts")
            
            # Test similarity
            pairs = [
                ("cat", "dog"),
                ("penguin", "bird"),
                ("penguin", "fly"),
                ("fish", "swim"),
            ]
            
            print("\nSimilarities:")
            for c1, c2 in pairs:
                sim = kb.get_similarity(c1, c2)
                print(f"  {c1} <-> {c2}: {sim:.3f}")
            
            # Test capability inference
            print("\nCapability Inference:")
            tests = [
                ("penguin", "fly"),
                ("penguin", "swim"),
                ("dog", "bark"),
                ("cat", "fly"),
            ]
            
            for subject, action in tests:
                capable, conf = kb.infer_capability(subject, action)
                result = "CAN" if capable else "CANNOT"
                print(f"  {subject} {result} {action} (confidence: {conf:.3f})")
