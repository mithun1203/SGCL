"""
ConceptNet API Client
=====================

Robust client for querying the ConceptNet knowledge graph.
Includes caching, rate limiting, and fallback mechanisms.
"""

import time
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from functools import lru_cache
import urllib.parse

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from .models import Triple, ConceptNetEdge, SemanticRelation

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ConceptNetConfig:
    """Configuration for ConceptNet client."""
    api_base_url: str = "https://api.conceptnet.io"
    cache_dir: Optional[str] = None
    cache_enabled: bool = True
    cache_ttl_seconds: int = 86400 * 7  # 1 week
    rate_limit_requests_per_second: float = 5.0
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    default_language: str = "en"
    min_edge_weight: float = 1.0  # Minimum weight to consider
    max_edges_per_query: int = 100
    knowledge_base_path: Optional[str] = None  # Path to external KB JSON file
    offline_only: bool = False  # If True, never call the API - use only local KB


class ConceptNetCache:
    """
    File-based cache for ConceptNet queries.
    
    Reduces API calls and improves response time for repeated queries.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, ttl_seconds: int = 86400 * 7):
        self.ttl_seconds = ttl_seconds
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "sgcl" / "conceptnet"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Any] = {}
        
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{key}.json"
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached result if valid."""
        key = self._get_cache_key(query)
        
        # Check memory cache first
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                logger.debug(f"Memory cache hit for: {query[:50]}...")
                return entry["data"]
        
        # Check file cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                if time.time() - entry["timestamp"] < self.ttl_seconds:
                    # Update memory cache
                    self._memory_cache[key] = entry
                    logger.debug(f"File cache hit for: {query[:50]}...")
                    return entry["data"]
                else:
                    # Cache expired, remove file
                    cache_path.unlink()
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Invalid cache file: {cache_path}")
                cache_path.unlink()
        
        return None
    
    def set(self, query: str, data: Dict[str, Any]) -> None:
        """Store result in cache."""
        key = self._get_cache_key(query)
        entry = {
            "timestamp": time.time(),
            "query": query,
            "data": data
        }
        
        # Update memory cache
        self._memory_cache[key] = entry
        
        # Write to file cache
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(entry, f)
            logger.debug(f"Cached: {query[:50]}...")
        except IOError as e:
            logger.warning(f"Failed to write cache: {e}")
    
    def clear(self) -> int:
        """Clear all cached entries. Returns number of entries cleared."""
        count = 0
        self._memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        file_count = len(list(self.cache_dir.glob("*.json")))
        memory_count = len(self._memory_cache)
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))
        return {
            "file_entries": file_count,
            "memory_entries": memory_count,
            "total_size_bytes": total_size,
            "cache_directory": str(self.cache_dir)
        }


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, requests_per_second: float):
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()


class ConceptNetClient:
    """
    Client for querying ConceptNet knowledge graph.
    
    Features:
        - Automatic caching of query results
        - Rate limiting to respect API limits
        - Retry logic with exponential backoff
        - Multiple query types (concept lookup, edge search, etc.)
    
    Example:
        >>> client = ConceptNetClient()
        >>> edges = client.get_edges_for_concept("penguin")
        >>> for edge in edges:
        ...     print(edge.to_triple().to_natural_language())
    """
    
    def __init__(self, config: Optional[ConceptNetConfig] = None):
        self.config = config or ConceptNetConfig()
        
        if not HAS_REQUESTS:
            logger.warning("requests library not available. Using offline mode.")
        
        # Initialize cache
        if self.config.cache_enabled:
            self.cache = ConceptNetCache(
                cache_dir=self.config.cache_dir,
                ttl_seconds=self.config.cache_ttl_seconds
            )
        else:
            self.cache = None
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(self.config.rate_limit_requests_per_second)
        
        # Session for connection pooling
        if HAS_REQUESTS:
            self.session = requests.Session()
            self.session.headers.update({
                "User-Agent": "SGCL-SID/1.0 (Semantic Inconsistency Detector)",
                "Accept": "application/json"
            })
        else:
            self.session = None
        
        # Offline knowledge base (fallback)
        self._offline_kb: Dict[str, List[Dict]] = {}
        self._load_offline_kb()
    
    def _load_offline_kb(self):
        """Load offline knowledge base for common concepts."""
        # Pre-loaded common knowledge for testing and offline use
        self._offline_kb = {
            "bird": [
                {"start": "/c/en/bird", "rel": "/r/CapableOf", "end": "/c/en/fly", "weight": 5.0},
                {"start": "/c/en/bird", "rel": "/r/HasA", "end": "/c/en/wing", "weight": 4.0},
                {"start": "/c/en/bird", "rel": "/r/HasA", "end": "/c/en/feather", "weight": 4.0},
                {"start": "/c/en/bird", "rel": "/r/IsA", "end": "/c/en/animal", "weight": 5.0},
            ],
            "penguin": [
                {"start": "/c/en/penguin", "rel": "/r/IsA", "end": "/c/en/bird", "weight": 5.0},
                {"start": "/c/en/penguin", "rel": "/r/NotCapableOf", "end": "/c/en/fly", "weight": 4.0},
                {"start": "/c/en/penguin", "rel": "/r/CapableOf", "end": "/c/en/swim", "weight": 4.5},
                {"start": "/c/en/penguin", "rel": "/r/AtLocation", "end": "/c/en/antarctica", "weight": 3.0},
                {"start": "/c/en/penguin", "rel": "/r/HasProperty", "end": "/c/en/flightless", "weight": 4.0},
            ],
            "ostrich": [
                {"start": "/c/en/ostrich", "rel": "/r/IsA", "end": "/c/en/bird", "weight": 5.0},
                {"start": "/c/en/ostrich", "rel": "/r/NotCapableOf", "end": "/c/en/fly", "weight": 4.0},
                {"start": "/c/en/ostrich", "rel": "/r/CapableOf", "end": "/c/en/run", "weight": 4.5},
                {"start": "/c/en/ostrich", "rel": "/r/HasProperty", "end": "/c/en/flightless", "weight": 4.0},
            ],
            "dog": [
                {"start": "/c/en/dog", "rel": "/r/IsA", "end": "/c/en/animal", "weight": 5.0},
                {"start": "/c/en/dog", "rel": "/r/IsA", "end": "/c/en/pet", "weight": 4.0},
                {"start": "/c/en/dog", "rel": "/r/CapableOf", "end": "/c/en/bark", "weight": 5.0},
                {"start": "/c/en/dog", "rel": "/r/HasA", "end": "/c/en/tail", "weight": 4.0},
                {"start": "/c/en/dog", "rel": "/r/NotCapableOf", "end": "/c/en/fly", "weight": 5.0},
            ],
            "cat": [
                {"start": "/c/en/cat", "rel": "/r/IsA", "end": "/c/en/animal", "weight": 5.0},
                {"start": "/c/en/cat", "rel": "/r/IsA", "end": "/c/en/pet", "weight": 4.0},
                {"start": "/c/en/cat", "rel": "/r/CapableOf", "end": "/c/en/climb", "weight": 4.0},
                {"start": "/c/en/cat", "rel": "/r/CapableOf", "end": "/c/en/meow", "weight": 5.0},
            ],
            "fish": [
                {"start": "/c/en/fish", "rel": "/r/IsA", "end": "/c/en/animal", "weight": 5.0},
                {"start": "/c/en/fish", "rel": "/r/CapableOf", "end": "/c/en/swim", "weight": 5.0},
                {"start": "/c/en/fish", "rel": "/r/AtLocation", "end": "/c/en/water", "weight": 5.0},
                {"start": "/c/en/fish", "rel": "/r/NotCapableOf", "end": "/c/en/walk", "weight": 4.0},
                {"start": "/c/en/fish", "rel": "/r/NotCapableOf", "end": "/c/en/fly", "weight": 4.0},
            ],
            "whale": [
                {"start": "/c/en/whale", "rel": "/r/IsA", "end": "/c/en/mammal", "weight": 5.0},
                {"start": "/c/en/whale", "rel": "/r/CapableOf", "end": "/c/en/swim", "weight": 5.0},
                {"start": "/c/en/whale", "rel": "/r/AtLocation", "end": "/c/en/ocean", "weight": 5.0},
                {"start": "/c/en/whale", "rel": "/r/NotCapableOf", "end": "/c/en/walk", "weight": 4.0},
            ],
            "bat": [
                {"start": "/c/en/bat", "rel": "/r/IsA", "end": "/c/en/mammal", "weight": 5.0},
                {"start": "/c/en/bat", "rel": "/r/CapableOf", "end": "/c/en/fly", "weight": 5.0},
                {"start": "/c/en/bat", "rel": "/r/HasA", "end": "/c/en/wing", "weight": 4.0},
            ],
            "fly": [
                {"start": "/c/en/fly", "rel": "/r/IsA", "end": "/c/en/action", "weight": 3.0},
                {"start": "/c/en/fly", "rel": "/r/HasPrerequisite", "end": "/c/en/wing", "weight": 3.0},
            ],
            "swim": [
                {"start": "/c/en/swim", "rel": "/r/IsA", "end": "/c/en/action", "weight": 3.0},
                {"start": "/c/en/swim", "rel": "/r/AtLocation", "end": "/c/en/water", "weight": 4.0},
            ],
            "water": [
                {"start": "/c/en/water", "rel": "/r/IsA", "end": "/c/en/liquid", "weight": 5.0},
                {"start": "/c/en/water", "rel": "/r/UsedFor", "end": "/c/en/drinking", "weight": 4.0},
            ],
            "fire": [
                {"start": "/c/en/fire", "rel": "/r/IsA", "end": "/c/en/element", "weight": 3.0},
                {"start": "/c/en/fire", "rel": "/r/HasProperty", "end": "/c/en/hot", "weight": 5.0},
                {"start": "/c/en/fire", "rel": "/r/Causes", "end": "/c/en/burn", "weight": 4.0},
            ],
            "ice": [
                {"start": "/c/en/ice", "rel": "/r/IsA", "end": "/c/en/solid", "weight": 4.0},
                {"start": "/c/en/ice", "rel": "/r/HasProperty", "end": "/c/en/cold", "weight": 5.0},
                {"start": "/c/en/ice", "rel": "/r/MadeOf", "end": "/c/en/water", "weight": 5.0},
            ],
            "human": [
                {"start": "/c/en/human", "rel": "/r/IsA", "end": "/c/en/mammal", "weight": 5.0},
                {"start": "/c/en/human", "rel": "/r/CapableOf", "end": "/c/en/think", "weight": 5.0},
                {"start": "/c/en/human", "rel": "/r/CapableOf", "end": "/c/en/walk", "weight": 5.0},
                {"start": "/c/en/human", "rel": "/r/NotCapableOf", "end": "/c/en/fly", "weight": 5.0},
            ],
            "car": [
                {"start": "/c/en/car", "rel": "/r/IsA", "end": "/c/en/vehicle", "weight": 5.0},
                {"start": "/c/en/car", "rel": "/r/UsedFor", "end": "/c/en/transportation", "weight": 5.0},
                {"start": "/c/en/car", "rel": "/r/HasA", "end": "/c/en/wheel", "weight": 5.0},
                {"start": "/c/en/car", "rel": "/r/CapableOf", "end": "/c/en/drive", "weight": 4.0},
            ],
            "airplane": [
                {"start": "/c/en/airplane", "rel": "/r/IsA", "end": "/c/en/vehicle", "weight": 5.0},
                {"start": "/c/en/airplane", "rel": "/r/CapableOf", "end": "/c/en/fly", "weight": 5.0},
                {"start": "/c/en/airplane", "rel": "/r/HasA", "end": "/c/en/wing", "weight": 5.0},
            ],
            "sun": [
                {"start": "/c/en/sun", "rel": "/r/IsA", "end": "/c/en/star", "weight": 5.0},
                {"start": "/c/en/sun", "rel": "/r/HasProperty", "end": "/c/en/hot", "weight": 5.0},
                {"start": "/c/en/sun", "rel": "/r/Causes", "end": "/c/en/light", "weight": 5.0},
            ],
            "moon": [
                {"start": "/c/en/moon", "rel": "/r/IsA", "end": "/c/en/satellite", "weight": 5.0},
                {"start": "/c/en/moon", "rel": "/r/AtLocation", "end": "/c/en/sky", "weight": 4.0},
            ],
        }
        
        # Try to load from external JSON file for more comprehensive KB
        self._load_external_kb()
    
    def _load_external_kb(self):
        """Load comprehensive knowledge base from external JSON file."""
        kb_paths = []
        
        # Check config path first
        if self.config.knowledge_base_path:
            kb_paths.append(Path(self.config.knowledge_base_path))
        
        # Check default location (same directory as this module)
        module_dir = Path(__file__).parent
        kb_paths.append(module_dir / "knowledge_base.json")
        
        for kb_path in kb_paths:
            if kb_path.exists():
                try:
                    with open(kb_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if "concepts" in data:
                        # Merge with existing KB (external takes precedence)
                        for concept, edges in data["concepts"].items():
                            self._offline_kb[concept] = edges
                        
                        logger.info(f"Loaded {len(data['concepts'])} concepts from {kb_path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load KB from {kb_path}: {e}")
    
    def add_knowledge(self, concept: str, edges: List[Dict]) -> None:
        """
        Dynamically add knowledge to the offline KB.
        
        Args:
            concept: The concept name (will be normalized)
            edges: List of edge dictionaries with start, rel, end, weight
        """
        concept_normalized = concept.lower().strip()
        if concept_normalized not in self._offline_kb:
            self._offline_kb[concept_normalized] = []
        self._offline_kb[concept_normalized].extend(edges)
    
    def get_known_concepts(self) -> List[str]:
        """Return list of all concepts in the offline KB."""
        return list(self._offline_kb.keys())
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make HTTP request to ConceptNet API with retry logic."""
        if not HAS_REQUESTS or not self.session:
            return None
        
        url = f"{self.config.api_base_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                self.rate_limiter.wait()
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.config.timeout_seconds
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limited
                    wait_time = self.config.retry_delay_seconds * (2 ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif e.response.status_code == 404:
                    return {"edges": []}
                else:
                    logger.error(f"HTTP error: {e}")
                    return None
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_seconds * (2 ** attempt))
        
        return None
    
    def get_concept_uri(self, concept: str, language: Optional[str] = None) -> str:
        """Convert concept to ConceptNet URI format."""
        lang = language or self.config.default_language
        # Normalize: lowercase, replace spaces with underscores
        normalized = concept.lower().strip().replace(" ", "_")
        return f"/c/{lang}/{normalized}"
    
    def _normalize_to_singular(self, word: str) -> str:
        """
        Simple singular/plural normalization for offline KB lookup.
        Handles common English plural patterns.
        """
        word = word.lower().strip()
        
        # Common irregular plurals
        irregulars = {
            "mice": "mouse", "men": "man", "women": "woman",
            "children": "child", "feet": "foot", "teeth": "tooth",
            "geese": "goose", "people": "person", "oxen": "ox",
            "cacti": "cactus", "fungi": "fungus", "nuclei": "nucleus",
            "syllabi": "syllabus", "alumni": "alumnus", "larvae": "larva",
            "vertebrae": "vertebra", "wolves": "wolf", "knives": "knife",
            "lives": "life", "wives": "wife", "leaves": "leaf",
            "selves": "self", "halves": "half", "calves": "calf",
            "loaves": "loaf", "thieves": "thief", "shelves": "shelf"
        }
        
        if word in irregulars:
            return irregulars[word]
        
        # Regular plural patterns (order matters - check longer suffixes first)
        if word.endswith("ies") and len(word) > 3:
            return word[:-3] + "y"  # flies -> fly
        elif word.endswith("ves"):
            return word[:-3] + "f"  # wolves -> wolf (handled above, but fallback)
        elif word.endswith("es"):
            if word.endswith("shes") or word.endswith("ches") or word.endswith("xes") or word.endswith("zes"):
                return word[:-2]  # boxes -> box, bushes -> bush
            elif word.endswith("sses"):
                return word[:-2]  # classes -> class
            elif word.endswith("oes"):
                return word[:-2]  # heroes -> hero
            else:
                # Could be just 'es' or 's' - try without 'es' first
                return word[:-1]  # tries -> try... but often just -s
        elif word.endswith("s") and len(word) > 1 and not word.endswith("ss"):
            return word[:-1]  # cats -> cat
        
        return word
    
    def get_edges_for_concept(
        self,
        concept: str,
        relations: Optional[List[str]] = None,
        as_subject: bool = True,
        as_object: bool = True,
        limit: Optional[int] = None
    ) -> List[ConceptNetEdge]:
        """
        Get all edges connected to a concept.
        
        Args:
            concept: The concept to look up (e.g., "penguin")
            relations: Optional list of relations to filter by
            as_subject: Include edges where concept is the subject
            as_object: Include edges where concept is the object
            limit: Maximum number of edges to return
        
        Returns:
            List of ConceptNetEdge objects
        """
        concept_normalized = concept.lower().strip()
        concept_singular = self._normalize_to_singular(concept_normalized)
        cache_key = f"edges:{concept_singular}:{relations}:{as_subject}:{as_object}"
        
        # Check cache
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                edges = [ConceptNetEdge.from_api_response(e) for e in cached]
                return self._filter_edges(edges, relations, limit)
        
        # Try offline KB first - try both singular and original forms
        all_edges = []
        for lookup_key in [concept_singular, concept_normalized]:
            if lookup_key in self._offline_kb:
                for edge_data in self._offline_kb[lookup_key]:
                    all_edges.append(ConceptNetEdge.from_api_response(edge_data))
                break  # Found edges, don't duplicate
        
        # Try API if online and not in offline-only mode
        if HAS_REQUESTS and self.session and not self.config.offline_only:
            concept_uri = self.get_concept_uri(concept)
            
            # Query for edges where concept is subject
            if as_subject:
                response = self._make_request(
                    f"/query",
                    params={
                        "start": concept_uri,
                        "limit": self.config.max_edges_per_query
                    }
                )
                if response and "edges" in response:
                    for edge_data in response["edges"]:
                        edge = ConceptNetEdge.from_api_response(edge_data)
                        if edge.weight >= self.config.min_edge_weight:
                            all_edges.append(edge)
            
            # Query for edges where concept is object
            if as_object:
                response = self._make_request(
                    f"/query",
                    params={
                        "end": concept_uri,
                        "limit": self.config.max_edges_per_query
                    }
                )
                if response and "edges" in response:
                    for edge_data in response["edges"]:
                        edge = ConceptNetEdge.from_api_response(edge_data)
                        if edge.weight >= self.config.min_edge_weight:
                            all_edges.append(edge)
            
            # Cache the results
            if self.cache and all_edges:
                self.cache.set(
                    cache_key, 
                    [{"start": e.start, "rel": e.relation, "end": e.end, 
                      "weight": e.weight, "surfaceText": e.surface_text}
                     for e in all_edges]
                )
        
        return self._filter_edges(all_edges, relations, limit)
    
    def _filter_edges(
        self,
        edges: List[ConceptNetEdge],
        relations: Optional[List[str]],
        limit: Optional[int]
    ) -> List[ConceptNetEdge]:
        """Filter edges by relation type and limit."""
        if relations:
            relation_set = {f"/r/{r}" if not r.startswith("/r/") else r for r in relations}
            edges = [e for e in edges if e.relation in relation_set]
        
        # Remove duplicates
        seen = set()
        unique_edges = []
        for edge in edges:
            key = (edge.start, edge.relation, edge.end)
            if key not in seen:
                seen.add(key)
                unique_edges.append(edge)
        
        # Sort by weight (most confident first)
        unique_edges.sort(key=lambda e: e.weight, reverse=True)
        
        if limit:
            unique_edges = unique_edges[:limit]
        
        return unique_edges
    
    def query_relation(
        self,
        subject: str,
        relation: str,
        object_concept: Optional[str] = None
    ) -> List[ConceptNetEdge]:
        """
        Query for specific relation between concepts.
        
        Args:
            subject: Subject concept
            relation: Relation type (e.g., "CapableOf")
            object_concept: Optional object concept to check
        
        Returns:
            List of matching edges
        """
        cache_key = f"relation:{subject}:{relation}:{object_concept}"
        
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return [ConceptNetEdge.from_api_response(e) for e in cached]
        
        # Check offline KB
        results = []
        subject_normalized = subject.lower().strip()
        relation_uri = f"/r/{relation}" if not relation.startswith("/r/") else relation
        
        if subject_normalized in self._offline_kb:
            for edge_data in self._offline_kb[subject_normalized]:
                if edge_data.get("rel") == relation_uri:
                    if object_concept is None:
                        results.append(ConceptNetEdge.from_api_response(edge_data))
                    else:
                        obj_uri = f"/c/en/{object_concept.lower().strip()}"
                        if edge_data.get("end") == obj_uri:
                            results.append(ConceptNetEdge.from_api_response(edge_data))
        
        # Try API if not in offline-only mode
        if HAS_REQUESTS and self.session and not self.config.offline_only:
            params = {
                "start": self.get_concept_uri(subject),
                "rel": relation_uri,
                "limit": self.config.max_edges_per_query
            }
            if object_concept:
                params["end"] = self.get_concept_uri(object_concept)
            
            response = self._make_request("/query", params=params)
            if response and "edges" in response:
                for edge_data in response["edges"]:
                    edge = ConceptNetEdge.from_api_response(edge_data)
                    if edge.weight >= self.config.min_edge_weight:
                        results.append(edge)
        
        # Cache results
        if self.cache:
            self.cache.set(
                cache_key,
                [{"start": e.start, "rel": e.relation, "end": e.end,
                  "weight": e.weight, "surfaceText": e.surface_text}
                 for e in results]
            )
        
        return results
    
    def check_relation_exists(
        self,
        subject: str,
        relation: str,
        object_concept: str
    ) -> Tuple[bool, Optional[ConceptNetEdge]]:
        """
        Check if a specific relation exists between two concepts.
        
        Returns:
            Tuple of (exists: bool, edge: Optional[ConceptNetEdge])
        """
        edges = self.query_relation(subject, relation, object_concept)
        if edges:
            return True, edges[0]
        return False, None
    
    def get_related_concepts(
        self,
        concept: str,
        relation: Optional[str] = None,
        limit: int = 20
    ) -> List[Tuple[str, str, float]]:
        """
        Get concepts related to the given concept.
        
        Returns:
            List of (related_concept, relation_type, weight) tuples
        """
        edges = self.get_edges_for_concept(
            concept,
            relations=[relation] if relation else None,
            limit=limit
        )
        
        results = []
        for edge in edges:
            triple = edge.to_triple()
            if triple.subject.lower() == concept.lower():
                results.append((triple.object, triple.relation, edge.weight))
            else:
                results.append((triple.subject, triple.relation, edge.weight))
        
        return results
    
    def get_superclasses(self, concept: str) -> List[str]:
        """Get all superclasses (IsA parents) of a concept."""
        edges = self.query_relation(concept, "IsA")
        return [edge.to_triple().object for edge in edges]
    
    def get_subclasses(self, concept: str) -> List[str]:
        """Get all subclasses (IsA children) of a concept."""
        edges = self.get_edges_for_concept(concept, relations=["IsA"], as_subject=False)
        return [edge.to_triple().subject for edge in edges]
    
    def get_capabilities(self, concept: str) -> List[Tuple[str, bool]]:
        """
        Get capabilities of a concept.
        
        Returns:
            List of (capability, is_positive) tuples
        """
        capabilities = []
        
        # Get CapableOf relations
        capable_edges = self.query_relation(concept, "CapableOf")
        for edge in capable_edges:
            capabilities.append((edge.to_triple().object, True))
        
        # Get NotCapableOf relations
        not_capable_edges = self.query_relation(concept, "NotCapableOf")
        for edge in not_capable_edges:
            capabilities.append((edge.to_triple().object, False))
        
        return capabilities
    
    def get_properties(self, concept: str) -> List[Tuple[str, bool]]:
        """
        Get properties of a concept.
        
        Returns:
            List of (property, is_positive) tuples
        """
        properties = []
        
        # Get HasProperty relations
        prop_edges = self.query_relation(concept, "HasProperty")
        for edge in prop_edges:
            properties.append((edge.to_triple().object, True))
        
        # Get NotHasProperty relations
        not_prop_edges = self.query_relation(concept, "NotHasProperty")
        for edge in not_prop_edges:
            properties.append((edge.to_triple().object, False))
        
        return properties
    
    def get_knowledge_for_concepts(
        self,
        concepts: List[str],
        include_related: bool = False
    ) -> Dict[str, List[Triple]]:
        """
        Get all knowledge for a list of concepts.
        
        Args:
            concepts: List of concept names
            include_related: Whether to include related concepts
        
        Returns:
            Dictionary mapping concept to list of triples
        """
        knowledge = {}
        processed = set()
        to_process = list(concepts)
        
        while to_process:
            concept = to_process.pop(0)
            if concept.lower() in processed:
                continue
            processed.add(concept.lower())
            
            edges = self.get_edges_for_concept(concept)
            triples = [edge.to_triple() for edge in edges]
            knowledge[concept.lower()] = triples
            
            if include_related:
                for triple in triples:
                    if triple.subject.lower() not in processed:
                        to_process.append(triple.subject)
                    if triple.object.lower() not in processed:
                        to_process.append(triple.object)
        
        return knowledge
    
    def clear_cache(self) -> int:
        """Clear the cache. Returns number of entries cleared."""
        if self.cache:
            return self.cache.clear()
        return 0
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return None
    
    def add_to_offline_kb(self, concept: str, edges: List[Dict]) -> None:
        """Add knowledge to the offline knowledge base."""
        concept_normalized = concept.lower().strip()
        if concept_normalized not in self._offline_kb:
            self._offline_kb[concept_normalized] = []
        self._offline_kb[concept_normalized].extend(edges)
    
    def export_offline_kb(self, filepath: str) -> None:
        """Export the offline knowledge base to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._offline_kb, f, indent=2)
    
    def import_offline_kb(self, filepath: str) -> None:
        """Import knowledge base from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self._offline_kb.update(data)
