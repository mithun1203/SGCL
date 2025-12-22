"""
Entity Extractor Module
=======================

Extracts entities from natural language text using multiple NLP backends.
Supports spaCy, Stanza, and rule-based extraction.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Types of entities that can be extracted."""
    NOUN = "NOUN"
    PROPER_NOUN = "PROPER_NOUN"
    VERB = "VERB"
    ADJECTIVE = "ADJECTIVE"
    CONCEPT = "CONCEPT"
    NAMED_ENTITY = "NAMED_ENTITY"
    PRONOUN = "PRONOUN"
    COMPOUND = "COMPOUND"


@dataclass
class Entity:
    """
    Represents an extracted entity.
    
    Attributes:
        text: The entity text
        lemma: Lemmatized form
        entity_type: Type of entity
        start_char: Start character position in original text
        end_char: End character position
        confidence: Extraction confidence
        metadata: Additional information
    """
    text: str
    lemma: str
    entity_type: EntityType
    start_char: int = 0
    end_char: int = 0
    confidence: float = 1.0
    pos_tag: str = ""
    dependency: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.lemma.lower(), self.entity_type))
    
    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.lemma.lower() == other.lemma.lower()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "lemma": self.lemma,
            "entity_type": self.entity_type.value,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "pos_tag": self.pos_tag,
            "dependency": self.dependency,
            "metadata": self.metadata
        }


class NLPBackend(str, Enum):
    """Available NLP backends for entity extraction."""
    SPACY = "spacy"
    STANZA = "stanza"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"


class EntityExtractor:
    """
    Multi-backend entity extractor for semantic analysis.
    
    Supports:
        - spaCy (recommended for speed)
        - Stanza (recommended for accuracy)
        - Rule-based (no dependencies, fallback)
        - Hybrid (combines multiple backends)
    
    Example:
        >>> extractor = EntityExtractor(backend="spacy")
        >>> entities = extractor.extract("Penguins cannot fly but they can swim")
        >>> for entity in entities:
        ...     print(f"{entity.text} ({entity.entity_type})")
    """
    
    # Common words to filter out
    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "cannot", "not",
        "and", "or", "but", "if", "then", "else", "when", "where", "why",
        "how", "what", "which", "who", "whom", "this", "that", "these",
        "those", "it", "its", "of", "to", "for", "with", "on", "at", "by",
        "from", "in", "out", "up", "down", "over", "under", "again", "further",
        "once", "here", "there", "all", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "only", "own", "same", "so", "than", "too",
        "very", "just", "also", "now", "about", "into", "through", "during",
        "before", "after", "above", "below", "between", "both", "any"
    }
    
    # Negation words
    NEGATIONS = {"not", "cannot", "can't", "won't", "don't", "doesn't", "didn't",
                 "isn't", "aren't", "wasn't", "weren't", "never", "no", "none"}
    
    # Modal verbs
    MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
    
    def __init__(
        self,
        backend: str = "hybrid",
        spacy_model: str = "en_core_web_sm",
        stanza_model: str = "en",
        use_gpu: bool = False
    ):
        """
        Initialize the entity extractor.
        
        Args:
            backend: NLP backend to use ("spacy", "stanza", "rule_based", "hybrid")
            spacy_model: spaCy model name
            stanza_model: Stanza model name
            use_gpu: Whether to use GPU acceleration
        """
        self.backend = NLPBackend(backend)
        self.spacy_model_name = spacy_model
        self.stanza_model_name = stanza_model
        self.use_gpu = use_gpu
        
        self._spacy_nlp = None
        self._stanza_nlp = None
        
        # Initialize backends
        self._init_backends()
    
    def _init_backends(self):
        """Initialize the selected NLP backend(s)."""
        if self.backend in [NLPBackend.SPACY, NLPBackend.HYBRID]:
            self._init_spacy()
        
        if self.backend in [NLPBackend.STANZA, NLPBackend.HYBRID]:
            self._init_stanza()
    
    def _init_spacy(self):
        """Initialize spaCy backend."""
        try:
            import spacy
            try:
                self._spacy_nlp = spacy.load(self.spacy_model_name)
                logger.info(f"Loaded spaCy model: {self.spacy_model_name}")
            except OSError:
                logger.warning(f"spaCy model '{self.spacy_model_name}' not found. "
                             "Attempting to download...")
                from spacy.cli import download
                download(self.spacy_model_name)
                self._spacy_nlp = spacy.load(self.spacy_model_name)
        except ImportError:
            logger.warning("spaCy not installed. Using rule-based extraction.")
            if self.backend == NLPBackend.SPACY:
                self.backend = NLPBackend.RULE_BASED
    
    def _init_stanza(self):
        """Initialize Stanza backend."""
        try:
            import stanza
            try:
                self._stanza_nlp = stanza.Pipeline(
                    lang=self.stanza_model_name,
                    processors='tokenize,pos,lemma,depparse,ner',
                    use_gpu=self.use_gpu,
                    verbose=False
                )
                logger.info(f"Loaded Stanza model: {self.stanza_model_name}")
            except Exception:
                logger.warning("Stanza model not found. Downloading...")
                stanza.download(self.stanza_model_name, verbose=False)
                self._stanza_nlp = stanza.Pipeline(
                    lang=self.stanza_model_name,
                    processors='tokenize,pos,lemma,depparse,ner',
                    use_gpu=self.use_gpu,
                    verbose=False
                )
        except ImportError:
            logger.warning("Stanza not installed. Using rule-based extraction.")
            if self.backend == NLPBackend.STANZA:
                self.backend = NLPBackend.RULE_BASED
    
    def extract(
        self,
        text: str,
        include_verbs: bool = True,
        include_adjectives: bool = True,
        min_length: int = 2,
        filter_stopwords: bool = True
    ) -> List[Entity]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            include_verbs: Whether to include verbs as entities
            include_adjectives: Whether to include adjectives
            min_length: Minimum entity length (characters)
            filter_stopwords: Whether to filter common stopwords
        
        Returns:
            List of extracted entities
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        # Route to appropriate backend
        if self.backend == NLPBackend.SPACY:
            entities = self._extract_spacy(text)
        elif self.backend == NLPBackend.STANZA:
            entities = self._extract_stanza(text)
        elif self.backend == NLPBackend.HYBRID:
            entities = self._extract_hybrid(text)
        else:
            entities = self._extract_rule_based(text)
        
        # Filter entities
        filtered = []
        seen_lemmas = set()
        
        for entity in entities:
            # Skip short entities
            if len(entity.lemma) < min_length:
                continue
            
            # Skip stopwords
            if filter_stopwords and entity.lemma.lower() in self.STOPWORDS:
                continue
            
            # Skip duplicates
            lemma_lower = entity.lemma.lower()
            if lemma_lower in seen_lemmas:
                continue
            seen_lemmas.add(lemma_lower)
            
            # Filter by type
            if entity.entity_type == EntityType.VERB and not include_verbs:
                continue
            if entity.entity_type == EntityType.ADJECTIVE and not include_adjectives:
                continue
            
            filtered.append(entity)
        
        return filtered
    
    def _extract_spacy(self, text: str) -> List[Entity]:
        """Extract entities using spaCy."""
        if not self._spacy_nlp:
            return self._extract_rule_based(text)
        
        doc = self._spacy_nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entities.append(Entity(
                text=ent.text,
                lemma=ent.text.lower(),
                entity_type=EntityType.NAMED_ENTITY,
                start_char=ent.start_char,
                end_char=ent.end_char,
                metadata={"ner_label": ent.label_}
            ))
        
        # Extract nouns and verbs
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                entity_type = EntityType.PROPER_NOUN if token.pos_ == "PROPN" else EntityType.NOUN
                entities.append(Entity(
                    text=token.text,
                    lemma=token.lemma_,
                    entity_type=entity_type,
                    start_char=token.idx,
                    end_char=token.idx + len(token.text),
                    pos_tag=token.pos_,
                    dependency=token.dep_
                ))
            elif token.pos_ == "VERB":
                entities.append(Entity(
                    text=token.text,
                    lemma=token.lemma_,
                    entity_type=EntityType.VERB,
                    start_char=token.idx,
                    end_char=token.idx + len(token.text),
                    pos_tag=token.pos_,
                    dependency=token.dep_
                ))
            elif token.pos_ == "ADJ":
                entities.append(Entity(
                    text=token.text,
                    lemma=token.lemma_,
                    entity_type=EntityType.ADJECTIVE,
                    start_char=token.idx,
                    end_char=token.idx + len(token.text),
                    pos_tag=token.pos_,
                    dependency=token.dep_
                ))
        
        # Extract noun chunks (compound nouns)
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:  # Multi-word chunk
                entities.append(Entity(
                    text=chunk.text,
                    lemma=chunk.root.lemma_,
                    entity_type=EntityType.COMPOUND,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata={"root": chunk.root.text}
                ))
        
        return entities
    
    def _extract_stanza(self, text: str) -> List[Entity]:
        """Extract entities using Stanza."""
        if not self._stanza_nlp:
            return self._extract_rule_based(text)
        
        doc = self._stanza_nlp(text)
        entities = []
        
        for sentence in doc.sentences:
            # Extract named entities
            for ent in sentence.ents:
                entities.append(Entity(
                    text=ent.text,
                    lemma=ent.text.lower(),
                    entity_type=EntityType.NAMED_ENTITY,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    metadata={"ner_label": ent.type}
                ))
            
            # Extract tokens
            for word in sentence.words:
                if word.upos in ["NOUN", "PROPN"]:
                    entity_type = EntityType.PROPER_NOUN if word.upos == "PROPN" else EntityType.NOUN
                    entities.append(Entity(
                        text=word.text,
                        lemma=word.lemma,
                        entity_type=entity_type,
                        start_char=word.start_char if hasattr(word, 'start_char') else 0,
                        end_char=word.end_char if hasattr(word, 'end_char') else 0,
                        pos_tag=word.upos,
                        dependency=word.deprel
                    ))
                elif word.upos == "VERB":
                    entities.append(Entity(
                        text=word.text,
                        lemma=word.lemma,
                        entity_type=EntityType.VERB,
                        pos_tag=word.upos,
                        dependency=word.deprel
                    ))
                elif word.upos == "ADJ":
                    entities.append(Entity(
                        text=word.text,
                        lemma=word.lemma,
                        entity_type=EntityType.ADJECTIVE,
                        pos_tag=word.upos,
                        dependency=word.deprel
                    ))
        
        return entities
    
    def _extract_hybrid(self, text: str) -> List[Entity]:
        """Extract entities using multiple backends and merge results."""
        all_entities = []
        
        # Try spaCy first
        if self._spacy_nlp:
            spacy_entities = self._extract_spacy(text)
            for ent in spacy_entities:
                ent.metadata["source"] = "spacy"
            all_entities.extend(spacy_entities)
        
        # Add Stanza results
        if self._stanza_nlp:
            stanza_entities = self._extract_stanza(text)
            for ent in stanza_entities:
                ent.metadata["source"] = "stanza"
            all_entities.extend(stanza_entities)
        
        # Fallback to rule-based if no NLP backends available
        if not all_entities:
            all_entities = self._extract_rule_based(text)
        
        # Merge and deduplicate
        return self._merge_entities(all_entities)
    
    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge entities from multiple sources, keeping highest confidence."""
        merged = {}
        for entity in entities:
            key = (entity.lemma.lower(), entity.entity_type)
            if key not in merged or entity.confidence > merged[key].confidence:
                merged[key] = entity
        return list(merged.values())
    
    def _extract_rule_based(self, text: str) -> List[Entity]:
        """
        Extract entities using rule-based patterns.
        No external dependencies required.
        """
        entities = []
        
        # Tokenize
        tokens = self._simple_tokenize(text)
        
        for i, token in enumerate(tokens):
            word = token["text"]
            word_lower = word.lower()
            
            # Skip punctuation and short words
            if len(word) < 2 or not word.isalpha():
                continue
            
            # Skip stopwords
            if word_lower in self.STOPWORDS:
                continue
            
            # Determine entity type based on patterns
            entity_type = EntityType.NOUN  # Default
            
            # Check if it's a verb (simple heuristic)
            if word_lower.endswith(('ing', 'ed', 'es', 's')) and len(word) > 4:
                # Check common verb patterns
                base = word_lower
                if word_lower.endswith('ing'):
                    base = word_lower[:-3]
                elif word_lower.endswith('ed'):
                    base = word_lower[:-2]
                elif word_lower.endswith('es'):
                    base = word_lower[:-2]
                elif word_lower.endswith('s'):
                    base = word_lower[:-1]
                
                # Common verb bases
                common_verbs = {'fly', 'swim', 'run', 'walk', 'eat', 'drink', 'sleep',
                               'think', 'know', 'believe', 'want', 'need', 'like', 'love',
                               'hate', 'make', 'take', 'give', 'get', 'see', 'hear',
                               'move', 'live', 'die', 'grow', 'become', 'bark', 'meow'}
                
                if base in common_verbs or word_lower in common_verbs:
                    entity_type = EntityType.VERB
            
            # Check if it's an adjective
            adjective_suffixes = ('ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ical')
            common_adjectives = {'big', 'small', 'hot', 'cold', 'fast', 'slow', 'old', 'new',
                                'good', 'bad', 'happy', 'sad', 'red', 'blue', 'green', 'black',
                                'white', 'tall', 'short', 'long', 'wide', 'deep', 'flightless'}
            
            if word_lower.endswith(adjective_suffixes) or word_lower in common_adjectives:
                entity_type = EntityType.ADJECTIVE
            
            # Check if proper noun (capitalized, not at start)
            if word[0].isupper() and i > 0:
                entity_type = EntityType.PROPER_NOUN
            
            # Get lemma (simple stemming)
            lemma = self._simple_lemmatize(word_lower)
            
            entities.append(Entity(
                text=word,
                lemma=lemma,
                entity_type=entity_type,
                start_char=token["start"],
                end_char=token["end"],
                confidence=0.7  # Lower confidence for rule-based
            ))
        
        return entities
    
    def _simple_tokenize(self, text: str) -> List[Dict[str, Any]]:
        """Simple whitespace and punctuation tokenizer."""
        tokens = []
        current_pos = 0
        
        # Split by whitespace and punctuation
        pattern = r"[\w']+|[.,!?;:\-\(\)]"
        for match in re.finditer(pattern, text):
            tokens.append({
                "text": match.group(),
                "start": match.start(),
                "end": match.end()
            })
        
        return tokens
    
    def _simple_lemmatize(self, word: str) -> str:
        """
        Simple rule-based lemmatization.
        
        This is a basic fallback - spaCy/Stanza lemmatization is preferred.
        """
        # Common irregular forms
        irregulars = {
            "are": "be", "is": "be", "was": "be", "were": "be", "been": "be",
            "has": "have", "had": "have",
            "does": "do", "did": "do",
            "goes": "go", "went": "go", "gone": "go",
            "says": "say", "said": "say",
            "makes": "make", "made": "make",
            "takes": "take", "took": "take", "taken": "take",
            "comes": "come", "came": "come",
            "sees": "see", "saw": "see", "seen": "see",
            "knows": "know", "knew": "know", "known": "know",
            "gives": "give", "gave": "give", "given": "give",
            "flies": "fly", "flew": "fly", "flown": "fly",
            "swims": "swim", "swam": "swim", "swum": "swim",
            "runs": "run", "ran": "run",
            "walks": "walk", "walked": "walk",
            "birds": "bird",
            "penguins": "penguin",
            "animals": "animal",
            "dogs": "dog",
            "cats": "cat",
            "fishes": "fish",
            "wings": "wing",
            "feathers": "feather",
        }
        
        if word in irregulars:
            return irregulars[word]
        
        # Remove common suffixes
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith("es") and len(word) > 3:
            return word[:-2]
        if word.endswith("s") and len(word) > 2 and not word.endswith("ss"):
            return word[:-1]
        if word.endswith("ing") and len(word) > 5:
            base = word[:-3]
            if base.endswith(base[-1]) and base[-1] not in "aeiou":
                return base[:-1]
            return base
        if word.endswith("ed") and len(word) > 4:
            base = word[:-2]
            if base.endswith(base[-1]) and base[-1] not in "aeiou":
                return base[:-1]
            return base
        
        return word
    
    def extract_subject_object_pairs(
        self,
        text: str
    ) -> List[Tuple[Entity, Optional[Entity], Entity]]:
        """
        Extract (subject, verb, object) triples from text.
        
        Returns:
            List of (subject, verb, object) tuples
        """
        if self._spacy_nlp:
            return self._extract_svo_spacy(text)
        elif self._stanza_nlp:
            return self._extract_svo_stanza(text)
        else:
            return self._extract_svo_rule_based(text)
    
    def _extract_svo_spacy(self, text: str) -> List[Tuple[Entity, Optional[Entity], Entity]]:
        """Extract subject-verb-object triples using spaCy."""
        doc = self._spacy_nlp(text)
        triples = []
        
        for token in doc:
            # Find verbs
            if token.pos_ == "VERB":
                subject = None
                obj = None
                verb = Entity(
                    text=token.text,
                    lemma=token.lemma_,
                    entity_type=EntityType.VERB,
                    pos_tag=token.pos_,
                    dependency=token.dep_
                )
                
                # Find subject (nsubj)
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = Entity(
                            text=child.text,
                            lemma=child.lemma_,
                            entity_type=EntityType.NOUN,
                            pos_tag=child.pos_,
                            dependency=child.dep_
                        )
                    elif child.dep_ in ["dobj", "pobj", "attr"]:
                        obj = Entity(
                            text=child.text,
                            lemma=child.lemma_,
                            entity_type=EntityType.NOUN,
                            pos_tag=child.pos_,
                            dependency=child.dep_
                        )
                
                if subject and obj:
                    triples.append((subject, verb, obj))
        
        return triples
    
    def _extract_svo_stanza(self, text: str) -> List[Tuple[Entity, Optional[Entity], Entity]]:
        """Extract subject-verb-object triples using Stanza."""
        doc = self._stanza_nlp(text)
        triples = []
        
        for sentence in doc.sentences:
            # Build dependency map
            word_map = {word.id: word for word in sentence.words}
            
            for word in sentence.words:
                if word.upos == "VERB":
                    subject = None
                    obj = None
                    verb = Entity(
                        text=word.text,
                        lemma=word.lemma,
                        entity_type=EntityType.VERB,
                        pos_tag=word.upos,
                        dependency=word.deprel
                    )
                    
                    # Find children
                    for child in sentence.words:
                        if child.head == word.id:
                            if child.deprel in ["nsubj", "nsubj:pass"]:
                                subject = Entity(
                                    text=child.text,
                                    lemma=child.lemma,
                                    entity_type=EntityType.NOUN,
                                    pos_tag=child.upos,
                                    dependency=child.deprel
                                )
                            elif child.deprel in ["obj", "obl"]:
                                obj = Entity(
                                    text=child.text,
                                    lemma=child.lemma,
                                    entity_type=EntityType.NOUN,
                                    pos_tag=child.upos,
                                    dependency=child.deprel
                                )
                    
                    if subject and obj:
                        triples.append((subject, verb, obj))
        
        return triples
    
    def _extract_svo_rule_based(self, text: str) -> List[Tuple[Entity, Optional[Entity], Entity]]:
        """
        Extract subject-verb-object triples using rules.
        Simple pattern: NOUN VERB NOUN
        """
        entities = self._extract_rule_based(text)
        triples = []
        
        # Find sequences of NOUN-VERB-NOUN
        i = 0
        while i < len(entities) - 2:
            if (entities[i].entity_type in [EntityType.NOUN, EntityType.PROPER_NOUN] and
                entities[i+1].entity_type == EntityType.VERB and
                entities[i+2].entity_type in [EntityType.NOUN, EntityType.PROPER_NOUN]):
                triples.append((entities[i], entities[i+1], entities[i+2]))
                i += 3
            else:
                i += 1
        
        return triples
    
    def detect_negation(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect if the text contains negation.
        
        Returns:
            Tuple of (has_negation, list_of_negation_words)
        """
        text_lower = text.lower()
        found_negations = []
        
        for neg in self.NEGATIONS:
            # Use word boundary matching
            pattern = r'\b' + re.escape(neg) + r'\b'
            if re.search(pattern, text_lower):
                found_negations.append(neg)
        
        return len(found_negations) > 0, found_negations
    
    def get_main_concepts(self, text: str, max_concepts: int = 5) -> List[str]:
        """
        Get the main concepts from text, suitable for ConceptNet lookup.
        
        Returns:
            List of concept strings (lemmatized nouns)
        """
        entities = self.extract(text, include_verbs=True, include_adjectives=False)
        
        # Prioritize nouns
        nouns = [e for e in entities if e.entity_type in [EntityType.NOUN, EntityType.PROPER_NOUN]]
        verbs = [e for e in entities if e.entity_type == EntityType.VERB]
        
        concepts = []
        for entity in nouns[:max_concepts]:
            concepts.append(entity.lemma.lower())
        
        # Add verbs if we have room
        remaining = max_concepts - len(concepts)
        for entity in verbs[:remaining]:
            concepts.append(entity.lemma.lower())
        
        return concepts
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the active NLP backend."""
        return {
            "backend": self.backend.value,
            "spacy_available": self._spacy_nlp is not None,
            "spacy_model": self.spacy_model_name if self._spacy_nlp else None,
            "stanza_available": self._stanza_nlp is not None,
            "stanza_model": self.stanza_model_name if self._stanza_nlp else None,
            "gpu_enabled": self.use_gpu
        }
