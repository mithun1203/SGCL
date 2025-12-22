"""
Symbolic Guardrail System for SG-CL
====================================

Training-time data augmentation for semantic consistency.
Hard-gated by SID conflict detection.

Key Principle:
    When SID detects a conflict, inject 2-4 symbolically grounded facts
    to stabilize the semantic space during gradient updates.

Author: Mithun Naik
Project: SGCL Capstone
"""

from .guardrail_generator import GuardrailGenerator, GuardrailFact
from .guardrail_controller import GuardrailController

__all__ = ['GuardrailGenerator', 'GuardrailFact', 'GuardrailController']
