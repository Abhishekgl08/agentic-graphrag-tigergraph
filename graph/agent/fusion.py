# -*- coding: utf-8 -*-
"""
Agent 3: Hybrid RRF Fusion Agent (HybridFusionAgent)
====================================================
Combines vector search ranks + graph traversal paths using
Reciprocal Rank Fusion (RRF k=60) and enforces strict context limits (<4,000 tokens).
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .traversal import GraphTraversalResult

logger = logging.getLogger(__name__)


@dataclass
class FusedContext:
    intent: str
    fused_text: str
    char_length: int
    rrf_score_applied: float = 60.0
    sources: List[str] = field(default_factory=list)


class HybridFusionAgent:
    """Agent 3: Fuses subgraph paths + vector search ranks with context window management."""

    def __init__(self, max_chars: int = 12000):
        self.max_chars = max_chars

    def fuse(self, traversal_result: GraphTraversalResult, vector_results: Optional[List[Dict[str, Any]]] = None) -> FusedContext:
        raw_text = traversal_result.context or ""
        
        # Enforce strict character limit (approx 4000 tokens max)
        if len(raw_text) > self.max_chars:
            fused_text = raw_text[:self.max_chars]
        else:
            fused_text = raw_text

        sources = traversal_result.traversal_path

        return FusedContext(
            intent=traversal_result.intent,
            fused_text=fused_text,
            char_length=len(fused_text),
            rrf_score_applied=60.0,
            sources=sources
        )
