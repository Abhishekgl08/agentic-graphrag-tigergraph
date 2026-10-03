# -*- coding: utf-8 -*-
"""
Agent 1: Intent & Query Routing Agent (QueryRouterAgent)
=========================================================
Classifies query intent into:
  - TEMPORAL
  - SUPERLATIVE
  - AGGREGATION
  - MULTI_HOP
  - LOOKUP
And determines the optimal retrieval path.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RouterDecision:
    intent: str
    target_qtype: str
    question: str = ""
    extracted_entities: List[str] = field(default_factory=list)
    year: Optional[int] = None
    threshold: Optional[int] = None
    use_graph_traversal: bool = True
    reasoning: str = ""


class QueryRouterAgent:
    """Agent 1: Classifies question intent and determines the routing path."""

    def __init__(self):
        pass

    def route(self, question: str, default_qtype: Optional[str] = None) -> RouterDecision:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        q_lower = question.lower()
        
        # If explicit dataset qtype is provided, use it as baseline intent anchor
        if default_qtype:
            dt = default_qtype.lower()
            if dt in ("lookup", "temporal", "multi_hop", "superlative", "aggregation"):
                return RouterDecision(
                    intent=dt.upper(),
                    target_qtype=dt,
                    question=question,
                    use_graph_traversal=True,
                    reasoning=f"Using explicit intent classification: {dt.upper()}"
                )

        # 1. Single Event Factual Lookup Intent
        if "how many nations competed" in q_lower or "how many athletes competed" in q_lower or "how many competitors" in q_lower:
            return RouterDecision(
                intent="LOOKUP",
                target_qtype="lookup",
                question=question,
                use_graph_traversal=True,
                reasoning="Detected single-event factual lookup query pattern."
            )

        # 2. Multi-Event Aggregation Intent
        if any(w in q_lower for w in ["how many events", "how many biathlon events", "how many athletics events", "how many shooting events", "how many swimming events", "how many canoeing events"]):
            m_thresh = re.search(r'more than\s+(\d+)|exceeded\s+(\d+)|>\s*(\d+)', q_lower)
            threshold = None
            if m_thresh:
                threshold = int(m_thresh.group(1) or m_thresh.group(2) or m_thresh.group(3))
            
            m_year = re.search(r'\b(19\d\d|20\d\d)\b', question)
            year = int(m_year.group(1)) if m_year else None

            return RouterDecision(
                intent="AGGREGATION",
                target_qtype="aggregation",
                question=question,
                year=year,
                threshold=threshold,
                use_graph_traversal=True,
                reasoning="Detected multi-event counting query pattern."
            )

        # 3. Superlative intent
        if any(w in q_lower for w in ["highest number", "most competitors", "largest field", "most participants", "highest", "most", "largest"]):
            m_year = re.search(r'\b(19\d\d|20\d\d)\b', question)
            year = int(m_year.group(1)) if m_year else None

            return RouterDecision(
                intent="SUPERLATIVE",
                target_qtype="superlative",
                question=question,
                year=year,
                use_graph_traversal=True,
                reasoning="Detected global maximum / superlative ranking query pattern."
            )

        # 4. Temporal intent
        if any(w in q_lower for w in ["immediately before", "held before", "immediately after", "preceding", "following year"]):
            m_year = re.search(r'\b(19\d\d|20\d\d)\b', question)
            year = int(m_year.group(1)) if m_year else None

            return RouterDecision(
                intent="TEMPORAL",
                target_qtype="temporal",
                question=question,
                year=year,
                use_graph_traversal=True,
                reasoning="Detected temporal sequence / relative year query pattern."
            )

        # 5. Multi-Hop Traversal intent
        if any(w in q_lower for w in ["who won", "held at", "where was", "venue", "team", "gold medal in the event"]):
            return RouterDecision(
                intent="MULTI_HOP",
                target_qtype="multi_hop",
                question=question,
                use_graph_traversal=True,
                reasoning="Detected multi-entity relationship traversal query pattern."
            )

        # 6. Default Fallback
        return RouterDecision(
            intent="LOOKUP",
            target_qtype="lookup",
            question=question,
            use_graph_traversal=True,
            reasoning="Default factual lookup routing."
        )
