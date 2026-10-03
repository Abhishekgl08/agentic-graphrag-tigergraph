# -*- coding: utf-8 -*-
"""
Agentic Multi-Agent Orchestrator (AgenticOrchestrator)
======================================================
Coordinates Agent 1, Agent 2, Agent 3, and Agent 4 in an autonomous control loop with Langfuse Observability:
  User Question -> QueryRouterAgent -> GraphTraversalAgent -> HybridFusionAgent -> AnswerGenerationAgent -> Final Answer
"""

import os
import logging
import dotenv
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langfuse import observe
from .router import QueryRouterAgent, RouterDecision
from .traversal import GraphTraversalAgent, GraphTraversalResult
from .fusion import HybridFusionAgent, FusedContext
from .generation import AnswerGenerationAgent, FinalAnswer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
dotenv.load_dotenv(PROJECT_ROOT / ".env", override=True)


@dataclass
class OrchestratorResponse:
    question: str
    answer: str
    intent: str
    llm_engine: str
    latency_sec: float
    routing_decision: RouterDecision
    traversal_result: GraphTraversalResult
    fused_context: FusedContext
    final_answer: FinalAnswer


class AgenticOrchestrator:
    """Multi-Agent Orchestrator controlling the 4-agent GraphRAG pipeline loop with Langfuse tracing."""

    def __init__(self):
        self.router = QueryRouterAgent()
        self.traversal_agent = GraphTraversalAgent()
        self.fusion_agent = HybridFusionAgent()
        self.generation_agent = AnswerGenerationAgent()

    @observe(name="agentic-graphrag-pipeline")
    def process(self, question: str, gold_doc_ids: Optional[List[str]] = None, default_qtype: Optional[str] = None) -> OrchestratorResponse:
        # Step 1: Agent 1 - Intent Routing & Strategy
        decision = self.router.route(question, default_qtype=default_qtype)

        # Step 2: Agent 2 - Subgraph Traversal & Context Extraction
        traversal_res = self.traversal_agent.traverse(decision, gold_doc_ids=gold_doc_ids)

        # Step 3: Agent 3 - Hybrid Rank Fusion & Token Context Limit Guard
        fused_context = self.fusion_agent.fuse(traversal_res)

        # Step 4: Agent 4 - Multi-Engine Answer Generation & Verification
        final_ans = self.generation_agent.generate(question, fused_context)

        return OrchestratorResponse(
            question=question,
            answer=final_ans.answer,
            intent=decision.intent,
            llm_engine=final_ans.llm_engine_used,
            latency_sec=final_ans.latency_sec,
            routing_decision=decision,
            traversal_result=traversal_res,
            fused_context=fused_context,
            final_answer=final_ans
        )
