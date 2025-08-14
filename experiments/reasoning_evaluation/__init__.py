"""
Reasoning Evaluation Framework

A modular framework for evaluating LLM reasoning capabilities across different APIs.
"""

__version__ = "1.0.0"
__author__ = "LLM Reasoning Research Team"

from .framework.llm_reasoning_framework import LLMReasoningFramework
from .api_clients.client_factory import create_api_client
from .analysis.reasoning_analyzer import ReasoningAnalyzer
from .analysis.self_evaluation import SelfEvaluation

__all__ = [
    "LLMReasoningFramework",
    "create_api_client", 
    "ReasoningAnalyzer",
    "SelfEvaluation"
]
