"""
6-Phase Validated Coding Module for AI-Ethics-HR Systematic Review.

Phases:
1. Initial AI Coding (Claude 3.5 Sonnet with RAG)
2. Multi-Model Consensus (Claude + GPT-4o + Groq)
3. Human Verification Sampling (Stratified 20%)
4. Inter-Coder Reliability (Kappa, Alpha, ICC)
5. Discrepancy Resolution (Audit trail)
6. Quality Assurance (Final validation)
"""

from .phase1_initial import Phase1Coder, Phase1CodingResult
from .phase2_consensus import Phase2ConsensusCoder, Phase2ConsensusResult
from .phase3_sampling import StratifiedSampler, SamplingConfig
from .phase4_reliability import ReliabilityCalculator, ReliabilityMetrics
from .phase5_resolution import DiscrepancyResolver, DiscrepancyResolution
from .phase6_qa import QualityAssurance, QualityGate

__all__ = [
    'Phase1Coder', 'Phase1CodingResult',
    'Phase2ConsensusCoder', 'Phase2ConsensusResult',
    'StratifiedSampler', 'SamplingConfig',
    'ReliabilityCalculator', 'ReliabilityMetrics',
    'DiscrepancyResolver', 'DiscrepancyResolution',
    'QualityAssurance', 'QualityGate'
]
