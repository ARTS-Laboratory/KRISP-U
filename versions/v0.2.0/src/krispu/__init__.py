"""KRISP-U: GPR-based active field reconstruction tools."""

from krispu.benchmarks import BenchmarkResult, MethodTrace, run_benchmark
from krispu.datasets import ToyDataset, get_dataset, list_datasets
from krispu.designs import corner_plus_interior_design
from krispu.models import (
    KernelCandidateScore,
    KernelPriorConfig,
    KernelPriorResult,
    VariogramSummary,
)
from krispu.optimizer import AcquisitionResult, KrispUOptimizer, OptimizationResult
from krispu.recommendation import (
    Recommendation,
    RecommendationSet,
    infer_continuous_space,
    recommend_next,
)
from krispu.space import ContinuousSpace, DiscreteCandidateSpace, HybridCandidateSpace

KRISPU = KrispUOptimizer

__all__ = [
    "AcquisitionResult",
    "BenchmarkResult",
    "ContinuousSpace",
    "DiscreteCandidateSpace",
    "HybridCandidateSpace",
    "KRISPU",
    "KernelCandidateScore",
    "KernelPriorConfig",
    "KernelPriorResult",
    "KrispUOptimizer",
    "MethodTrace",
    "OptimizationResult",
    "Recommendation",
    "RecommendationSet",
    "ToyDataset",
    "VariogramSummary",
    "corner_plus_interior_design",
    "get_dataset",
    "infer_continuous_space",
    "list_datasets",
    "recommend_next",
    "run_benchmark",
]
