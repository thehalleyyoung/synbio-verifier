"""Compositional verification module for BioProver.

Provides assume-guarantee contracts, automatic decomposition,
circular AG checking, contract synthesis, and proof composition
for modular verification of biological circuits.
"""

from __future__ import annotations

from bioprover.compositional.contracts import (
    Contract,
    ContractConjunction,
    ContractComposition,
    ContractRefinement,
    ContractSatisfaction,
    InterfaceVariable,
    SignalDirection,
    WellFormednessError,
)
from bioprover.compositional.decomposition import (
    DecompositionResult,
    DecompositionStrategy,
    Module,
    ModuleDecomposer,
    QualityMetrics,
)
from bioprover.compositional.circular_ag import (
    AGFailureDiagnostics,
    AGWellFormedness,
    CircularAGChecker,
    CircularAGResult,
    ConvergenceStatus,
    FixedPointState,
)
from bioprover.compositional.contract_synthesis import (
    ContractSynthesizer,
    SynthesisStrategy,
    ContractTemplate,
    ContractLibrary,
)
from bioprover.compositional.proof_composition import (
    ComposableProof,
    CompositionRule,
    ProofTree,
    ProofValidator,
    ProofCertificate,
)
from bioprover.compositional.ag_soundness import (
    AGSoundnessCertificate,
    CouplingAnalysis,
    ModuleODE,
    ProofStatus,
    SoundnessProver,
    Theorem1_AGComposition,
    Theorem2_RobustnessComposition,
    Theorem3_CircularAGConvergence,
)

__all__ = [
    # contracts
    "Contract",
    "ContractConjunction",
    "ContractComposition",
    "ContractRefinement",
    "ContractSatisfaction",
    "InterfaceVariable",
    "SignalDirection",
    "WellFormednessError",
    # decomposition
    "DecompositionResult",
    "DecompositionStrategy",
    "Module",
    "ModuleDecomposer",
    "QualityMetrics",
    # circular AG
    "AGFailureDiagnostics",
    "AGWellFormedness",
    "CircularAGChecker",
    "CircularAGResult",
    "ConvergenceStatus",
    "FixedPointState",
    # contract synthesis
    "ContractSynthesizer",
    "SynthesisStrategy",
    "ContractTemplate",
    "ContractLibrary",
    # proof composition
    "ComposableProof",
    "CompositionRule",
    "ProofTree",
    "ProofValidator",
    "ProofCertificate",
    # AG soundness
    "AGSoundnessCertificate",
    "CouplingAnalysis",
    "ModuleODE",
    "ProofStatus",
    "SoundnessProver",
    "Theorem1_AGComposition",
    "Theorem2_RobustnessComposition",
    "Theorem3_CircularAGConvergence",
]
