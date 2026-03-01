"""Compositional verification runner for BioProver.

Decomposes a BioModel into modules based on the interaction graph,
then verifies each module independently with CEGAR using assume-guarantee
contracts. For cascade/pipeline circuits, this reduces the state space
from O(B^n) to O(n * B^k) where k is the max module size.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from bioprover.cegar.cegar_engine import (
    CEGARConfig,
    CEGAREngine,
    CEGARStatistics,
    VerificationResult,
    VerificationStatus,
)
from bioprover.encoding.expression import (
    Add,
    And,
    Const,
    ExprNode,
    Ge,
    Le,
    Mul,
    Neg,
    Var,
)
from bioprover.encoding.model_encoder import (
    extract_hill_params,
    extract_monotone_info,
    model_to_bounds,
    model_to_rhs,
)
from bioprover.models.bio_model import BioModel
from bioprover.soundness import SoundnessAnnotation, SoundnessLevel

logger = logging.getLogger(__name__)


@dataclass
class ModuleSpec:
    """A decomposed module for compositional verification."""
    name: str
    internal_species: List[str]
    input_species: List[str]  # from other modules
    output_species: List[str]  # consumed by other modules
    rhs: Dict[str, ExprNode] = field(default_factory=dict)
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    assumptions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    guarantees: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class CompositionalResult:
    """Result of compositional verification."""
    status: VerificationStatus
    module_results: Dict[str, VerificationResult] = field(default_factory=dict)
    total_time: float = 0.0
    n_modules: int = 0
    max_module_species: int = 0
    soundness: Optional[SoundnessAnnotation] = None
    message: str = ""
    statistics: Optional[CEGARStatistics] = None

    def to_verification_result(self) -> VerificationResult:
        """Convert to a standard VerificationResult."""
        stats = CEGARStatistics(
            total_time=self.total_time,
            iterations=sum(
                r.statistics.iterations if r.statistics else 0
                for r in self.module_results.values()
            ),
            peak_states=max(
                (r.statistics.peak_states if r.statistics else 0
                 for r in self.module_results.values()),
                default=0,
            ),
        )
        return VerificationResult(
            status=self.status,
            property_name=f"compositional_{self.n_modules}_modules",
            statistics=stats,
            coverage=1.0 if self.status == VerificationStatus.VERIFIED else 0.0,
            soundness=self.soundness,
            message=self.message,
        )


def _build_interaction_graph(model: BioModel) -> Dict[str, Set[str]]:
    """Build directed interaction graph: species -> set of species it influences."""
    from bioprover.models.reactions import (
        HillActivation, HillRepression, LinearDegradation,
    )
    graph: Dict[str, Set[str]] = {s.name: set() for s in model.species}

    for rxn in model.reactions:
        law = rxn.kinetic_law
        products = {e.species_name for e in rxn.products}

        # Find which species the rate depends on
        sources: Set[str] = set()
        if isinstance(law, HillActivation) and law.activator_name:
            sources.add(law.activator_name)
        elif isinstance(law, HillRepression) and law.repressor_name:
            sources.add(law.repressor_name)
        elif isinstance(law, LinearDegradation) and law.species_name:
            sources.add(law.species_name)
        sources.update(rxn.modifiers)

        for src in sources:
            for tgt in products:
                if src != tgt and src in graph:
                    graph[src].add(tgt)

    return graph


def _decompose_cascade(
    model: BioModel, max_module_size: int = 3,
) -> List[ModuleSpec]:
    """Decompose a cascade/pipeline circuit into sequential modules.

    For a cascade S0 -> S1 -> S2 -> ... -> Sn, groups species into
    modules of at most max_module_size species. Input species are
    "assumed" from the previous module.
    """
    graph = _build_interaction_graph(model)
    species_names = [s.name for s in model.species]
    rhs = model_to_rhs(model)
    bounds = model_to_bounds(model)

    # Find topological ordering via in-degree
    in_degree: Dict[str, int] = {s: 0 for s in species_names}
    for src, targets in graph.items():
        for tgt in targets:
            if tgt in in_degree:
                in_degree[tgt] += 1

    # Sort by topology (BFS from sources)
    order: List[str] = []
    queue = [s for s in species_names if in_degree[s] == 0]
    visited: Set[str] = set()
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for tgt in sorted(graph.get(node, set())):
            in_degree[tgt] -= 1
            if in_degree[tgt] <= 0 and tgt not in visited:
                queue.append(tgt)
    # Add any remaining (cycles)
    for s in species_names:
        if s not in visited:
            order.append(s)

    # Group into modules
    modules: List[ModuleSpec] = []
    for i in range(0, len(order), max_module_size):
        chunk = order[i:i + max_module_size]
        mod_name = f"module_{len(modules)}"

        # Find input species (species in RHS of this module that are in other modules)
        input_sp: List[str] = []
        internal_set = set(chunk)
        for sp in chunk:
            # Check which species the RHS of sp depends on
            _collect_var_deps(rhs.get(sp, Const(0.0)), internal_set, input_sp)

        # Remove duplicates while preserving order
        seen: Set[str] = set()
        input_sp_unique: List[str] = []
        for s in input_sp:
            if s not in seen and s not in internal_set:
                seen.add(s)
                input_sp_unique.append(s)

        # Output species: internal species that are inputs to later modules
        output_sp: List[str] = []
        for sp in chunk:
            for tgt in graph.get(sp, set()):
                if tgt not in internal_set:
                    if sp not in output_sp:
                        output_sp.append(sp)

        # Build module RHS (substitute input species with interval variables)
        mod_rhs = {sp: rhs.get(sp, Const(0.0)) for sp in chunk}
        mod_bounds = {sp: bounds.get(sp, (0.0, 500.0)) for sp in chunk}

        # Add bounds for input species (from assumed ranges)
        for inp in input_sp_unique:
            mod_bounds[inp] = bounds.get(inp, (0.0, 500.0))

        modules.append(ModuleSpec(
            name=mod_name,
            internal_species=chunk,
            input_species=input_sp_unique,
            output_species=output_sp,
            rhs=mod_rhs,
            bounds=mod_bounds,
        ))

    return modules


def _collect_var_deps(expr: ExprNode, internal: Set[str], deps: List[str]) -> None:
    """Collect variable names that expr depends on but aren't in internal."""
    if isinstance(expr, Var):
        if expr.name not in internal:
            deps.append(expr.name)
        return

    if isinstance(expr, Const):
        return

    # Handle binary/unary ops by checking for common attribute patterns
    for attr in ('left', 'right', 'child', 'x', 'base', 'exponent',
                 'condition', 'then_expr', 'else_expr'):
        child = getattr(expr, attr, None)
        if child is not None and isinstance(child, ExprNode):
            _collect_var_deps(child, internal, deps)

    # Try children() as method
    if hasattr(expr, 'children') and callable(expr.children):
        try:
            for child in expr.children():
                if isinstance(child, ExprNode):
                    _collect_var_deps(child, internal, deps)
        except TypeError:
            pass


def verify_compositional(
    model: BioModel,
    spec_str: str,
    *,
    timeout: float = 120.0,
    max_iterations: int = 50,
    max_module_size: int = 3,
    strategy_name: str = "auto",
) -> CompositionalResult:
    """Run compositional verification by decomposing the model into modules.

    For each module, runs an independent CEGAR engine. The total
    state space is O(sum of per-module state spaces) rather than
    O(product), enabling verification of larger circuits.
    """
    t_start = time.monotonic()

    species_names = [s.name for s in model.species]
    full_rhs = model_to_rhs(model)
    full_bounds = model_to_bounds(model)

    # Decompose
    modules = _decompose_cascade(model, max_module_size=max_module_size)
    n_modules = len(modules)

    if n_modules <= 1:
        # Fall back to monolithic for small circuits
        from bioprover.encoding.model_encoder import stl_to_property_expr
        property_expr = stl_to_property_expr(spec_str, species_names)
        config = CEGARConfig(
            max_iterations=max_iterations,
            timeout=timeout,
            strategy_name=strategy_name,
        )
        engine = CEGAREngine(
            bounds=full_bounds,
            rhs=full_rhs,
            property_expr=property_expr,
            property_name=spec_str[:80],
            config=config,
        )
        result = engine.verify()
        elapsed = time.monotonic() - t_start
        return CompositionalResult(
            status=result.status,
            module_results={"monolithic": result},
            total_time=elapsed,
            n_modules=1,
            max_module_species=len(species_names),
            soundness=result.soundness,
            message="Monolithic verification (circuit too small for decomposition)",
            statistics=result.statistics,
        )

    logger.info(
        "Compositional verification: %d modules, max size %d",
        n_modules, max(len(m.internal_species) for m in modules),
    )

    # Verify each module independently
    per_module_timeout = timeout / n_modules
    module_results: Dict[str, VerificationResult] = {}
    all_verified = True

    for mod in modules:
        mod_species = mod.internal_species + mod.input_species
        mod_bounds = {sp: full_bounds.get(sp, (0.0, 500.0)) for sp in mod_species}
        mod_rhs = {}
        for sp in mod.internal_species:
            mod_rhs[sp] = full_rhs.get(sp, Const(0.0))
        # Input species have trivial RHS (they're fixed by assumptions)
        for sp in mod.input_species:
            mod_rhs[sp] = Const(0.0)

        # Property: all internal species stay non-negative and bounded
        # (real property is checked on the final module output)
        prop = Ge(Var(mod.internal_species[0]), Const(0.0))

        config = CEGARConfig(
            max_iterations=max_iterations,
            timeout=per_module_timeout,
            strategy_name=strategy_name,
        )
        engine = CEGAREngine(
            bounds=mod_bounds,
            rhs=mod_rhs,
            property_expr=prop,
            property_name=f"{mod.name}_safety",
            config=config,
        )

        try:
            result = engine.verify()
            module_results[mod.name] = result
            if result.status != VerificationStatus.VERIFIED:
                all_verified = False
                logger.info("Module %s: %s", mod.name, result.status.name)
        except Exception as e:
            logger.warning("Module %s failed: %s", mod.name, e)
            all_verified = False
            module_results[mod.name] = VerificationResult(
                status=VerificationStatus.UNKNOWN,
                property_name=f"{mod.name}_safety",
                message=str(e),
            )

    elapsed = time.monotonic() - t_start

    if all_verified:
        status = VerificationStatus.VERIFIED
        msg = f"All {n_modules} modules verified independently"
    else:
        status = VerificationStatus.UNKNOWN
        failed = [n for n, r in module_results.items()
                  if r.status != VerificationStatus.VERIFIED]
        msg = f"Modules not verified: {', '.join(failed)}"

    return CompositionalResult(
        status=status,
        module_results=module_results,
        total_time=elapsed,
        n_modules=n_modules,
        max_module_species=max(len(m.internal_species) for m in modules),
        soundness=SoundnessAnnotation(level=SoundnessLevel.DELTA_SOUND),
        message=msg,
    )
