"""Convert a BioModel to CEGAR-engine inputs (RHS expressions, bounds, property).

This module bridges the gap between BioModel's reaction-based representation
and the CEGAR engine's expression-tree based RHS format.  It converts
Hill kinetics, mass-action, degradation, and constitutive production
reactions into ExprNode trees suitable for interval-based model checking.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from bioprover.encoding.expression import (
    Add,
    Const,
    Div,
    ExprNode,
    Ge,
    Le,
    Mul,
    Neg,
    Pow,
    Var,
)
from bioprover.models.bio_model import BioModel
from bioprover.models.reactions import (
    ConstitutiveProduction,
    DimerFormation,
    HillActivation,
    HillRepression,
    LinearDegradation,
    MassAction,
    MichaelisMenten,
    Reaction,
)

logger = logging.getLogger(__name__)


def _hill_act_expr(x: ExprNode, vmax: float, K: float, n: float) -> ExprNode:
    """Build Vmax * x^n / (K^n + x^n)."""
    if n == 1.0:
        # Michaelis-Menten form: Vmax * x / (K + x)
        return Mul(Const(vmax), Div(x, Add(Const(K), x)))
    n_int = int(n) if float(n).is_integer() else None
    if n_int is not None and n_int >= 1:
        x_n = x
        for _ in range(n_int - 1):
            x_n = Mul(x_n, x)
        K_n = Const(K ** n_int)
        return Mul(Const(vmax), Div(x_n, Add(K_n, x_n)))
    # Non-integer Hill coefficient: use Pow
    x_n = Pow(x, Const(n))
    K_n = Const(K ** n)
    return Mul(Const(vmax), Div(x_n, Add(K_n, x_n)))


def _hill_rep_expr(x: ExprNode, vmax: float, K: float, n: float) -> ExprNode:
    """Build Vmax * K^n / (K^n + x^n)."""
    if n == 1.0:
        return Mul(Const(vmax), Div(Const(K), Add(Const(K), x)))
    n_int = int(n) if float(n).is_integer() else None
    if n_int is not None and n_int >= 1:
        x_n = x
        for _ in range(n_int - 1):
            x_n = Mul(x_n, x)
        K_n = Const(K ** n_int)
        return Mul(Const(vmax), Div(K_n, Add(K_n, x_n)))
    x_n = Pow(x, Const(n))
    K_n = Const(K ** n)
    return Mul(Const(vmax), Div(K_n, Add(K_n, x_n)))


def _reaction_rate_expr(
    rxn: Reaction,
    species_vars: Dict[str, ExprNode],
) -> ExprNode:
    """Convert a single reaction's kinetic law to an ExprNode."""
    law = rxn.kinetic_law

    if isinstance(law, HillActivation):
        activator = law.activator_name
        if not activator:
            activator = rxn.modifiers[0] if rxn.modifiers else (
                rxn.reactants[0].species_name if rxn.reactants else ""
            )
        if activator and activator in species_vars:
            return _hill_act_expr(species_vars[activator], law.Vmax, law.K, law.n)
        return Const(law.Vmax * 0.5)  # fallback: half-max

    if isinstance(law, HillRepression):
        repressor = law.repressor_name
        if not repressor:
            repressor = rxn.modifiers[0] if rxn.modifiers else (
                rxn.reactants[0].species_name if rxn.reactants else ""
            )
        if repressor and repressor in species_vars:
            return _hill_rep_expr(species_vars[repressor], law.Vmax, law.K, law.n)
        return Const(law.Vmax * 0.5)

    if isinstance(law, LinearDegradation):
        sp = law.species_name
        if not sp and rxn.reactants:
            sp = rxn.reactants[0].species_name
        if sp and sp in species_vars:
            return Mul(Const(law._rate), species_vars[sp])
        return Const(0.0)

    if isinstance(law, ConstitutiveProduction):
        return Const(law._rate)

    if isinstance(law, MassAction):
        # k_forward * prod(reactants) - k_reverse * prod(products)
        fwd: ExprNode = Const(law.k_forward)
        for name in law.reactant_names:
            coeff = law._reactant_coeffs.get(name, 1)
            v = species_vars.get(name, Const(0.0))
            for _ in range(coeff):
                fwd = Mul(fwd, v)
        if law.k_reverse > 0:
            rev: ExprNode = Const(law.k_reverse)
            for name in law.product_names:
                coeff = law._product_coeffs.get(name, 1)
                v = species_vars.get(name, Const(0.0))
                for _ in range(coeff):
                    rev = Mul(rev, v)
            return Add(fwd, Neg(rev))
        return fwd

    if isinstance(law, MichaelisMenten):
        sp = law.substrate_name
        if not sp and rxn.reactants:
            sp = rxn.reactants[0].species_name
        if sp and sp in species_vars:
            s = species_vars[sp]
            return Mul(Const(law.Vmax), Div(s, Add(Const(law.Km), s)))
        return Const(0.0)

    # Unknown kinetic law: return zero
    logger.warning("Unknown kinetic law type %s for reaction %s", type(law).__name__, rxn.name)
    return Const(0.0)


def model_to_rhs(model: BioModel) -> Dict[str, ExprNode]:
    """Convert a BioModel to a dict of RHS expressions for each species.

    For each species s, computes:
        dx_s/dt = sum of (stoichiometry * rate) for each reaction

    Returns a dict {species_name: ExprNode} representing the ODE RHS.
    """
    species_names = [s.name for s in model.species]
    species_vars = {name: Var(name) for name in species_names}
    rhs: Dict[str, ExprNode] = {name: Const(0.0) for name in species_names}

    for rxn in model.reactions:
        rate = _reaction_rate_expr(rxn, species_vars)
        net = rxn.net_stoichiometry

        for sp_name, coeff in net.items():
            if sp_name not in rhs:
                continue
            if coeff == 0:
                continue
            contribution = Mul(Const(float(coeff)), rate) if coeff != 1 else rate
            if coeff == -1:
                contribution = Neg(rate)

            current = rhs[sp_name]
            # Check if current is Const(0.0) to avoid trivial additions
            if isinstance(current, Const) and current.value == 0.0:
                rhs[sp_name] = contribution
            else:
                rhs[sp_name] = Add(current, contribution)

    return rhs


def model_to_bounds(
    model: BioModel,
    default_upper: float = 500.0,
) -> Dict[str, Tuple[float, float]]:
    """Extract species bounds from a BioModel.

    Returns {species_name: (lower, upper)} with non-negative lower bound.
    """
    bounds: Dict[str, Tuple[float, float]] = {}
    for sp in model.species:
        lo = 0.0
        hi = default_upper
        # Use initial concentration to set a reasonable upper bound
        if sp.initial_concentration > 0:
            hi = max(default_upper, sp.initial_concentration * 5.0)
        bounds[sp.name] = (lo, hi)
    return bounds


def stl_to_property_expr(
    spec_str: str,
    species_names: List[str],
) -> ExprNode:
    """Parse an STL spec string and extract a property expression.

    For simple specs like "G[0,100](X > 0.5)", extracts the atomic
    predicate as an ExprNode. For complex specs, returns a conjunction
    of atomic predicates.
    """
    from bioprover.temporal.bio_stl_parser import BioSTLParser
    from bioprover.temporal.stl_ast import (
        Predicate as STLPredicate,
        ComparisonOp,
    )

    parser = BioSTLParser()
    formula = parser.parse(spec_str)

    # Extract atomic predicates from the formula tree
    predicates = _collect_predicates(formula)

    if not predicates:
        # Fallback: the first species should be non-negative
        return Ge(Var(species_names[0]), Const(0.0))

    # Build conjunction of all atomic predicates
    from bioprover.encoding.expression import And
    exprs: List[ExprNode] = []
    for pred in predicates:
        if hasattr(pred, 'variable_name') and hasattr(pred, 'threshold'):
            var_expr = Var(pred.variable_name)
            if pred.op == ComparisonOp.GE:
                exprs.append(Ge(var_expr, Const(pred.threshold)))
            elif pred.op == ComparisonOp.LE:
                exprs.append(Le(var_expr, Const(pred.threshold)))
            elif pred.op == ComparisonOp.GT:
                exprs.append(Ge(var_expr, Const(pred.threshold)))
            elif pred.op == ComparisonOp.LT:
                exprs.append(Le(var_expr, Const(pred.threshold)))

    if not exprs:
        return Ge(Var(species_names[0]), Const(0.0))
    result = exprs[0]
    for e in exprs[1:]:
        result = And(result, e)
    return result


def _collect_predicates(formula: Any) -> List[Any]:
    """Recursively collect atomic predicates from an STL formula."""
    from bioprover.temporal.stl_ast import (
        Predicate as STLPredicate,
    )
    from bioprover.temporal import (
        Always,
        Eventually,
        Until,
        STLAnd,
        STLOr,
        STLNot,
    )

    results: List[Any] = []
    if isinstance(formula, STLPredicate):
        results.append(formula)
    elif isinstance(formula, (Always, Eventually)):
        results.extend(_collect_predicates(formula.child))
    elif isinstance(formula, Until):
        results.extend(_collect_predicates(formula.left))
        results.extend(_collect_predicates(formula.right))
    elif isinstance(formula, (STLAnd, STLOr)):
        results.extend(_collect_predicates(formula.left))
        results.extend(_collect_predicates(formula.right))
    elif isinstance(formula, STLNot):
        results.extend(_collect_predicates(formula.child))
    return results


def extract_hill_params(model: BioModel) -> List[Dict[str, Any]]:
    """Extract Hill function parameters for monotonicity analysis."""
    params: List[Dict[str, Any]] = []
    for rxn in model.reactions:
        law = rxn.kinetic_law
        if isinstance(law, HillActivation):
            params.append({
                "type": "activation",
                "species": law.activator_name,
                "Vmax": law.Vmax,
                "K": law.K,
                "n": law.n,
                "reaction": rxn.name,
            })
        elif isinstance(law, HillRepression):
            params.append({
                "type": "repression",
                "species": law.repressor_name,
                "Vmax": law.Vmax,
                "K": law.K,
                "n": law.n,
                "reaction": rxn.name,
            })
    return params


def extract_monotone_info(model: BioModel) -> Dict[str, Dict[str, int]]:
    """Analyze monotonicity of the species interaction graph.

    Returns {target: {source: +1 or -1}} indicating activation (+1) or
    repression (-1) interactions.
    """
    interactions: Dict[str, Dict[str, int]] = {}
    for rxn in model.reactions:
        law = rxn.kinetic_law
        products = [e.species_name for e in rxn.products]

        if isinstance(law, HillActivation):
            src = law.activator_name
            for tgt in products:
                interactions.setdefault(tgt, {})[src] = 1
        elif isinstance(law, HillRepression):
            src = law.repressor_name
            for tgt in products:
                interactions.setdefault(tgt, {})[src] = -1
        elif isinstance(law, LinearDegradation):
            sp = law.species_name
            if sp:
                interactions.setdefault(sp, {})[sp] = -1

    return interactions


def is_monotone_system(model: BioModel) -> bool:
    """Check if the interaction graph is monotone (all cycles have even
    number of negative edges)."""
    info = extract_monotone_info(model)
    species = [s.name for s in model.species]

    # Build adjacency with signs
    import itertools
    # Check 2-coloring: if graph is monotone, it's 2-colorable
    # with activations as same-color and repressions as different-color
    color: Dict[str, int] = {}

    def dfs(node: str, c: int) -> bool:
        if node in color:
            return color[node] == c
        color[node] = c
        for target, sign_map in info.items():
            if node in sign_map:
                sign = sign_map[node]
                next_color = c if sign == 1 else (1 - c)
                if target in color:
                    if color[target] != next_color:
                        return False
                else:
                    if not dfs(target, next_color):
                        return False
        return True

    for sp in species:
        if sp not in color:
            if not dfs(sp, 0):
                return False
    return True
