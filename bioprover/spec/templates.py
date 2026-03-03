"""Bio-STL specification template library for BioProver.

Provides parameterised Bio-STL specification templates for common
biological circuit behaviours: Boolean logic correctness, oscillation,
bistability, adaptation, monotone dose-response, pulse generation,
steady-state convergence, rise-time, overshoot, and signal separation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bioprover.temporal.stl_ast import (
    Always,
    ComparisonOp,
    Eventually,
    Expression,
    Interval,
    Predicate,
    STLAnd,
    STLFormula,
    STLImplies,
    STLNot,
    STLOr,
    Until,
    make_var_expr,
)


# ---------------------------------------------------------------------------
# SpecificationTemplate
# ---------------------------------------------------------------------------

@dataclass
class TemplateParameter:
    """Descriptor for a template parameter.

    Attributes:
        name:        Parameter name used in the template.
        description: Human-readable description.
        default:     Default value (if any).
        units:       Expected units string.
        param_type:  ``"species"``, ``"threshold"``, ``"time"``, or ``"number"``.
    """

    name: str
    description: str
    default: Any = None
    units: str = ""
    param_type: str = "number"


@dataclass
class SpecificationTemplate:
    """A reusable Bio-STL specification template.

    Templates are parameterised by species names, thresholds, and time
    bounds.  Calling :meth:`instantiate` with concrete values produces
    an :class:`~bioprover.temporal.stl_ast.STLFormula`.

    Attributes:
        name:           Short template identifier.
        description:    Biological interpretation.
        parameters:     List of template parameters.
        builder:        Callable that maps parameter dict → STLFormula.
        category:       Functional category string.
        notes:          Usage notes or caveats.
    """

    name: str
    description: str
    parameters: List[TemplateParameter] = field(default_factory=list)
    builder: Optional[Any] = None  # Callable[[Dict[str, Any]], STLFormula]
    category: str = ""
    notes: str = ""

    def instantiate(self, **kwargs: Any) -> STLFormula:
        """Produce a concrete STL formula from template parameters.

        Missing parameters fall back to their declared defaults.

        Raises
        ------
        ValueError
            If a required parameter without a default is missing.
        """
        if self.builder is None:
            raise RuntimeError(f"Template '{self.name}' has no builder")

        # Fill defaults
        params: Dict[str, Any] = {}
        for p in self.parameters:
            if p.name in kwargs:
                params[p.name] = kwargs[p.name]
            elif p.default is not None:
                params[p.name] = p.default
            else:
                raise ValueError(
                    f"Template '{self.name}': required parameter "
                    f"'{p.name}' not provided"
                )
        return self.builder(params)

    def parameter_names(self) -> List[str]:
        return [p.name for p in self.parameters]

    def documentation(self) -> str:
        """Return a human-readable documentation string."""
        lines = [
            f"Template: {self.name}",
            f"  {self.description}",
            f"  Category: {self.category}",
            "  Parameters:",
        ]
        for p in self.parameters:
            default_str = f" [default={p.default}]" if p.default is not None else ""
            lines.append(f"    {p.name}: {p.description}{default_str}")
        if self.notes:
            lines.append(f"  Notes: {self.notes}")
        return "\n".join(lines)


# ======================================================================== #
#                         Builder functions                                 #
# ======================================================================== #

def _build_correct_boolean_logic(params: Dict[str, Any]) -> STLFormula:
    """Specification: genetic gate implements correct Boolean logic.

    For a NOT gate: when input > threshold, output should be < threshold
    (and vice versa), after a settling time.
    """
    inp: str = params["input_species"]
    out: str = params["output_species"]
    hi: float = params["high_threshold"]
    lo: float = params["low_threshold"]
    t_settle: float = params["settling_time"]
    t_end: float = params["time_horizon"]
    gate: str = params.get("gate_type", "NOT")

    if gate == "NOT":
        # Input high → output low (after settling)
        inp_hi = Predicate(make_var_expr(inp), ComparisonOp.GT, hi)
        out_lo = Predicate(make_var_expr(out), ComparisonOp.LT, lo)
        rule1 = Always(
            STLImplies(inp_hi, Eventually(out_lo, Interval(0, t_settle))),
            Interval(0, t_end),
        )
        # Input low → output high
        inp_lo = Predicate(make_var_expr(inp), ComparisonOp.LT, lo)
        out_hi = Predicate(make_var_expr(out), ComparisonOp.GT, hi)
        rule2 = Always(
            STLImplies(inp_lo, Eventually(out_hi, Interval(0, t_settle))),
            Interval(0, t_end),
        )
        return STLAnd(rule1, rule2)

    elif gate == "AND":
        a: str = params.get("input_species_b", "B")
        a_hi = Predicate(make_var_expr(inp), ComparisonOp.GT, hi)
        b_hi = Predicate(make_var_expr(a), ComparisonOp.GT, hi)
        out_hi = Predicate(make_var_expr(out), ComparisonOp.GT, hi)
        out_lo = Predicate(make_var_expr(out), ComparisonOp.LT, lo)
        # Both high → output high
        both = STLAnd(a_hi, b_hi)
        rule1 = Always(
            STLImplies(both, Eventually(out_hi, Interval(0, t_settle))),
            Interval(0, t_end),
        )
        # Either low → output low
        either_lo = STLOr(
            Predicate(make_var_expr(inp), ComparisonOp.LT, lo),
            Predicate(make_var_expr(a), ComparisonOp.LT, lo),
        )
        rule2 = Always(
            STLImplies(either_lo, Eventually(out_lo, Interval(0, t_settle))),
            Interval(0, t_end),
        )
        return STLAnd(rule1, rule2)

    # Default: simple threshold crossing
    inp_hi = Predicate(make_var_expr(inp), ComparisonOp.GT, hi)
    out_hi = Predicate(make_var_expr(out), ComparisonOp.GT, hi)
    return Always(
        STLImplies(inp_hi, Eventually(out_hi, Interval(0, t_settle))),
        Interval(0, t_end),
    )


def _build_oscillation(params: Dict[str, Any]) -> STLFormula:
    """Specification: sustained periodic oscillation."""
    species: str = params["species"]
    period: float = params["period"]
    amplitude: float = params["amplitude"]
    t_end: float = params["time_horizon"]
    half = period / 2.0

    peak = Predicate(make_var_expr(species), ComparisonOp.GT, amplitude)
    trough = Predicate(make_var_expr(species), ComparisonOp.LT,
                       params.get("trough_level", amplitude * 0.2))
    return Always(
        STLAnd(
            Eventually(peak, Interval(0, period)),
            Eventually(trough, Interval(0, period)),
        ),
        Interval(period, t_end),
    )


def _build_bistability(params: Dict[str, Any]) -> STLFormula:
    """Specification: bistability — system settles to one of two states."""
    species: str = params["species"]
    low_state: float = params["low_state"]
    high_state: float = params["high_state"]
    t_settle: float = params["settling_time"]
    t_end: float = params["time_horizon"]
    margin: float = params.get("margin", (high_state - low_state) * 0.1)

    near_low = STLAnd(
        Predicate(make_var_expr(species), ComparisonOp.GE, low_state - margin),
        Predicate(make_var_expr(species), ComparisonOp.LE, low_state + margin),
    )
    near_high = STLAnd(
        Predicate(make_var_expr(species), ComparisonOp.GE, high_state - margin),
        Predicate(make_var_expr(species), ComparisonOp.LE, high_state + margin),
    )
    stable_state = STLOr(near_low, near_high)
    return Eventually(
        Always(stable_state, Interval(0, t_end - t_settle)),
        Interval(0, t_settle),
    )


def _build_adaptation(params: Dict[str, Any]) -> STLFormula:
    """Specification: perfect adaptation — response then return to baseline."""
    species: str = params["species"]
    baseline: float = params["baseline"]
    peak_min: float = params["peak_min"]
    t_peak: float = params["peak_time"]
    t_adapt: float = params["adaptation_time"]
    tolerance: float = params.get("tolerance", baseline * 0.15)

    # Must reach peak
    responds = Eventually(
        Predicate(make_var_expr(species), ComparisonOp.GT, peak_min),
        Interval(0, t_peak),
    )
    # Must return to baseline
    returns = Eventually(
        STLAnd(
            Predicate(make_var_expr(species), ComparisonOp.GE,
                      baseline - tolerance),
            Predicate(make_var_expr(species), ComparisonOp.LE,
                      baseline + tolerance),
        ),
        Interval(t_peak, t_adapt),
    )
    return STLAnd(responds, returns)


def _build_monotone_dose_response(params: Dict[str, Any]) -> STLFormula:
    """Specification: monotone input-output relationship."""
    input_sp: str = params["input_species"]
    output_sp: str = params["output_species"]
    thresh_lo: float = params["input_low"]
    thresh_hi: float = params["input_high"]
    out_lo: float = params["output_low"]
    out_hi: float = params["output_high"]
    delay: float = params["response_delay"]
    t_end: float = params["time_horizon"]

    # Low input → low output
    rule_lo = Always(
        STLImplies(
            Predicate(make_var_expr(input_sp), ComparisonOp.LT, thresh_lo),
            Eventually(
                Predicate(make_var_expr(output_sp), ComparisonOp.LT, out_lo),
                Interval(0, delay),
            ),
        ),
        Interval(0, t_end),
    )
    # High input → high output
    rule_hi = Always(
        STLImplies(
            Predicate(make_var_expr(input_sp), ComparisonOp.GT, thresh_hi),
            Eventually(
                Predicate(make_var_expr(output_sp), ComparisonOp.GT, out_hi),
                Interval(0, delay),
            ),
        ),
        Interval(0, t_end),
    )
    return STLAnd(rule_lo, rule_hi)


def _build_pulse_generation(params: Dict[str, Any]) -> STLFormula:
    """Specification: transient pulse — peak followed by return."""
    species: str = params["species"]
    peak_level: float = params["peak_level"]
    baseline: float = params.get("baseline", peak_level * 0.1)
    t_peak: float = params["peak_time"]
    t_return: float = params["return_time"]

    spike = Eventually(
        Predicate(make_var_expr(species), ComparisonOp.GT, peak_level),
        Interval(0, t_peak),
    )
    returns = Eventually(
        Predicate(make_var_expr(species), ComparisonOp.LT, baseline),
        Interval(t_peak, t_return),
    )
    return STLAnd(spike, returns)


def _build_steady_state_convergence(params: Dict[str, Any]) -> STLFormula:
    """Specification: convergence to steady state within tolerance."""
    species: str = params["species"]
    target: float = params["target"]
    tolerance: float = params["tolerance"]
    t_conv: float = params["convergence_time"]
    t_stay: float = params.get("stay_duration", t_conv)

    in_band = STLAnd(
        Predicate(make_var_expr(species), ComparisonOp.GE, target - tolerance),
        Predicate(make_var_expr(species), ComparisonOp.LE, target + tolerance),
    )
    return Eventually(
        Always(in_band, Interval(0, t_stay)),
        Interval(0, t_conv),
    )


def _build_rise_time(params: Dict[str, Any]) -> STLFormula:
    """Specification: output must reach threshold within rise time."""
    species: str = params["species"]
    threshold: float = params["threshold"]
    t_rise: float = params["rise_time"]

    return Eventually(
        Predicate(make_var_expr(species), ComparisonOp.GE, threshold),
        Interval(0, t_rise),
    )


def _build_overshoot_constraint(params: Dict[str, Any]) -> STLFormula:
    """Specification: output must not overshoot a maximum level."""
    species: str = params["species"]
    max_level: float = params["max_level"]
    t_start: float = params.get("start_time", 0.0)
    t_end: float = params["time_horizon"]

    return Always(
        Predicate(make_var_expr(species), ComparisonOp.LE, max_level),
        Interval(t_start, t_end),
    )


def _build_separation(params: Dict[str, Any]) -> STLFormula:
    """Specification: two signals maintain minimum separation (don't cross)."""
    species_hi: str = params["species_high"]
    species_lo: str = params["species_low"]
    min_gap: float = params["min_gap"]
    t_start: float = params.get("start_time", 0.0)
    t_end: float = params["time_horizon"]

    # species_high - species_low >= min_gap
    # Approximated as: species_high > T_high AND species_low < T_low
    # where T_high - T_low = min_gap
    midpoint = min_gap / 2.0
    hi_above = Predicate(
        Expression(variable=species_hi, offset=-midpoint),
        ComparisonOp.GT, 0.0,
    )
    lo_below = Predicate(
        Expression(variable=species_lo, offset=midpoint),
        ComparisonOp.LT, min_gap,
    )
    return Always(STLAnd(hi_above, lo_below), Interval(t_start, t_end))


# ======================================================================== #
#  Extended Bio-STL: steady-state, oscillation, stochastic fragments       #
# ======================================================================== #

def _build_ss_convergence_tolerance(params: Dict[str, Any]) -> STLFormula:
    """SS[T,ε](φ): after settling_time, x stays within ε of steady-state value.

    Formula: G[T_settle, T_end]( x_ss - ε ≤ x ≤ x_ss + ε )
    """
    species: str = params["species"]
    x_ss: float = params["steady_state_value"]
    eps: float = params["tolerance"]
    t_settle: float = params["settling_time"]
    t_end: float = params.get("time_horizon", 10 * t_settle)

    in_band = STLAnd(
        Predicate(make_var_expr(species), ComparisonOp.GE, x_ss - eps),
        Predicate(make_var_expr(species), ComparisonOp.LE, x_ss + eps),
    )
    return Always(in_band, Interval(t_settle, t_end))


def _build_damped_oscillation(params: Dict[str, Any]) -> STLFormula:
    """Damped oscillation: successive peak amplitudes decrease by factor α.

    Encodes a two-cycle finite witness: in cycle 1 the signal peaks above an
    initial amplitude A, and in cycle 2 the peak is bounded above by A while
    still exceeding A·α (confirming continued but decaying oscillation).
    """
    species: str = params["species"]
    alpha: float = params["decay_rate"]
    p_min: float = params["min_period"]
    p_max: float = params["max_period"]
    initial_amplitude: float = params.get("initial_amplitude", 10.0)

    # Cycle 1 [0, p_max]: signal peaks above initial amplitude
    cycle1_peak = Eventually(
        Predicate(make_var_expr(species), ComparisonOp.GT, initial_amplitude),
        Interval(0, p_max),
    )
    # Cycle 1: signal also drops below decayed level (confirms oscillation)
    cycle1_trough = Eventually(
        Predicate(make_var_expr(species), ComparisonOp.LT,
                  initial_amplitude * alpha),
        Interval(0, p_max),
    )
    # Cycle 2 [p_min, 2·p_max]: peak bounded by initial amplitude (decay)
    cycle2_bounded = Always(
        Predicate(make_var_expr(species), ComparisonOp.LT, initial_amplitude),
        Interval(p_min, 2 * p_max),
    )
    # Cycle 2: still oscillating — exceeds decayed amplitude
    cycle2_peak = Eventually(
        Predicate(make_var_expr(species), ComparisonOp.GT,
                  initial_amplitude * alpha),
        Interval(p_min, 2 * p_max),
    )

    return STLAnd(
        STLAnd(cycle1_peak, cycle1_trough),
        STLAnd(cycle2_peak, cycle2_bounded),
    )


def _build_sustained_oscillation(params: Dict[str, Any]) -> STLFormula:
    """Sustained oscillation with period and amplitude bounds.

    Formula: G[T_settle, T_end]( F[0, P_max](x > A_high) ∧ F[0, P_max](x < A_low) )
    """
    species: str = params["species"]
    p_min: float = params["period_min"]
    p_max: float = params["period_max"]
    a_hi: float = params["amplitude_high"]
    a_lo: float = params["amplitude_low"]
    t_settle: float = params["settling_time"]
    t_end: float = params.get("time_horizon", t_settle + 10 * p_max)

    reaches_high = Eventually(
        Predicate(make_var_expr(species), ComparisonOp.GT, a_hi),
        Interval(0, p_max),
    )
    reaches_low = Eventually(
        Predicate(make_var_expr(species), ComparisonOp.LT, a_lo),
        Interval(0, p_max),
    )
    return Always(STLAnd(reaches_high, reaches_low), Interval(t_settle, t_end))


def _build_probability_threshold(params: Dict[str, Any]) -> STLFormula:
    """Stochastic threshold: P( G[0,T](x > threshold) ) ≥ p_min.

    Returns the inner sample-path STL formula G[0,T](x > threshold).
    The probabilistic bound min_probability is enforced by the stochastic
    verification engine via statistical model checking.
    """
    species: str = params["species"]
    threshold: float = params["threshold"]
    t_horizon: float = params["time_horizon"]
    # min_probability is consumed by the stochastic verification layer
    return Always(
        Predicate(make_var_expr(species), ComparisonOp.GT, threshold),
        Interval(0, t_horizon),
    )


def _build_bimodal_steady_state(params: Dict[str, Any]) -> STLFormula:
    """Bimodal distribution: steady state near one of two distinct modes.

    Formula: G[T_settle, T_end]( (|x - m1| < sep/2) ∨ (|x - m2| < sep/2) )
    Each trajectory should settle near one of the two modes.
    """
    species: str = params["species"]
    m1: float = params["mode1_center"]
    m2: float = params["mode2_center"]
    sep: float = params["separation"]
    t_settle: float = params.get("settling_time", 50.0)
    t_end: float = params.get("time_horizon", 200.0)
    half_sep = sep / 2.0

    near_mode1 = STLAnd(
        Predicate(make_var_expr(species), ComparisonOp.GE, m1 - half_sep),
        Predicate(make_var_expr(species), ComparisonOp.LE, m1 + half_sep),
    )
    near_mode2 = STLAnd(
        Predicate(make_var_expr(species), ComparisonOp.GE, m2 - half_sep),
        Predicate(make_var_expr(species), ComparisonOp.LE, m2 + half_sep),
    )
    return Always(STLOr(near_mode1, near_mode2), Interval(t_settle, t_end))


def _build_switching_rate(params: Dict[str, Any]) -> STLFormula:
    """Bounded noise-induced switching between bistable states.

    Once the signal enters a state (high or low), it must dwell there for
    at least 1/max_switching_rate time units, bounding the transition rate.
    """
    species: str = params["species"]
    s_lo: float = params["state_low"]
    s_hi: float = params["state_high"]
    max_rate: float = params["max_switching_rate"]
    t_obs: float = params["observation_time"]
    min_dwell = 1.0 / max_rate if max_rate > 0 else t_obs

    # In high state → stay above low boundary for min_dwell time
    in_high = Predicate(make_var_expr(species), ComparisonOp.GT, s_hi)
    stay_above = Always(
        Predicate(make_var_expr(species), ComparisonOp.GT, s_lo),
        Interval(0, min_dwell),
    )
    high_dwell = Always(
        STLImplies(in_high, stay_above), Interval(0, t_obs),
    )

    # In low state → stay below high boundary for min_dwell time
    in_low = Predicate(make_var_expr(species), ComparisonOp.LT, s_lo)
    stay_below = Always(
        Predicate(make_var_expr(species), ComparisonOp.LT, s_hi),
        Interval(0, min_dwell),
    )
    low_dwell = Always(
        STLImplies(in_low, stay_below), Interval(0, t_obs),
    )

    return STLAnd(high_dwell, low_dwell)


# ======================================================================== #
#                         TemplateLibrary                                   #
# ======================================================================== #

class TemplateLibrary:
    """Registry of specification templates.

    Ships with ~10 built-in templates covering common biological circuit
    behaviours.  Users may add custom templates.
    """

    def __init__(self, *, load_builtins: bool = True) -> None:
        self._templates: Dict[str, SpecificationTemplate] = {}
        if load_builtins:
            self._register_builtins()

    # -- CRUD ---------------------------------------------------------------

    def register(self, template: SpecificationTemplate) -> None:
        self._templates[template.name] = template

    def get(self, name: str) -> Optional[SpecificationTemplate]:
        return self._templates.get(name)

    @property
    def names(self) -> List[str]:
        return sorted(self._templates)

    @property
    def all_templates(self) -> List[SpecificationTemplate]:
        return list(self._templates.values())

    def __len__(self) -> int:
        return len(self._templates)

    def __contains__(self, name: str) -> bool:
        return name in self._templates

    # -- search -------------------------------------------------------------

    def search_by_category(self, category: str) -> List[SpecificationTemplate]:
        cat = category.lower()
        return [t for t in self._templates.values() if cat in t.category.lower()]

    def search_by_keyword(self, keyword: str) -> List[SpecificationTemplate]:
        kw = keyword.lower()
        return [
            t for t in self._templates.values()
            if kw in t.description.lower() or kw in t.name.lower()
        ]

    # -- composition --------------------------------------------------------

    @staticmethod
    def compose(*specs: STLFormula) -> STLFormula:
        """Compose multiple specifications via conjunction (AND)."""
        if not specs:
            raise ValueError("At least one specification required")
        result = specs[0]
        for s in specs[1:]:
            result = STLAnd(result, s)
        return result

    # -- documentation ------------------------------------------------------

    def documentation(self) -> str:
        """Return full documentation for all templates."""
        lines: List[str] = ["Specification Template Library", "=" * 35]
        for t in sorted(self._templates.values(), key=lambda x: x.name):
            lines.append("")
            lines.append(t.documentation())
        return "\n".join(lines)

    # -- built-ins ----------------------------------------------------------

    def _register_builtins(self) -> None:
        """Register built-in specification templates."""

        # 1. Correct Boolean logic
        self.register(SpecificationTemplate(
            name="correct_boolean_logic",
            description="Genetic gate implements correct Boolean logic "
                        "(NOT, AND, OR, etc.)",
            category="logic",
            builder=_build_correct_boolean_logic,
            parameters=[
                TemplateParameter("input_species", "Input signal species name",
                                  param_type="species"),
                TemplateParameter("output_species", "Output reporter species name",
                                  param_type="species"),
                TemplateParameter("high_threshold", "Concentration defining logic HIGH",
                                  default=5.0, units="nM", param_type="threshold"),
                TemplateParameter("low_threshold", "Concentration defining logic LOW",
                                  default=1.0, units="nM", param_type="threshold"),
                TemplateParameter("settling_time", "Time to reach stable output",
                                  default=30.0, units="min", param_type="time"),
                TemplateParameter("time_horizon", "Total specification time",
                                  default=200.0, units="min", param_type="time"),
                TemplateParameter("gate_type", "Boolean function (NOT, AND, OR)",
                                  default="NOT"),
            ],
            notes="For multi-input gates, supply 'input_species_b' in kwargs.",
        ))

        # 2. Oscillation
        self.register(SpecificationTemplate(
            name="oscillation",
            description="Sustained periodic oscillation: repeated peaks "
                        "and troughs within a period window",
            category="dynamic",
            builder=_build_oscillation,
            parameters=[
                TemplateParameter("species", "Oscillating species name",
                                  param_type="species"),
                TemplateParameter("period", "Expected oscillation period",
                                  default=40.0, units="min", param_type="time"),
                TemplateParameter("amplitude", "Minimum peak concentration",
                                  default=3.0, units="nM", param_type="threshold"),
                TemplateParameter("time_horizon", "Total observation time",
                                  default=200.0, units="min", param_type="time"),
            ],
        ))

        # 3. Bistability
        self.register(SpecificationTemplate(
            name="bistability",
            description="Two stable steady states with switching: system "
                        "settles near one of two concentration levels",
            category="memory",
            builder=_build_bistability,
            parameters=[
                TemplateParameter("species", "Bistable species name",
                                  param_type="species"),
                TemplateParameter("low_state", "Concentration of low stable state",
                                  default=1.0, units="nM", param_type="threshold"),
                TemplateParameter("high_state", "Concentration of high stable state",
                                  default=10.0, units="nM", param_type="threshold"),
                TemplateParameter("settling_time", "Time to settle into state",
                                  default=50.0, units="min", param_type="time"),
                TemplateParameter("time_horizon", "Total specification time",
                                  default=200.0, units="min", param_type="time"),
            ],
        ))

        # 4. Adaptation
        self.register(SpecificationTemplate(
            name="adaptation",
            description="Perfect adaptation: transient response followed by "
                        "return to baseline level",
            category="dynamic",
            builder=_build_adaptation,
            parameters=[
                TemplateParameter("species", "Adapting species name",
                                  param_type="species"),
                TemplateParameter("baseline", "Baseline (pre-stimulus) level",
                                  default=0.5, units="nM", param_type="threshold"),
                TemplateParameter("peak_min", "Minimum peak during response",
                                  default=5.0, units="nM", param_type="threshold"),
                TemplateParameter("peak_time", "Time to reach peak",
                                  default=10.0, units="min", param_type="time"),
                TemplateParameter("adaptation_time", "Time to return to baseline",
                                  default=60.0, units="min", param_type="time"),
            ],
        ))

        # 5. Monotone dose-response
        self.register(SpecificationTemplate(
            name="monotone_dose_response",
            description="Monotone input-output relationship: higher input "
                        "leads to higher (or lower) output",
            category="transfer_function",
            builder=_build_monotone_dose_response,
            parameters=[
                TemplateParameter("input_species", "Input signal species",
                                  param_type="species"),
                TemplateParameter("output_species", "Output reporter species",
                                  param_type="species"),
                TemplateParameter("input_low", "Low input threshold",
                                  default=1.0, units="nM", param_type="threshold"),
                TemplateParameter("input_high", "High input threshold",
                                  default=10.0, units="nM", param_type="threshold"),
                TemplateParameter("output_low", "Expected low output",
                                  default=1.0, units="nM", param_type="threshold"),
                TemplateParameter("output_high", "Expected high output",
                                  default=8.0, units="nM", param_type="threshold"),
                TemplateParameter("response_delay", "Max input-to-output delay",
                                  default=20.0, units="min", param_type="time"),
                TemplateParameter("time_horizon", "Total specification time",
                                  default=200.0, units="min", param_type="time"),
            ],
        ))

        # 6. Pulse generation
        self.register(SpecificationTemplate(
            name="pulse_generation",
            description="Transient pulse: signal spikes above a level then "
                        "returns to baseline",
            category="dynamic",
            builder=_build_pulse_generation,
            parameters=[
                TemplateParameter("species", "Pulsing species name",
                                  param_type="species"),
                TemplateParameter("peak_level", "Minimum peak amplitude",
                                  default=5.0, units="nM", param_type="threshold"),
                TemplateParameter("peak_time", "Time window for peak",
                                  default=15.0, units="min", param_type="time"),
                TemplateParameter("return_time", "Time by which signal returns",
                                  default=60.0, units="min", param_type="time"),
            ],
        ))

        # 7. Steady-state convergence
        self.register(SpecificationTemplate(
            name="steady_state_convergence",
            description="Convergence to a target steady-state value within "
                        "a tolerance band",
            category="stability",
            builder=_build_steady_state_convergence,
            parameters=[
                TemplateParameter("species", "Converging species name",
                                  param_type="species"),
                TemplateParameter("target", "Target steady-state value",
                                  default=5.0, units="nM", param_type="threshold"),
                TemplateParameter("tolerance", "Acceptable deviation from target",
                                  default=0.5, units="nM", param_type="threshold"),
                TemplateParameter("convergence_time", "Max time to reach steady state",
                                  default=50.0, units="min", param_type="time"),
            ],
        ))

        # 8. Rise time
        self.register(SpecificationTemplate(
            name="rise_time",
            description="Output must reach a threshold within a specified "
                        "rise time",
            category="performance",
            builder=_build_rise_time,
            parameters=[
                TemplateParameter("species", "Rising species name",
                                  param_type="species"),
                TemplateParameter("threshold", "Target threshold to reach",
                                  default=5.0, units="nM", param_type="threshold"),
                TemplateParameter("rise_time", "Maximum allowed rise time",
                                  default=20.0, units="min", param_type="time"),
            ],
        ))

        # 9. Overshoot constraint
        self.register(SpecificationTemplate(
            name="overshoot_constraint",
            description="Output must not exceed a maximum concentration level "
                        "(overshoot prevention)",
            category="performance",
            builder=_build_overshoot_constraint,
            parameters=[
                TemplateParameter("species", "Constrained species name",
                                  param_type="species"),
                TemplateParameter("max_level", "Maximum allowed concentration",
                                  default=15.0, units="nM", param_type="threshold"),
                TemplateParameter("time_horizon", "Observation window",
                                  default=200.0, units="min", param_type="time"),
            ],
        ))

        # 10. Signal separation
        self.register(SpecificationTemplate(
            name="separation",
            description="Two signals maintain a minimum separation (never "
                        "cross); useful for toggle-switch verification",
            category="memory",
            builder=_build_separation,
            parameters=[
                TemplateParameter("species_high", "Species expected to be higher",
                                  param_type="species"),
                TemplateParameter("species_low", "Species expected to be lower",
                                  param_type="species"),
                TemplateParameter("min_gap", "Minimum concentration gap",
                                  default=3.0, units="nM", param_type="threshold"),
                TemplateParameter("time_horizon", "Observation window",
                                  default=200.0, units="min", param_type="time"),
            ],
        ))

        # ---- Extended Bio-STL fragments (steady-state, oscillation, stochastic) ----

        # 11. Steady-state convergence with tolerance — SS[T,ε](φ)
        self.register(SpecificationTemplate(
            name="steady_state_convergence",
            description="SS[T,ε](φ): after settling time T, species x stays "
                        "within tolerance ε of its steady-state value x_ss. "
                        "Formula: G[T, T_end]( |x - x_ss| < ε )",
            category="stability",
            builder=_build_ss_convergence_tolerance,
            parameters=[
                TemplateParameter("species", "Species to monitor",
                                  param_type="species"),
                TemplateParameter("steady_state_value",
                                  "Expected steady-state concentration",
                                  default=5.0, units="nM",
                                  param_type="threshold"),
                TemplateParameter("tolerance",
                                  "Acceptable deviation from steady state",
                                  default=0.5, units="nM",
                                  param_type="threshold"),
                TemplateParameter("settling_time",
                                  "Time after which convergence must hold",
                                  default=50.0, units="min", param_type="time"),
                TemplateParameter("time_horizon",
                                  "End of observation window",
                                  default=500.0, units="min", param_type="time"),
            ],
            notes="Overwrites the simpler steady_state_convergence template "
                  "with an explicit settling-time + tolerance formulation.",
        ))

        # 12. Damped oscillation detection
        self.register(SpecificationTemplate(
            name="damped_oscillation",
            description="Damped oscillation: successive peak amplitudes "
                        "decrease by decay factor α. Two-cycle finite "
                        "witness checks peak decay between periods.",
            category="dynamic",
            builder=_build_damped_oscillation,
            parameters=[
                TemplateParameter("species", "Oscillating species",
                                  param_type="species"),
                TemplateParameter("decay_rate",
                                  "Fraction of amplitude retained per cycle "
                                  "(0 < α < 1)",
                                  default=0.5, param_type="number"),
                TemplateParameter("min_period",
                                  "Minimum oscillation period",
                                  default=20.0, units="min", param_type="time"),
                TemplateParameter("max_period",
                                  "Maximum oscillation period",
                                  default=60.0, units="min", param_type="time"),
                TemplateParameter("initial_amplitude",
                                  "Reference peak amplitude in cycle 1",
                                  default=10.0, units="nM",
                                  param_type="threshold"),
            ],
            notes="Uses a finite two-cycle unrolling; extend with additional "
                  "cycles for stronger guarantees.",
        ))

        # 13. Sustained oscillation with period and amplitude bounds
        self.register(SpecificationTemplate(
            name="sustained_oscillation",
            description="Sustained oscillation: after settling, signal "
                        "repeatedly exceeds amplitude_high and drops below "
                        "amplitude_low within each period window. "
                        "Formula: G[T_settle, T_end]( F[0, P_max](x > A_hi) "
                        "∧ F[0, P_max](x < A_lo) )",
            category="dynamic",
            builder=_build_sustained_oscillation,
            parameters=[
                TemplateParameter("species", "Oscillating species",
                                  param_type="species"),
                TemplateParameter("period_min",
                                  "Minimum oscillation period",
                                  default=20.0, units="min", param_type="time"),
                TemplateParameter("period_max",
                                  "Maximum oscillation period",
                                  default=60.0, units="min", param_type="time"),
                TemplateParameter("amplitude_high",
                                  "Threshold that must be exceeded each cycle",
                                  default=8.0, units="nM",
                                  param_type="threshold"),
                TemplateParameter("amplitude_low",
                                  "Threshold that must be undercut each cycle",
                                  default=2.0, units="nM",
                                  param_type="threshold"),
                TemplateParameter("settling_time",
                                  "Transient settling time before checking",
                                  default=30.0, units="min", param_type="time"),
                TemplateParameter("time_horizon",
                                  "End of observation window",
                                  default=500.0, units="min", param_type="time"),
            ],
        ))

        # 14. Stochastic threshold probability — P(G[0,T](x > θ)) ≥ p_min
        self.register(SpecificationTemplate(
            name="probability_threshold",
            description="Probabilistic specification: P( G[0,T](x > θ) ) ≥ "
                        "p_min. Builder returns the inner sample-path STL "
                        "formula; min_probability is consumed by the "
                        "stochastic verification engine.",
            category="stochastic",
            builder=_build_probability_threshold,
            parameters=[
                TemplateParameter("species", "Species to monitor",
                                  param_type="species"),
                TemplateParameter("threshold",
                                  "Concentration threshold θ",
                                  default=5.0, units="nM",
                                  param_type="threshold"),
                TemplateParameter("time_horizon",
                                  "Time horizon T for the inner G operator",
                                  default=100.0, units="min", param_type="time"),
                TemplateParameter("min_probability",
                                  "Minimum probability p_min (used by "
                                  "statistical model checker)",
                                  default=0.95, param_type="number"),
            ],
            notes="The returned STL formula is the deterministic sample-path "
                  "property. Use min_probability with a statistical model "
                  "checker (e.g., sequential hypothesis testing) to enforce "
                  "the probabilistic bound.",
        ))

        # 15. Bimodal steady-state distribution detection
        self.register(SpecificationTemplate(
            name="bimodal_steady_state",
            description="Bimodal distribution: after settling, the signal "
                        "stays near one of two distinct modes. "
                        "Formula: G[T, T_end]( |x-m1|<sep/2 ∨ |x-m2|<sep/2 )",
            category="stochastic",
            builder=_build_bimodal_steady_state,
            parameters=[
                TemplateParameter("species", "Species to monitor",
                                  param_type="species"),
                TemplateParameter("mode1_center",
                                  "Center of first distribution mode",
                                  default=2.0, units="nM",
                                  param_type="threshold"),
                TemplateParameter("mode2_center",
                                  "Center of second distribution mode",
                                  default=8.0, units="nM",
                                  param_type="threshold"),
                TemplateParameter("separation",
                                  "Width of acceptance band around each mode",
                                  default=2.0, units="nM",
                                  param_type="threshold"),
                TemplateParameter("settling_time",
                                  "Transient settling time",
                                  default=50.0, units="min", param_type="time"),
                TemplateParameter("time_horizon",
                                  "End of observation window",
                                  default=200.0, units="min", param_type="time"),
            ],
            notes="Verify across an ensemble of stochastic trajectories to "
                  "confirm bimodality of the steady-state distribution.",
        ))

        # 16. Noise-induced switching rate
        self.register(SpecificationTemplate(
            name="switching_rate",
            description="Bounded noise-induced switching: transitions between "
                        "bistable states (high/low) occur at most at "
                        "max_switching_rate, enforced via minimum dwell time "
                        "1/rate in each state.",
            category="stochastic",
            builder=_build_switching_rate,
            parameters=[
                TemplateParameter("species", "Bistable species",
                                  param_type="species"),
                TemplateParameter("state_low",
                                  "Upper boundary of low stable state",
                                  default=2.0, units="nM",
                                  param_type="threshold"),
                TemplateParameter("state_high",
                                  "Lower boundary of high stable state",
                                  default=8.0, units="nM",
                                  param_type="threshold"),
                TemplateParameter("max_switching_rate",
                                  "Maximum allowed transitions per time unit",
                                  default=0.1, units="1/min",
                                  param_type="number"),
                TemplateParameter("observation_time",
                                  "Total observation window",
                                  default=200.0, units="min", param_type="time"),
            ],
            notes="Minimum dwell time = 1/max_switching_rate. Useful for "
                  "verifying that stochastic noise does not cause excessively "
                  "rapid toggling between states.",
        ))
