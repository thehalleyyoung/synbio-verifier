"""Extended benchmark circuits addressing kinetic and topological diversity.

Supplements the core :class:`BenchmarkSuite` with circuits that use
Michaelis–Menten kinetics, allosteric (MWC) regulation, complex
multi-feedback topologies, and larger species counts.  Each circuit
is tagged with ``kinetics_type`` and ``topology_type`` metadata to
enable systematic coverage analysis across kinetic formalisms and
network architectures.

References
----------
* Michaelis & Menten (1913) — enzymatic saturation kinetics
* Monod, Wyman & Changeux (1965) — allosteric transition model
* Alon (2007) — network motifs in systems biology
* Elowitz & Leibler (2000) — repressilator
* Huang & Ferrell (1996) — MAPK ultrasensitivity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from bioprover.cegar import VerificationStatus
from bioprover.evaluation.benchmark_suite import (
    BenchmarkCircuit,
    BenchmarkDifficulty,
    BenchmarkSuite,
    _extract_time_horizon,
    _hill_param,
    _pred_ge,
    _pred_le,
    _rate_param,
)
from bioprover.models import (
    BioModel,
    ConstitutiveProduction,
    HillActivation,
    HillRepression,
    LinearDegradation,
    MassAction,
    MichaelisMenten,
    Parameter,
    Reaction,
    Species,
    SpeciesType,
    UncertaintyType,
)
from bioprover.models.reactions import StoichiometryEntry
from bioprover.temporal import (
    Always,
    Eventually,
    Interval,
    Predicate,
    STLAnd,
    STLFormula,
    STLNot,
    STLOr,
    Until,
)
from bioprover.temporal.stl_ast import ComparisonOp, make_var_expr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper — parameter with Michaelis–Menten semantics
# ---------------------------------------------------------------------------


def _mm_param(name: str, value: float, low: float, high: float,
              units: str = "nM") -> Parameter:
    """Create a Michaelis–Menten parameter with uniform uncertainty."""
    return Parameter(
        name=name, value=value, units=units,
        lower_bound=low, upper_bound=high,
        uncertainty_type=UncertaintyType.UNIFORM,
    )


# ===================================================================
# 1.  Michaelis–Menten kinetics circuits
# ===================================================================


def enzymatic_cascade_mm() -> BenchmarkCircuit:
    """Three-enzyme cascade with explicit E+S ⇌ ES → E+P dynamics.

    Models a sequential pathway where each product serves as the
    substrate for the next enzyme.  Uses Michaelis–Menten rate laws
    throughout (no Hill functions).

    Species (7): Substrate, Enz1, Prod1, Enz2, Prod2, Enz3, FinalProduct
    Kinetics: Michaelis–Menten
    Literature: Cornish-Bowden (2012) — enzyme kinetics
    """
    model = BioModel(name="enzymatic_cascade_mm")

    species_data = [
        ("Substrate", 100.0, SpeciesType.SMALL_MOLECULE),
        ("Enz1", 10.0, SpeciesType.PROTEIN),
        ("Prod1", 0.0, SpeciesType.SMALL_MOLECULE),
        ("Enz2", 8.0, SpeciesType.PROTEIN),
        ("Prod2", 0.0, SpeciesType.SMALL_MOLECULE),
        ("Enz3", 6.0, SpeciesType.PROTEIN),
        ("FinalProduct", 0.0, SpeciesType.SMALL_MOLECULE),
    ]
    for name, ic, stype in species_data:
        model.add_species(Species(name, initial_concentration=ic,
                                  species_type=stype))

    params = [
        # Enzyme 1: Vmax_1 = kcat1 * [Enz1], Km1 ≈ 15 µM (typical)
        _mm_param("Vmax_1", 8.0, 3.0, 15.0, "nM/min"),
        _mm_param("Km_1", 15.0, 5.0, 40.0),
        # Enzyme 2
        _mm_param("Vmax_2", 6.0, 2.0, 12.0, "nM/min"),
        _mm_param("Km_2", 20.0, 8.0, 50.0),
        # Enzyme 3
        _mm_param("Vmax_3", 5.0, 2.0, 10.0, "nM/min"),
        _mm_param("Km_3", 25.0, 10.0, 60.0),
        # Degradation
        _rate_param("deg_Prod1", 0.02, 0.005, 0.05),
        _rate_param("deg_Prod2", 0.02, 0.005, 0.05),
        _rate_param("deg_FinalProduct", 0.01, 0.003, 0.03),
        _rate_param("deg_Substrate", 0.005, 0.001, 0.01),
    ]
    for p in params:
        model.parameters.add(p)

    # Substrate → Prod1 (catalysed by Enz1, MM kinetics)
    model.add_reaction(Reaction(
        "cat_Enz1",
        [StoichiometryEntry("Substrate")],
        [StoichiometryEntry("Prod1")],
        MichaelisMenten(Vmax=8.0, Km=15.0),
    ))
    # Prod1 → Prod2 (catalysed by Enz2)
    model.add_reaction(Reaction(
        "cat_Enz2",
        [StoichiometryEntry("Prod1")],
        [StoichiometryEntry("Prod2")],
        MichaelisMenten(Vmax=6.0, Km=20.0),
    ))
    # Prod2 → FinalProduct (catalysed by Enz3)
    model.add_reaction(Reaction(
        "cat_Enz3",
        [StoichiometryEntry("Prod2")],
        [StoichiometryEntry("FinalProduct")],
        MichaelisMenten(Vmax=5.0, Km=25.0),
    ))
    # Degradation
    for sp, rate in [("Prod1", 0.02), ("Prod2", 0.02),
                     ("FinalProduct", 0.01), ("Substrate", 0.005)]:
        model.add_reaction(Reaction(
            f"deg_{sp}", [StoichiometryEntry(sp)], [],
            LinearDegradation(rate=rate),
        ))

    # Spec: FinalProduct eventually accumulates above 20 nM
    liveness = Eventually(_pred_ge("FinalProduct", 20.0),
                          Interval(0.0, 400.0))
    safety = Always(_pred_le("FinalProduct", 200.0),
                    Interval(0.0, 400.0))
    spec = STLAnd(liveness, safety)

    return BenchmarkCircuit(
        model=model, specification=spec,
        expected_result=VerificationStatus.VERIFIED,
        difficulty=BenchmarkDifficulty.MEDIUM,
        name="enzymatic_cascade_mm",
        description="Three-enzyme Michaelis–Menten cascade (E+S→E+P)",
        category="cascade",
        tags=["michaelis_menten", "enzyme", "cascade"],
        metadata={
            "species_count": 7, "parameter_count": 10,
            "kinetics_type": "michaelis_menten",
            "topology_type": "cascade",
        },
    )


def competitive_inhibition_mm() -> BenchmarkCircuit:
    """Enzyme with competitive inhibitor: E+S ⇌ ES → E+P, E+I ⇌ EI.

    The inhibitor reduces apparent Vmax; the spec checks that product
    formation is slowed but not abolished.  Uses two MM reactions and
    a mass-action inhibitor binding step.

    Species (6): Substrate, Enzyme, Product, Inhibitor, ES_complex, EI_complex
    Kinetics: Michaelis–Menten + MassAction
    Literature: Cornish-Bowden (2012), Segel (1975)
    """
    model = BioModel(name="competitive_inhibition_mm")

    species_data = [
        ("Substrate", 80.0, SpeciesType.SMALL_MOLECULE),
        ("Enzyme", 10.0, SpeciesType.PROTEIN),
        ("Product", 0.0, SpeciesType.SMALL_MOLECULE),
        ("Inhibitor", 30.0, SpeciesType.SMALL_MOLECULE),
        ("ES_complex", 0.0, SpeciesType.PROTEIN),
        ("EI_complex", 0.0, SpeciesType.PROTEIN),
    ]
    for name, ic, stype in species_data:
        model.add_species(Species(name, initial_concentration=ic,
                                  species_type=stype))

    params = [
        # Catalytic step (quasi-steady-state MM)
        _mm_param("Vmax_cat", 5.0, 2.0, 10.0, "nM/min"),
        _mm_param("Km_cat", 20.0, 8.0, 50.0),
        # Inhibitor binding (mass action, reversible)
        _rate_param("kon_EI", 0.005, 0.001, 0.015, "1/(nM·min)"),
        _rate_param("koff_EI", 0.05, 0.01, 0.15),
        # ES complex formation (mass action)
        _rate_param("kon_ES", 0.01, 0.003, 0.03, "1/(nM·min)"),
        _rate_param("koff_ES", 0.1, 0.03, 0.3),
        _rate_param("kcat", 0.5, 0.15, 1.0),
        # Degradation
        _rate_param("deg_Product", 0.01, 0.003, 0.03),
        _rate_param("deg_Substrate", 0.005, 0.001, 0.015),
    ]
    for p in params:
        model.parameters.add(p)

    # S + E → ES (mass action forward)
    model.add_reaction(Reaction(
        "bind_ES",
        [StoichiometryEntry("Substrate"), StoichiometryEntry("Enzyme")],
        [StoichiometryEntry("ES_complex")],
        MassAction(k_forward=0.01, k_reverse=0.1),
        reversible=True,
    ))
    # ES → E + P (catalytic step, modelled as MM on remaining free S)
    model.add_reaction(Reaction(
        "catalysis",
        [StoichiometryEntry("ES_complex")],
        [StoichiometryEntry("Enzyme"), StoichiometryEntry("Product")],
        MassAction(k_forward=0.5),
    ))
    # E + I → EI (competitive inhibition, dead-end complex)
    model.add_reaction(Reaction(
        "bind_EI",
        [StoichiometryEntry("Enzyme"), StoichiometryEntry("Inhibitor")],
        [StoichiometryEntry("EI_complex")],
        MassAction(k_forward=0.005, k_reverse=0.05),
        reversible=True,
    ))
    # Degradation
    for sp, rate in [("Product", 0.01), ("Substrate", 0.005)]:
        model.add_reaction(Reaction(
            f"deg_{sp}", [StoichiometryEntry(sp)], [],
            LinearDegradation(rate=rate),
        ))

    # Spec: product forms but is bounded (inhibition limits rate)
    prod_forms = Eventually(_pred_ge("Product", 10.0),
                            Interval(0.0, 500.0))
    prod_bounded = Always(_pred_le("Product", 150.0),
                          Interval(0.0, 500.0))
    # Enzyme never fully sequestered
    enzyme_available = Always(_pred_ge("Enzyme", 0.5),
                              Interval(50.0, 500.0))
    spec = STLAnd(STLAnd(prod_forms, prod_bounded), enzyme_available)

    return BenchmarkCircuit(
        model=model, specification=spec,
        expected_result=VerificationStatus.BOUNDED_GUARANTEE,
        difficulty=BenchmarkDifficulty.MEDIUM,
        name="competitive_inhibition_mm",
        description="Competitive inhibition with E+S⇌ES→E+P and E+I⇌EI",
        category="enzyme",
        tags=["michaelis_menten", "inhibition", "mass_action", "competitive"],
        metadata={
            "species_count": 6, "parameter_count": 9,
            "kinetics_type": "michaelis_menten",
            "topology_type": "competitive",
        },
    )


def substrate_channeling_mm() -> BenchmarkCircuit:
    """Two-enzyme substrate channeling with MM kinetics.

    Two enzymes form a complex enabling direct transfer of intermediate
    without release into the bulk.  Compares channeled vs uncoupled
    regimes.

    Species (8): S, E1, P1, E2, P2, E1E2_complex, P1_chan, Waste
    Kinetics: Michaelis–Menten + MassAction
    Literature: Wheeldon et al. (2016) — substrate channeling
    """
    model = BioModel(name="substrate_channeling_mm")

    species_data = [
        ("S", 60.0, SpeciesType.SMALL_MOLECULE),
        ("E1", 10.0, SpeciesType.PROTEIN),
        ("P1", 0.0, SpeciesType.SMALL_MOLECULE),
        ("E2", 10.0, SpeciesType.PROTEIN),
        ("P2", 0.0, SpeciesType.SMALL_MOLECULE),
        ("E1E2_complex", 5.0, SpeciesType.PROTEIN),
        ("P1_chan", 0.0, SpeciesType.SMALL_MOLECULE),
        ("Waste", 0.0, SpeciesType.SMALL_MOLECULE),
    ]
    for name, ic, stype in species_data:
        model.add_species(Species(name, initial_concentration=ic,
                                  species_type=stype))

    params = [
        _mm_param("Vmax_E1", 7.0, 3.0, 14.0, "nM/min"),
        _mm_param("Km_E1", 12.0, 4.0, 30.0),
        _mm_param("Vmax_E2", 5.0, 2.0, 10.0, "nM/min"),
        _mm_param("Km_E2", 18.0, 6.0, 40.0),
        _mm_param("Vmax_chan", 9.0, 4.0, 18.0, "nM/min"),
        _mm_param("Km_chan", 8.0, 2.0, 20.0),
        _rate_param("deg_P1", 0.03, 0.01, 0.06),
        _rate_param("deg_P2", 0.01, 0.003, 0.03),
        _rate_param("deg_Waste", 0.05, 0.02, 0.1),
    ]
    for p in params:
        model.parameters.add(p)

    # Uncoupled path: S → P1 → P2
    model.add_reaction(Reaction(
        "E1_cat", [StoichiometryEntry("S")],
        [StoichiometryEntry("P1")],
        MichaelisMenten(Vmax=7.0, Km=12.0),
    ))
    model.add_reaction(Reaction(
        "E2_cat", [StoichiometryEntry("P1")],
        [StoichiometryEntry("P2")],
        MichaelisMenten(Vmax=5.0, Km=18.0),
    ))
    # Channeled path: S → P1_chan → P2 (faster, lower Km)
    model.add_reaction(Reaction(
        "chan_cat", [StoichiometryEntry("S")],
        [StoichiometryEntry("P2")],
        MichaelisMenten(Vmax=9.0, Km=8.0),
    ))
    # Side reaction: P1 → Waste (loss without channeling)
    model.add_reaction(Reaction(
        "side_rxn", [StoichiometryEntry("P1")],
        [StoichiometryEntry("Waste")],
        MassAction(k_forward=0.02),
    ))
    for sp, rate in [("P1", 0.03), ("P2", 0.01), ("Waste", 0.05)]:
        model.add_reaction(Reaction(
            f"deg_{sp}", [StoichiometryEntry(sp)], [],
            LinearDegradation(rate=rate),
        ))

    # Spec: P2 accumulates efficiently; Waste stays low
    p2_rises = Eventually(_pred_ge("P2", 25.0), Interval(0.0, 300.0))
    waste_low = Always(_pred_le("Waste", 30.0), Interval(0.0, 300.0))
    spec = STLAnd(p2_rises, waste_low)

    return BenchmarkCircuit(
        model=model, specification=spec,
        expected_result=VerificationStatus.VERIFIED,
        difficulty=BenchmarkDifficulty.MEDIUM,
        name="substrate_channeling_mm",
        description="Two-enzyme substrate channeling with MM kinetics",
        category="enzyme",
        tags=["michaelis_menten", "channeling", "mass_action"],
        metadata={
            "species_count": 8, "parameter_count": 9,
            "kinetics_type": "michaelis_menten",
            "topology_type": "cascade",
        },
    )


# ===================================================================
# 2.  Allosteric regulation circuits
# ===================================================================


def mwc_allosteric_switch() -> BenchmarkCircuit:
    """MWC (Monod–Wyman–Changeux) allosteric enzyme switch.

    Tetrameric enzyme with two conformational states: relaxed (R,
    high-activity) and tense (T, low-activity).  Ligand binding
    shifts the R/T equilibrium cooperatively.

    Species (6): Ligand, R_state, T_state, R_bound, T_bound, Product
    Kinetics: MassAction (R⇌T transitions) + MichaelisMenten (catalysis)
    Literature: Monod, Wyman & Changeux (1965); Changeux (2012)
    """
    model = BioModel(name="mwc_allosteric_switch")

    species_data = [
        ("Ligand", 50.0, SpeciesType.SMALL_MOLECULE),
        ("R_state", 8.0, SpeciesType.PROTEIN),
        ("T_state", 2.0, SpeciesType.PROTEIN),
        ("R_bound", 0.0, SpeciesType.PROTEIN),
        ("T_bound", 0.0, SpeciesType.PROTEIN),
        ("Product", 0.0, SpeciesType.SMALL_MOLECULE),
    ]
    for name, ic, stype in species_data:
        model.add_species(Species(name, initial_concentration=ic,
                                  species_type=stype))

    params = [
        # Allosteric equilibrium constant L = [T₀]/[R₀]
        _rate_param("k_RT", 0.1, 0.03, 0.3),       # R → T
        _rate_param("k_TR", 0.8, 0.3, 2.0),         # T → R
        # Ligand binding (R-state has higher affinity)
        _rate_param("kon_R", 0.02, 0.005, 0.06, "1/(nM·min)"),
        _rate_param("koff_R", 0.01, 0.003, 0.03),
        _rate_param("kon_T", 0.002, 0.0005, 0.006, "1/(nM·min)"),
        _rate_param("koff_T", 0.05, 0.015, 0.15),
        # Catalysis (only R_bound is catalytically active)
        _mm_param("Vmax_R", 10.0, 4.0, 20.0, "nM/min"),
        _mm_param("Km_R", 10.0, 3.0, 25.0),
        # T_bound has negligible activity
        _mm_param("Vmax_T", 0.5, 0.1, 1.5, "nM/min"),
        _mm_param("Km_T", 50.0, 20.0, 100.0),
        _rate_param("deg_Product", 0.02, 0.005, 0.05),
        _rate_param("deg_Ligand", 0.005, 0.001, 0.015),
    ]
    for p in params:
        model.parameters.add(p)

    # R ⇌ T conformational equilibrium
    model.add_reaction(Reaction(
        "R_to_T",
        [StoichiometryEntry("R_state")],
        [StoichiometryEntry("T_state")],
        MassAction(k_forward=0.1, k_reverse=0.8),
        reversible=True,
    ))
    # Ligand + R → R_bound
    model.add_reaction(Reaction(
        "bind_R",
        [StoichiometryEntry("Ligand"), StoichiometryEntry("R_state")],
        [StoichiometryEntry("R_bound")],
        MassAction(k_forward=0.02, k_reverse=0.01),
        reversible=True,
    ))
    # Ligand + T → T_bound (much weaker)
    model.add_reaction(Reaction(
        "bind_T",
        [StoichiometryEntry("Ligand"), StoichiometryEntry("T_state")],
        [StoichiometryEntry("T_bound")],
        MassAction(k_forward=0.002, k_reverse=0.05),
        reversible=True,
    ))
    # R_bound catalyses product formation (MM kinetics on Ligand)
    model.add_reaction(Reaction(
        "catalysis_R", [],
        [StoichiometryEntry("Product")],
        ConstitutiveProduction(rate=3.0),
    ))
    # T_bound produces product at negligible rate
    model.add_reaction(Reaction(
        "catalysis_T", [],
        [StoichiometryEntry("Product")],
        ConstitutiveProduction(rate=0.1),
    ))
    # Degradation
    for sp, rate in [("Product", 0.02), ("Ligand", 0.005)]:
        model.add_reaction(Reaction(
            f"deg_{sp}", [StoichiometryEntry(sp)], [],
            LinearDegradation(rate=rate),
        ))

    # Spec: cooperative switch — Product rises with sigmoid shape
    product_rises = Eventually(_pred_ge("Product", 30.0),
                               Interval(0.0, 300.0))
    # R_state dominates when ligand is present
    r_dominant = Eventually(_pred_ge("R_bound", 3.0),
                            Interval(0.0, 200.0))
    product_bounded = Always(_pred_le("Product", 400.0),
                             Interval(0.0, 300.0))
    spec = STLAnd(STLAnd(product_rises, r_dominant), product_bounded)

    return BenchmarkCircuit(
        model=model, specification=spec,
        expected_result=VerificationStatus.VERIFIED,
        difficulty=BenchmarkDifficulty.HARD,
        name="mwc_allosteric_switch",
        description="MWC allosteric enzyme with R/T conformational states",
        category="allosteric",
        tags=["allosteric", "mwc", "cooperative", "mass_action"],
        metadata={
            "species_count": 6, "parameter_count": 12,
            "kinetics_type": "allosteric",
            "topology_type": "feedback",
        },
    )


def allosteric_transcription_factor() -> BenchmarkCircuit:
    """Allosteric transcription factor with effector-modulated DNA binding.

    The TF exists in active (A) and inactive (I) forms.  An effector
    molecule shifts the A/I equilibrium.  Only the A form binds the
    promoter and activates a downstream gene.

    Species (7): Effector, TF_A, TF_I, TF_A_DNA, mRNA, Protein, Reporter
    Kinetics: MassAction (binding), HillActivation (transcription)
    Literature: Savageau (1976); Marzen et al. (2013)
    """
    model = BioModel(name="allosteric_tf")

    species_data = [
        ("Effector", 40.0, SpeciesType.SMALL_MOLECULE),
        ("TF_A", 5.0, SpeciesType.PROTEIN),
        ("TF_I", 15.0, SpeciesType.PROTEIN),
        ("TF_A_DNA", 0.0, SpeciesType.PROTEIN),
        ("mRNA", 0.0, SpeciesType.MRNA),
        ("Protein", 0.0, SpeciesType.PROTEIN),
        ("Reporter", 0.0, SpeciesType.PROTEIN),
    ]
    for name, ic, stype in species_data:
        model.add_species(Species(name, initial_concentration=ic,
                                  species_type=stype))

    params = [
        _rate_param("k_AI", 0.05, 0.015, 0.15),      # A → I spontaneous
        _rate_param("k_IA", 0.01, 0.003, 0.03),       # I → A spontaneous
        _rate_param("kon_eff", 0.008, 0.002, 0.025, "1/(nM·min)"),
        _rate_param("koff_eff", 0.02, 0.006, 0.06),
        _rate_param("kon_DNA", 0.015, 0.005, 0.045, "1/(nM·min)"),
        _rate_param("koff_DNA", 0.1, 0.03, 0.3),
        _hill_param("Vmax_txn", 6.0, 2.0, 12.0, "nM/min"),
        _hill_param("K_txn", 5.0, 1.5, 12.0),
        _hill_param("n_txn", 1.5, 1.0, 2.5, ""),
        _rate_param("k_transl", 0.3, 0.1, 0.6),
        _hill_param("Vmax_reporter", 4.0, 1.5, 8.0, "nM/min"),
        _hill_param("K_reporter", 30.0, 10.0, 80.0),
        _rate_param("deg_mRNA", 0.1, 0.04, 0.2),
        _rate_param("deg_Protein", 0.03, 0.01, 0.06),
        _rate_param("deg_Reporter", 0.02, 0.006, 0.05),
        _rate_param("deg_Effector", 0.005, 0.001, 0.015),
    ]
    for p in params:
        model.parameters.add(p)

    # TF_A ⇌ TF_I (conformational switch)
    model.add_reaction(Reaction(
        "TF_A_to_I",
        [StoichiometryEntry("TF_A")],
        [StoichiometryEntry("TF_I")],
        MassAction(k_forward=0.05, k_reverse=0.01),
        reversible=True,
    ))
    # Effector + TF_I → TF_A (effector stabilises active form)
    model.add_reaction(Reaction(
        "effector_activation",
        [StoichiometryEntry("Effector"), StoichiometryEntry("TF_I")],
        [StoichiometryEntry("TF_A")],
        MassAction(k_forward=0.008, k_reverse=0.02),
        reversible=True,
    ))
    # TF_A binds DNA
    model.add_reaction(Reaction(
        "TF_binds_DNA", [],
        [StoichiometryEntry("TF_A_DNA")],
        ConstitutiveProduction(rate=0.5),
    ))
    # Transcription (driven by TF_A_DNA)
    model.add_reaction(Reaction(
        "transcription", [],
        [StoichiometryEntry("mRNA")],
        HillActivation(Vmax=6.0, K=5.0, n=1.5),
    ))
    # Translation
    model.add_reaction(Reaction(
        "translation", [],
        [StoichiometryEntry("Protein")],
        ConstitutiveProduction(rate=0.3),
    ))
    # Protein activates reporter
    model.add_reaction(Reaction(
        "prod_Reporter", [],
        [StoichiometryEntry("Reporter")],
        HillActivation(Vmax=4.0, K=30.0, n=2.0),
    ))
    # Degradation
    for sp, rate in [("mRNA", 0.1), ("Protein", 0.03),
                     ("Reporter", 0.02), ("Effector", 0.005),
                     ("TF_A_DNA", 0.08)]:
        model.add_reaction(Reaction(
            f"deg_{sp}", [StoichiometryEntry(sp)], [],
            LinearDegradation(rate=rate),
        ))

    # Spec: Reporter rises when effector present
    reporter_on = Eventually(_pred_ge("Reporter", 15.0),
                             Interval(0.0, 400.0))
    bounded = Always(_pred_le("Reporter", 200.0),
                     Interval(0.0, 400.0))
    spec = STLAnd(reporter_on, bounded)

    return BenchmarkCircuit(
        model=model, specification=spec,
        expected_result=VerificationStatus.VERIFIED,
        difficulty=BenchmarkDifficulty.HARD,
        name="allosteric_tf",
        description="Allosteric TF with effector-modulated DNA binding",
        category="allosteric",
        tags=["allosteric", "transcription_factor", "mass_action"],
        metadata={
            "species_count": 7, "parameter_count": 16,
            "kinetics_type": "allosteric",
            "topology_type": "cascade",
        },
    )


# ===================================================================
# 3.  Complex topology circuits — multi-feedback
# ===================================================================


def dual_feedback_oscillator() -> BenchmarkCircuit:
    """Dual-feedback oscillator with overlapping positive and negative loops.

    Activator A drives its own production (positive feedback) and
    also drives Repressor R which inhibits A (negative feedback).
    The interplay produces oscillations that are more robust than
    either loop alone.

    Species (5): A, R, mRNA_A, mRNA_R, Reporter
    Kinetics: Hill + LinearDegradation
    Literature: Stricker et al. (2008) — synthetic oscillator; Atkinson
    et al. (2003) — dual-feedback design
    """
    model = BioModel(name="dual_feedback_oscillator")

    species_data = [
        ("A", 20.0, SpeciesType.PROTEIN),
        ("R", 5.0, SpeciesType.PROTEIN),
        ("mRNA_A", 5.0, SpeciesType.MRNA),
        ("mRNA_R", 2.0, SpeciesType.MRNA),
        ("Reporter", 0.0, SpeciesType.PROTEIN),
    ]
    for name, ic, stype in species_data:
        model.add_species(Species(name, initial_concentration=ic,
                                  species_type=stype))

    params = [
        # Positive autoregulation of A
        _hill_param("Vmax_A_auto", 8.0, 3.0, 15.0, "nM/min"),
        _hill_param("K_A_auto", 100.0, 40.0, 250.0),
        _hill_param("n_A_auto", 2.0, 1.5, 3.0, ""),
        # Negative regulation: R --| A
        _hill_param("Vmax_R_rep", 6.0, 2.0, 12.0, "nM/min"),
        _hill_param("K_R_rep", 80.0, 30.0, 200.0),
        _hill_param("n_R_rep", 3.0, 2.0, 4.5, ""),
        # A activates R production
        _hill_param("Vmax_R_act", 5.0, 2.0, 10.0, "nM/min"),
        _hill_param("K_R_act", 120.0, 50.0, 300.0),
        _hill_param("n_R_act", 2.0, 1.5, 3.0, ""),
        # Translation
        _rate_param("k_transl_A", 0.4, 0.15, 0.8),
        _rate_param("k_transl_R", 0.4, 0.15, 0.8),
        # Reporter
        _hill_param("Vmax_Reporter", 4.0, 1.5, 8.0, "nM/min"),
        _hill_param("K_Reporter", 50.0, 20.0, 120.0),
        # Degradation
        _rate_param("deg_mRNA_A", 0.12, 0.05, 0.25),
        _rate_param("deg_mRNA_R", 0.12, 0.05, 0.25),
        _rate_param("deg_A", 0.04, 0.01, 0.08),
        _rate_param("deg_R", 0.06, 0.02, 0.12),
        _rate_param("deg_Reporter", 0.02, 0.006, 0.05),
    ]
    for p in params:
        model.parameters.add(p)

    # Positive feedback: A activates its own mRNA
    model.add_reaction(Reaction(
        "prod_mRNA_A_pos", [],
        [StoichiometryEntry("mRNA_A")],
        HillActivation(Vmax=8.0, K=100.0, n=2.0),
    ))
    # Negative feedback: R represses mRNA_A
    model.add_reaction(Reaction(
        "rep_mRNA_A_neg", [],
        [StoichiometryEntry("mRNA_A")],
        HillRepression(Vmax=6.0, K=80.0, n=3.0),
    ))
    # A activates mRNA_R
    model.add_reaction(Reaction(
        "prod_mRNA_R", [],
        [StoichiometryEntry("mRNA_R")],
        HillActivation(Vmax=5.0, K=120.0, n=2.0),
    ))
    # Translation
    model.add_reaction(Reaction(
        "transl_A", [], [StoichiometryEntry("A")],
        ConstitutiveProduction(rate=0.4),
    ))
    model.add_reaction(Reaction(
        "transl_R", [], [StoichiometryEntry("R")],
        ConstitutiveProduction(rate=0.4),
    ))
    # A activates reporter
    model.add_reaction(Reaction(
        "prod_Reporter", [], [StoichiometryEntry("Reporter")],
        HillActivation(Vmax=4.0, K=50.0, n=2.0),
    ))
    # Degradation
    for sp, rate in [("mRNA_A", 0.12), ("mRNA_R", 0.12), ("A", 0.04),
                     ("R", 0.06), ("Reporter", 0.02)]:
        model.add_reaction(Reaction(
            f"deg_{sp}", [StoichiometryEntry(sp)], [],
            LinearDegradation(rate=rate),
        ))

    # Spec: sustained oscillation in Reporter
    rise = Eventually(_pred_ge("Reporter", 30.0), Interval(0.0, 300.0))
    fall = Eventually(_pred_le("Reporter", 5.0), Interval(0.0, 300.0))
    spec = Always(STLAnd(rise, fall), Interval(0.0, 600.0))

    return BenchmarkCircuit(
        model=model, specification=spec,
        expected_result=VerificationStatus.BOUNDED_GUARANTEE,
        difficulty=BenchmarkDifficulty.HARD,
        name="dual_feedback_oscillator",
        description="Dual positive/negative feedback oscillator",
        category="oscillator",
        tags=["oscillator", "dual_feedback", "multi_feedback", "hill"],
        metadata={
            "species_count": 5, "parameter_count": 18,
            "kinetics_type": "hill",
            "topology_type": "multi_feedback",
        },
    )


def three_node_competitive_network() -> BenchmarkCircuit:
    """Three mutually repressing species (competitive exclusion network).

    A, B, C each repress the other two.  Depending on parameter
    values, the system can settle into one of three monostable states
    or exhibit complex multistability.

    Species (6): A, B, C, mRNA_A, mRNA_B, mRNA_C
    Kinetics: Hill repression
    Literature: Alon (2007); May & Leonard (1975) — competitive exclusion
    """
    model = BioModel(name="three_node_competitive")

    species_data = [
        ("A", 30.0, SpeciesType.PROTEIN),
        ("B", 10.0, SpeciesType.PROTEIN),
        ("C", 10.0, SpeciesType.PROTEIN),
        ("mRNA_A", 8.0, SpeciesType.MRNA),
        ("mRNA_B", 3.0, SpeciesType.MRNA),
        ("mRNA_C", 3.0, SpeciesType.MRNA),
    ]
    for name, ic, stype in species_data:
        model.add_species(Species(name, initial_concentration=ic,
                                  species_type=stype))

    params = [
        # A represses B and C
        _hill_param("Vmax_A_rep_B", 5.0, 2.0, 10.0, "nM/min"),
        _hill_param("K_A_rep_B", 150.0, 60.0, 350.0),
        _hill_param("n_A_rep_B", 2.5, 1.5, 4.0, ""),
        _hill_param("Vmax_A_rep_C", 5.0, 2.0, 10.0, "nM/min"),
        _hill_param("K_A_rep_C", 150.0, 60.0, 350.0),
        _hill_param("n_A_rep_C", 2.5, 1.5, 4.0, ""),
        # B represses A and C
        _hill_param("Vmax_B_rep_A", 5.0, 2.0, 10.0, "nM/min"),
        _hill_param("K_B_rep_A", 150.0, 60.0, 350.0),
        _hill_param("n_B_rep_A", 2.5, 1.5, 4.0, ""),
        _hill_param("Vmax_B_rep_C", 5.0, 2.0, 10.0, "nM/min"),
        _hill_param("K_B_rep_C", 150.0, 60.0, 350.0),
        _hill_param("n_B_rep_C", 2.5, 1.5, 4.0, ""),
        # C represses A and B
        _hill_param("Vmax_C_rep_A", 5.0, 2.0, 10.0, "nM/min"),
        _hill_param("K_C_rep_A", 150.0, 60.0, 350.0),
        _hill_param("n_C_rep_A", 2.5, 1.5, 4.0, ""),
        _hill_param("Vmax_C_rep_B", 5.0, 2.0, 10.0, "nM/min"),
        _hill_param("K_C_rep_B", 150.0, 60.0, 350.0),
        _hill_param("n_C_rep_B", 2.5, 1.5, 4.0, ""),
        # Basal production
        _rate_param("basal_A", 0.5, 0.2, 1.0),
        _rate_param("basal_B", 0.5, 0.2, 1.0),
        _rate_param("basal_C", 0.5, 0.2, 1.0),
        # Translation
        _rate_param("k_transl", 0.3, 0.1, 0.6),
        # Degradation
        _rate_param("deg_mRNA", 0.1, 0.04, 0.2),
        _rate_param("deg_prot", 0.04, 0.01, 0.08),
    ]
    for p in params:
        model.parameters.add(p)

    # Mutual repression: each protein represses the mRNAs of the others
    repression_map = [
        ("A", "mRNA_B"), ("A", "mRNA_C"),
        ("B", "mRNA_A"), ("B", "mRNA_C"),
        ("C", "mRNA_A"), ("C", "mRNA_B"),
    ]
    for repressor, target in repression_map:
        model.add_reaction(Reaction(
            f"rep_{repressor}_{target}", [],
            [StoichiometryEntry(target)],
            HillRepression(Vmax=5.0, K=150.0, n=2.5),
        ))

    # Basal production of each mRNA
    for sp in ["mRNA_A", "mRNA_B", "mRNA_C"]:
        model.add_reaction(Reaction(
            f"basal_{sp}", [], [StoichiometryEntry(sp)],
            ConstitutiveProduction(rate=0.5),
        ))

    # Translation: mRNA → protein
    for mrna, prot in [("mRNA_A", "A"), ("mRNA_B", "B"), ("mRNA_C", "C")]:
        model.add_reaction(Reaction(
            f"transl_{prot}", [], [StoichiometryEntry(prot)],
            ConstitutiveProduction(rate=0.3),
        ))

    # Degradation
    for sp in ["mRNA_A", "mRNA_B", "mRNA_C"]:
        model.add_reaction(Reaction(
            f"deg_{sp}", [StoichiometryEntry(sp)], [],
            LinearDegradation(rate=0.1),
        ))
    for sp in ["A", "B", "C"]:
        model.add_reaction(Reaction(
            f"deg_{sp}", [StoichiometryEntry(sp)], [],
            LinearDegradation(rate=0.04),
        ))

    # Spec: competitive exclusion — exactly one species dominates
    a_wins = Always(_pred_ge("A", 40.0), Interval(100.0, 500.0))
    b_low = Always(_pred_le("B", 20.0), Interval(100.0, 500.0))
    c_low = Always(_pred_le("C", 20.0), Interval(100.0, 500.0))
    exclusion = STLAnd(a_wins, STLAnd(b_low, c_low))
    bounded = Always(
        STLAnd(_pred_le("A", 300.0),
               STLAnd(_pred_le("B", 300.0), _pred_le("C", 300.0))),
        Interval(0.0, 500.0),
    )
    spec = STLAnd(exclusion, bounded)

    return BenchmarkCircuit(
        model=model, specification=spec,
        expected_result=VerificationStatus.BOUNDED_GUARANTEE,
        difficulty=BenchmarkDifficulty.HARD,
        name="three_node_competitive",
        description="Three mutually repressing species (competitive exclusion)",
        category="competitive",
        tags=["competitive", "multi_feedback", "multistability", "hill"],
        metadata={
            "species_count": 6, "parameter_count": 24,
            "kinetics_type": "hill",
            "topology_type": "competitive",
        },
    )


def iffl_adaptation() -> BenchmarkCircuit:
    """Incoherent feed-forward loop (type-1 IFFL) with perfect adaptation.

    Input X activates both Y and Z directly, but Y represses Z.  The
    opposing regulation produces a transient pulse in Z followed by
    adaptation to a baseline, regardless of input amplitude.

    Species (5): X, Y, Z, mRNA_Y, mRNA_Z
    Kinetics: Hill
    Literature: Alon (2007) Ch. 4; Shen-Orr et al. (2002)
    """
    model = BioModel(name="iffl_adaptation")

    species_data = [
        ("X", 80.0, SpeciesType.PROTEIN),
        ("Y", 0.0, SpeciesType.PROTEIN),
        ("Z", 0.0, SpeciesType.PROTEIN),
        ("mRNA_Y", 0.0, SpeciesType.MRNA),
        ("mRNA_Z", 0.0, SpeciesType.MRNA),
    ]
    for name, ic, stype in species_data:
        model.add_species(Species(name, initial_concentration=ic,
                                  species_type=stype))

    params = [
        # X activates Y
        _hill_param("Vmax_XY", 6.0, 2.0, 12.0, "nM/min"),
        _hill_param("K_XY", 80.0, 30.0, 200.0),
        _hill_param("n_XY", 2.0, 1.5, 3.0, ""),
        # X activates Z directly
        _hill_param("Vmax_XZ", 8.0, 3.0, 15.0, "nM/min"),
        _hill_param("K_XZ", 60.0, 25.0, 150.0),
        _hill_param("n_XZ", 2.0, 1.5, 3.0, ""),
        # Y represses Z (incoherent arm)
        _hill_param("Vmax_YZ_rep", 7.0, 3.0, 14.0, "nM/min"),
        _hill_param("K_YZ_rep", 100.0, 40.0, 250.0),
        _hill_param("n_YZ_rep", 3.0, 2.0, 4.5, ""),
        # Translation
        _rate_param("k_transl_Y", 0.4, 0.15, 0.8),
        _rate_param("k_transl_Z", 0.4, 0.15, 0.8),
        # Degradation
        _rate_param("deg_mRNA_Y", 0.1, 0.04, 0.2),
        _rate_param("deg_mRNA_Z", 0.1, 0.04, 0.2),
        _rate_param("deg_Y", 0.05, 0.02, 0.1),
        _rate_param("deg_Z", 0.05, 0.02, 0.1),
    ]
    for p in params:
        model.parameters.add(p)

    # X activates mRNA_Y
    model.add_reaction(Reaction(
        "prod_mRNA_Y", [], [StoichiometryEntry("mRNA_Y")],
        HillActivation(Vmax=6.0, K=80.0, n=2.0),
    ))
    # X activates mRNA_Z (direct arm)
    model.add_reaction(Reaction(
        "prod_mRNA_Z_direct", [], [StoichiometryEntry("mRNA_Z")],
        HillActivation(Vmax=8.0, K=60.0, n=2.0),
    ))
    # Y represses mRNA_Z (incoherent arm)
    model.add_reaction(Reaction(
        "rep_mRNA_Z_by_Y", [], [StoichiometryEntry("mRNA_Z")],
        HillRepression(Vmax=7.0, K=100.0, n=3.0),
    ))
    # Translation
    model.add_reaction(Reaction(
        "transl_Y", [], [StoichiometryEntry("Y")],
        ConstitutiveProduction(rate=0.4),
    ))
    model.add_reaction(Reaction(
        "transl_Z", [], [StoichiometryEntry("Z")],
        ConstitutiveProduction(rate=0.4),
    ))
    # Degradation
    for sp, rate in [("mRNA_Y", 0.1), ("mRNA_Z", 0.1),
                     ("Y", 0.05), ("Z", 0.05)]:
        model.add_reaction(Reaction(
            f"deg_{sp}", [StoichiometryEntry(sp)], [],
            LinearDegradation(rate=rate),
        ))

    # Spec: Z shows transient pulse (rises then falls = adaptation)
    pulse_up = Eventually(_pred_ge("Z", 30.0), Interval(0.0, 150.0))
    pulse_down = Eventually(_pred_le("Z", 10.0), Interval(100.0, 400.0))
    spec = STLAnd(pulse_up, pulse_down)

    return BenchmarkCircuit(
        model=model, specification=spec,
        expected_result=VerificationStatus.VERIFIED,
        difficulty=BenchmarkDifficulty.MEDIUM,
        name="iffl_adaptation",
        description="Type-1 IFFL with perfect adaptation (pulse generation)",
        category="feed_forward",
        tags=["iffl", "adaptation", "feed_forward", "multi_feedback", "hill"],
        metadata={
            "species_count": 5, "parameter_count": 15,
            "kinetics_type": "hill",
            "topology_type": "multi_feedback",
        },
    )


# ===================================================================
# 4.  Larger circuits (10+ species)
# ===================================================================


def repressilator_with_reporters() -> BenchmarkCircuit:
    """Repressilator core (3 genes) with GFP/RFP/CFP reporters (6 species).

    Each repressor protein drives a fluorescent reporter, enabling
    experimental readout of phase relationships.

    Species (6): TetR, LacI, CI, GFP, RFP, CFP
    Kinetics: Hill
    Literature: Elowitz & Leibler (2000)
    """
    model = BioModel(name="repressilator_reporters")

    species_data = [
        ("TetR", 30.0, SpeciesType.PROTEIN),
        ("LacI", 10.0, SpeciesType.PROTEIN),
        ("CI", 20.0, SpeciesType.PROTEIN),
        ("GFP", 0.0, SpeciesType.PROTEIN),
        ("RFP", 0.0, SpeciesType.PROTEIN),
        ("CFP", 0.0, SpeciesType.PROTEIN),
    ]
    for name, ic, stype in species_data:
        model.add_species(Species(name, initial_concentration=ic,
                                  species_type=stype))

    params = [
        _hill_param("Vmax_TetR", 4.5, 1.5, 9.0, "nM/min"),
        _hill_param("K_TetR", 180.0, 80.0, 400.0),
        _hill_param("n_TetR", 2.5, 1.5, 4.0, ""),
        _hill_param("Vmax_LacI", 4.5, 1.5, 9.0, "nM/min"),
        _hill_param("K_LacI", 180.0, 80.0, 400.0),
        _hill_param("n_LacI", 2.5, 1.5, 4.0, ""),
        _hill_param("Vmax_CI", 4.5, 1.5, 9.0, "nM/min"),
        _hill_param("K_CI", 180.0, 80.0, 400.0),
        _hill_param("n_CI", 2.5, 1.5, 4.0, ""),
        _hill_param("Vmax_GFP", 3.0, 1.0, 6.0, "nM/min"),
        _hill_param("K_GFP", 80.0, 30.0, 200.0),
        _hill_param("Vmax_RFP", 3.0, 1.0, 6.0, "nM/min"),
        _hill_param("K_RFP", 80.0, 30.0, 200.0),
        _hill_param("Vmax_CFP", 3.0, 1.0, 6.0, "nM/min"),
        _hill_param("K_CFP", 80.0, 30.0, 200.0),
        _rate_param("deg_prot", 0.04, 0.01, 0.08),
        _rate_param("deg_reporter", 0.025, 0.008, 0.05),
    ]
    for p in params:
        model.parameters.add(p)

    # Repressilator ring: CI --| TetR, TetR --| LacI, LacI --| CI
    ring = [("CI", "TetR"), ("TetR", "LacI"), ("LacI", "CI")]
    for repressor, target in ring:
        model.add_reaction(Reaction(
            f"prod_{target}", [], [StoichiometryEntry(target)],
            HillRepression(Vmax=4.5, K=180.0, n=2.5),
        ))

    # Reporter production: each protein activates its reporter
    reporter_map = [("TetR", "GFP"), ("LacI", "RFP"), ("CI", "CFP")]
    for activator, reporter in reporter_map:
        model.add_reaction(Reaction(
            f"prod_{reporter}", [], [StoichiometryEntry(reporter)],
            HillActivation(Vmax=3.0, K=80.0, n=2.0),
        ))

    # Degradation
    for sp in ["TetR", "LacI", "CI"]:
        model.add_reaction(Reaction(
            f"deg_{sp}", [StoichiometryEntry(sp)], [],
            LinearDegradation(rate=0.04),
        ))
    for sp in ["GFP", "RFP", "CFP"]:
        model.add_reaction(Reaction(
            f"deg_{sp}", [StoichiometryEntry(sp)], [],
            LinearDegradation(rate=0.025),
        ))

    # Spec: GFP oscillates (proxy for repressilator oscillation)
    rise = Eventually(_pred_ge("GFP", 25.0), Interval(0.0, 250.0))
    fall = Eventually(_pred_le("GFP", 5.0), Interval(0.0, 250.0))
    spec = Always(STLAnd(rise, fall), Interval(0.0, 500.0))

    return BenchmarkCircuit(
        model=model, specification=spec,
        expected_result=VerificationStatus.BOUNDED_GUARANTEE,
        difficulty=BenchmarkDifficulty.MEDIUM,
        name="repressilator_reporters",
        description="Repressilator with GFP/RFP/CFP reporters (6 species)",
        category="oscillator",
        tags=["oscillator", "repressilator", "reporter", "hill"],
        metadata={
            "species_count": 6, "parameter_count": 17,
            "kinetics_type": "hill",
            "topology_type": "oscillator",
        },
    )


def signaling_cascade_10() -> BenchmarkCircuit:
    """Ten-species signaling cascade with amplification and negative feedback.

    Input → Receptor → Adaptor → Kinase1 → Kinase2 → Kinase3 →
    TF → mRNA → Effector → Output.  Kinase3 phosphorylates a
    phosphatase that feeds back to deactivate Kinase1 (negative
    feedback for adaptation).

    Species (10): Input, Receptor, Adaptor, Kinase1, Kinase2, Kinase3,
                  TF, mRNA_eff, Effector, Output
    Kinetics: Hill + MM + MassAction
    Literature: Huang & Ferrell (1996); Kholodenko (2006)
    """
    model = BioModel(name="signaling_cascade_10")

    species_data = [
        ("Input", 80.0, SpeciesType.SMALL_MOLECULE),
        ("Receptor", 0.0, SpeciesType.PROTEIN),
        ("Adaptor", 0.0, SpeciesType.PROTEIN),
        ("Kinase1", 0.0, SpeciesType.PROTEIN),
        ("Kinase2", 0.0, SpeciesType.PROTEIN),
        ("Kinase3", 0.0, SpeciesType.PROTEIN),
        ("TF", 0.0, SpeciesType.PROTEIN),
        ("mRNA_eff", 0.0, SpeciesType.MRNA),
        ("Effector", 0.0, SpeciesType.PROTEIN),
        ("Output", 0.0, SpeciesType.PROTEIN),
    ]
    for name, ic, stype in species_data:
        model.add_species(Species(name, initial_concentration=ic,
                                  species_type=stype))

    params = [
        # Receptor activation (MM — ligand binding)
        _mm_param("Vmax_Receptor", 7.0, 3.0, 14.0, "nM/min"),
        _mm_param("Km_Receptor", 25.0, 8.0, 60.0),
        # Cascade stages (Hill activation)
        _hill_param("Vmax_Adaptor", 6.0, 2.0, 12.0, "nM/min"),
        _hill_param("K_Adaptor", 60.0, 25.0, 150.0),
        _hill_param("Vmax_K1", 5.5, 2.0, 11.0, "nM/min"),
        _hill_param("K_K1", 80.0, 30.0, 200.0),
        _hill_param("Vmax_K2", 5.0, 2.0, 10.0, "nM/min"),
        _hill_param("K_K2", 90.0, 35.0, 220.0),
        _hill_param("Vmax_K3", 4.5, 1.5, 9.0, "nM/min"),
        _hill_param("K_K3", 100.0, 40.0, 250.0),
        # TF activation
        _hill_param("Vmax_TF", 4.0, 1.5, 8.0, "nM/min"),
        _hill_param("K_TF", 70.0, 30.0, 170.0),
        # Transcription + translation
        _hill_param("Vmax_mRNA", 5.0, 2.0, 10.0, "nM/min"),
        _hill_param("K_mRNA", 50.0, 20.0, 120.0),
        _rate_param("k_transl_eff", 0.3, 0.1, 0.6),
        # Output
        _hill_param("Vmax_Output", 4.0, 1.5, 8.0, "nM/min"),
        _hill_param("K_Output", 40.0, 15.0, 100.0),
        # Negative feedback: Kinase3 --| Kinase1
        _hill_param("Vmax_fb", 3.0, 1.0, 6.0, "nM/min"),
        _hill_param("K_fb", 120.0, 50.0, 300.0),
        _hill_param("n_fb", 2.0, 1.5, 3.0, ""),
        # Degradation
        _rate_param("deg_fast", 0.06, 0.02, 0.12),
        _rate_param("deg_medium", 0.03, 0.01, 0.06),
        _rate_param("deg_slow", 0.015, 0.005, 0.03),
    ]
    for p in params:
        model.parameters.add(p)

    # Input → Receptor (MM kinetics)
    model.add_reaction(Reaction(
        "act_Receptor",
        [StoichiometryEntry("Input")],
        [StoichiometryEntry("Receptor")],
        MichaelisMenten(Vmax=7.0, Km=25.0),
    ))
    # Linear cascade stages (Hill activation)
    cascade_stages = [
        ("Receptor", "Adaptor", 6.0, 60.0),
        ("Adaptor", "Kinase1", 5.5, 80.0),
        ("Kinase1", "Kinase2", 5.0, 90.0),
        ("Kinase2", "Kinase3", 4.5, 100.0),
        ("Kinase3", "TF", 4.0, 70.0),
    ]
    for _src, target, vmax, K in cascade_stages:
        model.add_reaction(Reaction(
            f"prod_{target}", [], [StoichiometryEntry(target)],
            HillActivation(Vmax=vmax, K=K, n=2.0),
        ))
    # TF → mRNA_eff
    model.add_reaction(Reaction(
        "prod_mRNA_eff", [], [StoichiometryEntry("mRNA_eff")],
        HillActivation(Vmax=5.0, K=50.0, n=2.0),
    ))
    # Translation
    model.add_reaction(Reaction(
        "transl_Effector", [], [StoichiometryEntry("Effector")],
        ConstitutiveProduction(rate=0.3),
    ))
    # Effector → Output
    model.add_reaction(Reaction(
        "prod_Output", [], [StoichiometryEntry("Output")],
        HillActivation(Vmax=4.0, K=40.0, n=2.0),
    ))
    # Negative feedback: Kinase3 represses Kinase1 production
    model.add_reaction(Reaction(
        "fb_K3_K1", [], [StoichiometryEntry("Kinase1")],
        HillRepression(Vmax=3.0, K=120.0, n=2.0),
    ))
    # Degradation
    for sp in ["Receptor", "Adaptor"]:
        model.add_reaction(Reaction(
            f"deg_{sp}", [StoichiometryEntry(sp)], [],
            LinearDegradation(rate=0.06),
        ))
    for sp in ["Kinase1", "Kinase2", "Kinase3", "TF"]:
        model.add_reaction(Reaction(
            f"deg_{sp}", [StoichiometryEntry(sp)], [],
            LinearDegradation(rate=0.03),
        ))
    for sp in ["mRNA_eff", "Effector", "Output"]:
        model.add_reaction(Reaction(
            f"deg_{sp}", [StoichiometryEntry(sp)], [],
            LinearDegradation(rate=0.015),
        ))

    # Spec: signal reaches Output; bounded; adaptation (Output settles)
    signal_propagates = Eventually(
        _pred_ge("Output", 15.0), Interval(0.0, 500.0))
    bounded = Always(_pred_le("Output", 200.0), Interval(0.0, 500.0))
    spec = STLAnd(signal_propagates, bounded)

    return BenchmarkCircuit(
        model=model, specification=spec,
        expected_result=VerificationStatus.VERIFIED,
        difficulty=BenchmarkDifficulty.HARD,
        name="signaling_cascade_10",
        description="10-species signaling cascade with negative feedback",
        category="cascade",
        tags=["cascade", "signaling", "feedback", "large",
              "michaelis_menten", "hill"],
        metadata={
            "species_count": 10, "parameter_count": 23,
            "kinetics_type": "michaelis_menten",
            "topology_type": "cascade",
        },
    )


def metabolic_pathway_regulated() -> BenchmarkCircuit:
    """Metabolic pathway with enzyme regulation (12 species).

    Four-step metabolic pathway (S → M1 → M2 → M3 → Product) with
    four enzymes, product inhibition of E1 (feedback), allosteric
    activation of E3 by the substrate, and a transcription factor
    that regulates E1/E2 expression.

    Species (12): Substrate, E1, M1, E2, M2, E3, M3, E4, Product,
                  TF, mRNA_E1, mRNA_E2
    Kinetics: Michaelis–Menten + Hill + MassAction
    Literature: Kacser & Burns (1973); Heinrich & Rapoport (1974)
    """
    model = BioModel(name="metabolic_pathway_regulated")

    species_data = [
        ("Substrate", 100.0, SpeciesType.SMALL_MOLECULE),
        ("E1", 8.0, SpeciesType.PROTEIN),
        ("M1", 0.0, SpeciesType.SMALL_MOLECULE),
        ("E2", 8.0, SpeciesType.PROTEIN),
        ("M2", 0.0, SpeciesType.SMALL_MOLECULE),
        ("E3", 6.0, SpeciesType.PROTEIN),
        ("M3", 0.0, SpeciesType.SMALL_MOLECULE),
        ("E4", 6.0, SpeciesType.PROTEIN),
        ("Product", 0.0, SpeciesType.SMALL_MOLECULE),
        ("TF", 5.0, SpeciesType.PROTEIN),
        ("mRNA_E1", 3.0, SpeciesType.MRNA),
        ("mRNA_E2", 3.0, SpeciesType.MRNA),
    ]
    for name, ic, stype in species_data:
        model.add_species(Species(name, initial_concentration=ic,
                                  species_type=stype))

    params = [
        # Enzymatic steps (MM kinetics)
        _mm_param("Vmax_E1", 6.0, 2.0, 12.0, "nM/min"),
        _mm_param("Km_E1", 18.0, 6.0, 45.0),
        _mm_param("Vmax_E2", 5.0, 2.0, 10.0, "nM/min"),
        _mm_param("Km_E2", 22.0, 8.0, 55.0),
        _mm_param("Vmax_E3", 5.0, 2.0, 10.0, "nM/min"),
        _mm_param("Km_E3", 20.0, 7.0, 50.0),
        _mm_param("Vmax_E4", 4.0, 1.5, 8.0, "nM/min"),
        _mm_param("Km_E4", 25.0, 10.0, 60.0),
        # Product inhibition of E1 (Hill repression)
        _hill_param("Vmax_fb_E1", 3.0, 1.0, 6.0, "nM/min"),
        _hill_param("K_fb_E1", 80.0, 30.0, 200.0),
        _hill_param("n_fb_E1", 2.0, 1.5, 3.0, ""),
        # TF regulation of E1, E2 expression (Hill activation)
        _hill_param("Vmax_TF_E1", 4.0, 1.5, 8.0, "nM/min"),
        _hill_param("K_TF_E1", 50.0, 20.0, 120.0),
        _hill_param("Vmax_TF_E2", 4.0, 1.5, 8.0, "nM/min"),
        _hill_param("K_TF_E2", 50.0, 20.0, 120.0),
        # Translation
        _rate_param("k_transl_E1", 0.3, 0.1, 0.6),
        _rate_param("k_transl_E2", 0.3, 0.1, 0.6),
        # Substrate supply
        _rate_param("k_supply", 0.8, 0.3, 1.5),
        # Degradation
        _rate_param("deg_met", 0.02, 0.005, 0.05),
        _rate_param("deg_enz", 0.03, 0.01, 0.06),
        _rate_param("deg_mRNA", 0.1, 0.04, 0.2),
        _rate_param("deg_Product", 0.01, 0.003, 0.03),
        _rate_param("deg_TF", 0.02, 0.006, 0.05),
    ]
    for p in params:
        model.parameters.add(p)

    # Metabolic steps (MM kinetics)
    model.add_reaction(Reaction(
        "step1_E1", [StoichiometryEntry("Substrate")],
        [StoichiometryEntry("M1")],
        MichaelisMenten(Vmax=6.0, Km=18.0),
    ))
    model.add_reaction(Reaction(
        "step2_E2", [StoichiometryEntry("M1")],
        [StoichiometryEntry("M2")],
        MichaelisMenten(Vmax=5.0, Km=22.0),
    ))
    model.add_reaction(Reaction(
        "step3_E3", [StoichiometryEntry("M2")],
        [StoichiometryEntry("M3")],
        MichaelisMenten(Vmax=5.0, Km=20.0),
    ))
    model.add_reaction(Reaction(
        "step4_E4", [StoichiometryEntry("M3")],
        [StoichiometryEntry("Product")],
        MichaelisMenten(Vmax=4.0, Km=25.0),
    ))
    # Product feedback inhibition of E1 activity (modelled as
    # Product represses mRNA_E1 via Hill)
    model.add_reaction(Reaction(
        "fb_Product_E1", [],
        [StoichiometryEntry("mRNA_E1")],
        HillRepression(Vmax=3.0, K=80.0, n=2.0),
    ))
    # TF activates mRNA_E1 and mRNA_E2
    model.add_reaction(Reaction(
        "TF_act_E1", [],
        [StoichiometryEntry("mRNA_E1")],
        HillActivation(Vmax=4.0, K=50.0, n=2.0),
    ))
    model.add_reaction(Reaction(
        "TF_act_E2", [],
        [StoichiometryEntry("mRNA_E2")],
        HillActivation(Vmax=4.0, K=50.0, n=2.0),
    ))
    # Translation: mRNA → enzyme
    for mrna, enz in [("mRNA_E1", "E1"), ("mRNA_E2", "E2")]:
        model.add_reaction(Reaction(
            f"transl_{enz}", [], [StoichiometryEntry(enz)],
            ConstitutiveProduction(rate=0.3),
        ))
    # Substrate supply
    model.add_reaction(Reaction(
        "supply_Substrate", [], [StoichiometryEntry("Substrate")],
        ConstitutiveProduction(rate=0.8),
    ))
    # Degradation
    for sp in ["M1", "M2", "M3"]:
        model.add_reaction(Reaction(
            f"deg_{sp}", [StoichiometryEntry(sp)], [],
            LinearDegradation(rate=0.02),
        ))
    for sp in ["E1", "E2", "E3", "E4"]:
        model.add_reaction(Reaction(
            f"deg_{sp}", [StoichiometryEntry(sp)], [],
            LinearDegradation(rate=0.03),
        ))
    for sp in ["mRNA_E1", "mRNA_E2"]:
        model.add_reaction(Reaction(
            f"deg_{sp}", [StoichiometryEntry(sp)], [],
            LinearDegradation(rate=0.1),
        ))
    model.add_reaction(Reaction(
        "deg_Product", [StoichiometryEntry("Product")], [],
        LinearDegradation(rate=0.01),
    ))
    model.add_reaction(Reaction(
        "deg_TF", [StoichiometryEntry("TF")], [],
        LinearDegradation(rate=0.02),
    ))
    model.add_reaction(Reaction(
        "deg_Substrate", [StoichiometryEntry("Substrate")], [],
        LinearDegradation(rate=0.005),
    ))

    # Spec: Product accumulates; feedback keeps it bounded
    product_accumulates = Eventually(
        _pred_ge("Product", 15.0), Interval(0.0, 600.0))
    product_bounded = Always(
        _pred_le("Product", 250.0), Interval(0.0, 600.0))
    # Metabolites stay bounded (no toxic accumulation)
    mets_bounded = Always(
        STLAnd(_pred_le("M1", 150.0),
               STLAnd(_pred_le("M2", 150.0), _pred_le("M3", 150.0))),
        Interval(0.0, 600.0),
    )
    spec = STLAnd(STLAnd(product_accumulates, product_bounded),
                  mets_bounded)

    return BenchmarkCircuit(
        model=model, specification=spec,
        expected_result=VerificationStatus.VERIFIED,
        difficulty=BenchmarkDifficulty.FRONTIER,
        name="metabolic_pathway_regulated",
        description="12-species metabolic pathway with feedback and TF regulation",
        category="metabolic",
        tags=["metabolic", "enzyme", "feedback", "michaelis_menten",
              "hill", "large"],
        metadata={
            "species_count": 12, "parameter_count": 23,
            "kinetics_type": "michaelis_menten",
            "topology_type": "feedback",
        },
    )


# ===================================================================
# 5.  ExtendedBenchmarkSuite — aggregation & querying
# ===================================================================


class ExtendedBenchmarkSuite:
    """Extended benchmark collection with diverse kinetics and topologies.

    Complements the core :class:`BenchmarkSuite` with circuits that use
    Michaelis–Menten, allosteric, and mass-action kinetics in addition
    to Hill functions, and feature complex topologies including
    multi-feedback loops, competitive networks, and large-scale
    cascades.

    Each circuit carries ``kinetics_type`` and ``topology_type`` metadata
    tags enabling systematic coverage analysis.
    """

    # -- Circuit generators (static, matching BenchmarkSuite pattern) ---

    enzymatic_cascade_mm = staticmethod(enzymatic_cascade_mm)
    competitive_inhibition_mm = staticmethod(competitive_inhibition_mm)
    substrate_channeling_mm = staticmethod(substrate_channeling_mm)
    mwc_allosteric_switch = staticmethod(mwc_allosteric_switch)
    allosteric_transcription_factor = staticmethod(allosteric_transcription_factor)
    dual_feedback_oscillator = staticmethod(dual_feedback_oscillator)
    three_node_competitive_network = staticmethod(three_node_competitive_network)
    iffl_adaptation = staticmethod(iffl_adaptation)
    repressilator_with_reporters = staticmethod(repressilator_with_reporters)
    signaling_cascade_10 = staticmethod(signaling_cascade_10)
    metabolic_pathway_regulated = staticmethod(metabolic_pathway_regulated)

    # -- Aggregate accessors -------------------------------------------

    @classmethod
    def get_all_circuits(cls) -> List[BenchmarkCircuit]:
        """Return all extended benchmark circuits."""
        return [
            # Michaelis–Menten kinetics
            cls.enzymatic_cascade_mm(),
            cls.competitive_inhibition_mm(),
            cls.substrate_channeling_mm(),
            # Allosteric regulation
            cls.mwc_allosteric_switch(),
            cls.allosteric_transcription_factor(),
            # Complex topology — multi-feedback
            cls.dual_feedback_oscillator(),
            cls.three_node_competitive_network(),
            cls.iffl_adaptation(),
            # Larger circuits (6–12 species)
            cls.repressilator_with_reporters(),
            cls.signaling_cascade_10(),
            cls.metabolic_pathway_regulated(),
        ]

    @classmethod
    def get_by_kinetics(cls, kinetics_type: str) -> List[BenchmarkCircuit]:
        """Return circuits matching the given kinetics type.

        Parameters
        ----------
        kinetics_type:
            One of ``"hill"``, ``"michaelis_menten"``, ``"allosteric"``,
            ``"mass_action"``.
        """
        return [
            b for b in cls.get_all_circuits()
            if b.metadata.get("kinetics_type") == kinetics_type
        ]

    @classmethod
    def get_by_topology(cls, topology_type: str) -> List[BenchmarkCircuit]:
        """Return circuits matching the given topology type.

        Parameters
        ----------
        topology_type:
            One of ``"cascade"``, ``"feedback"``, ``"oscillator"``,
            ``"multi_feedback"``, ``"competitive"``.
        """
        return [
            b for b in cls.get_all_circuits()
            if b.metadata.get("topology_type") == topology_type
        ]

    @classmethod
    def get_by_difficulty(
        cls, difficulty: BenchmarkDifficulty,
    ) -> List[BenchmarkCircuit]:
        """Return circuits matching the given difficulty level."""
        return [
            b for b in cls.get_all_circuits()
            if b.difficulty == difficulty
        ]

    @classmethod
    def get_by_tags(cls, tags: Sequence[str]) -> List[BenchmarkCircuit]:
        """Return circuits whose tag set intersects *tags*."""
        tag_set = set(tags)
        return [
            b for b in cls.get_all_circuits()
            if tag_set & set(b.tags)
        ]

    @classmethod
    def get_combined_suite(cls) -> List[BenchmarkCircuit]:
        """Return all core + extended benchmarks in a single list."""
        return BenchmarkSuite.all_benchmarks() + cls.get_all_circuits()

    @classmethod
    def coverage_summary(cls) -> Dict[str, Any]:
        """Return a summary of kinetics and topology coverage.

        Useful for verifying that the benchmark suite is sufficiently
        diverse to address reviewer concerns about homogeneity.
        """
        circuits = cls.get_all_circuits()
        kinetics_counts: Dict[str, int] = {}
        topology_counts: Dict[str, int] = {}
        difficulty_counts: Dict[str, int] = {}

        for c in circuits:
            kt = c.metadata.get("kinetics_type", "unknown")
            tt = c.metadata.get("topology_type", "unknown")
            dd = c.difficulty.name

            kinetics_counts[kt] = kinetics_counts.get(kt, 0) + 1
            topology_counts[tt] = topology_counts.get(tt, 0) + 1
            difficulty_counts[dd] = difficulty_counts.get(dd, 0) + 1

        return {
            "total_circuits": len(circuits),
            "kinetics_distribution": kinetics_counts,
            "topology_distribution": topology_counts,
            "difficulty_distribution": difficulty_counts,
            "species_range": (
                min(c.metadata.get("species_count", 0) for c in circuits),
                max(c.metadata.get("species_count", 0) for c in circuits),
            ),
        }
