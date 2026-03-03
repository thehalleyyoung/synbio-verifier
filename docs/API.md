# BioProver API Reference

API documentation for the BioProver verification and repair pipeline,
covering only implemented classes and functions.

---

## Table of Contents

1. [Top-Level API (`bioprover`)](#top-level-api)
2. [Models (`bioprover.models`)](#models)
3. [Temporal Logic (`bioprover.temporal`)](#temporal)
4. [CEGAR Engine (`bioprover.cegar`)](#cegar)
5. [Repair & Synthesis (`bioprover.repair`)](#repair)
6. [Soundness (`bioprover.soundness`)](#soundness)
7. [Solver (`bioprover.solver`)](#solver)
   - [Proof Certificates (`bioprover.solver.proof_certificate`)](#proof-certificates)
8. [SMT (`bioprover.smt`)](#smt)
9. [Encoding (`bioprover.encoding`)](#encoding)
10. [Stochastic (`bioprover.stochastic`)](#stochastic)
    - [LNA (`bioprover.stochastic.lna`)](#lna)
11. [AI / ML (`bioprover.ai`)](#ai)
    - [Training Pipeline (`bioprover.ai.training_pipeline`)](#training-pipeline)
    - [Online ML (`bioprover.ai.online_learner`)](#online-ml)
12. [Compositional (`bioprover.compositional`)](#compositional)
    - [Circular AG (`bioprover.compositional.circular_ag`)](#circular-ag)
    - [AG Soundness (`bioprover.compositional.ag_soundness`)](#ag-soundness)
13. [Library (`bioprover.library`)](#library)
14. [Evaluation (`bioprover.evaluation`)](#evaluation)
    - [Ablation (`bioprover.evaluation.ablation`)](#ablation)
    - [Extended Benchmarks (`bioprover.evaluation.extended_benchmarks`)](#extended-benchmarks)
15. [Visualization (`bioprover.visualization`)](#visualization)
16. [CLI (`bioprover.cli`)](#cli)
17. [Certificate Verifier (`bioprover.certificate_verifier`)](#certificate-verifier)
18. [Error Propagation (`bioprover.soundness`)](#error-propagation)
19. [Bio-STL Templates (`bioprover.spec.templates`)](#bio-stl-templates)

---

<a id="top-level-api"></a>
## 1. Top-Level API — `bioprover`

The package root re-exports core classes and provides three convenience
functions covering the primary workflows.

### `verify(model, spec, *, mode="full", timeout=300.0, config=None)`

Verify a biological model against a Bio-STL specification.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `BioModel` | — | The biological model to verify. |
| `spec` | `str` | — | Bio-STL specification string, e.g. `"G[0,100](GFP > 0.5)"`. |
| `mode` | `str` | `"full"` | `"full"`, `"bounded"`, or `"compositional"`. |
| `timeout` | `float` | `300.0` | Maximum wall-clock seconds. |
| `config` | `CEGARConfig \| None` | `None` | Optional CEGAR configuration overrides. |

**Returns:** `VerificationResult`

Internally parses the spec with `BioSTLParser`, constructs a `CEGAREngine`,
and calls `engine.verify(model, formula)`.

```python
from bioprover import BioModel, verify

model = BioModel.from_sbml("toggle_switch.xml")
result = verify(model, spec="G[0,100](GFP > 0.5)", timeout=120.0)
print(result.status)        # VerificationStatus.VERIFIED
print(result.is_verified)   # True
print(result.coverage)      # 0.98
```

---

### `synthesize(model, spec, *, objective="feasibility", timeout=600.0)`

Synthesize parameters for a model satisfying a specification.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `BioModel` | — | Model with uncertain parameters. |
| `spec` | `str` | — | Bio-STL specification string. |
| `objective` | `str` | `"feasibility"` | `"feasibility"`, `"robustness"`, or `"minimal"`. |
| `timeout` | `float` | `600.0` | Maximum wall-clock seconds. |

**Returns:** `SynthesisResult`

Maps the `objective` string to a `SynthesisMode` enum, creates a
`ParameterSynthesizer`, and calls `synthesizer.synthesize(model, formula)`.

```python
from bioprover import BioModel, synthesize

model = BioModel.from_sbml("repressilator.xml")
result = synthesize(model, spec="F[0,50](GFP > 1.0)", objective="robustness")
print(result.feasible)       # True
print(result.parameters)     # {'alpha': 12.3, ...}
print(result.robustness)     # 0.42
```

---

### `repair(model, spec, *, budget=0.5, timeout=600.0)`

Repair model parameters so the specification is satisfied.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `BioModel` | — | Model whose parameters may be adjusted. |
| `spec` | `str` | — | Bio-STL specification string. |
| `budget` | `float` | `0.5` | Maximum fractional perturbation per parameter (0–1). |
| `timeout` | `float` | `600.0` | Maximum wall-clock seconds. |

**Returns:** `RepairResult`

Uses `ParameterSynthesizer` in `MINIMAL` mode with the given perturbation
budget. Returns a `RepairResult` with original/repaired parameters and
robustness.

```python
from bioprover import BioModel, repair

model = BioModel.from_sbml("band_pass.xml")
fix = repair(model, spec="G[10,50](0.3 < GFP < 0.8)", budget=0.2)
if fix.success:
    print(fix.repaired_parameters)
    print(fix.robustness)
```

---

### Re-exported Classes

| Class | Origin Module |
|-------|---------------|
| `BioModel` | `bioprover.models.bio_model` |
| `CEGARConfig` | `bioprover.cegar.cegar_engine` |
| `CEGAREngine` | `bioprover.cegar.cegar_engine` |
| `VerificationResult` | `bioprover.cegar.cegar_engine` |
| `VerificationStatus` | `bioprover.cegar.cegar_engine` |
| `ParameterSynthesizer` | `bioprover.repair.parameter_synthesis` |
| `SynthesisConfig` | `bioprover.repair.parameter_synthesis` |
| `SynthesisMode` | `bioprover.repair.parameter_synthesis` |
| `SynthesisResult` | `bioprover.repair.parameter_synthesis` |
| `RepairResult` | `bioprover.repair.repair_report` |
| `RepairReport` | `bioprover.repair.repair_report` |

---

<a id="models"></a>
## 2. Models — `bioprover.models`

### `BioModel`

Core biological model representation combining species, reactions, and
parameters into an ODE system.

```python
class BioModel:
    def __init__(self, name: str = "unnamed_model") -> None: ...
```

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `from_sbml` | `(filepath: str) -> BioModel` | `BioModel` | Class method. Import from SBML file. |
| `add_species` | `(species: Species) -> None` | `None` | Register a molecular species. |
| `add_reaction` | `(reaction: Reaction) -> None` | `None` | Register a biochemical reaction. |
| `add_parameter` | `(param: Parameter) -> None` | `None` | Register a kinetic parameter. |
| `add_compartment` | `(compartment: Compartment) -> None` | `None` | Register a compartment. |
| `stoichiometry_matrix` | property | `np.ndarray` | Stoichiometry matrix (species × reactions). |
| `rate_vector` | `(concentrations: Dict[str, float]) -> np.ndarray` | `ndarray` | Evaluate reaction rates at given concentrations. |
| `ode_rhs` | `(state: np.ndarray, t: float = 0.0) -> np.ndarray` | `ndarray` | ODE right-hand side evaluation. |
| `ode_rhs_callable` | `() -> Callable[[float, np.ndarray], np.ndarray]` | callable | Scipy-compatible ODE RHS. |
| `jacobian` | `(concentrations: Optional[Dict] = None) -> np.ndarray` | `ndarray` | Jacobian of the ODE RHS. |
| `jacobian_symbolic` | `() -> sympy.Matrix` | `Matrix` | Symbolic Jacobian via SymPy. |
| `steady_state` | `(initial_guess: Optional[np.ndarray] = None, method: str = "fsolve") -> Optional[np.ndarray]` | `ndarray \| None` | Compute steady state. |
| `steady_state_stability` | `(steady_state: np.ndarray) -> str` | `str` | `"stable"`, `"unstable"`, or `"saddle"`. |
| `parameter_sensitivity` | `(parameter_name: str, delta: float = 1e-6) -> np.ndarray` | `ndarray` | Local parameter sensitivity. |
| `simulate` | `(t_span: Tuple[float, float], num_points: int = 100, method: str = "RK45") -> Tuple[np.ndarray, np.ndarray]` | tuple | ODE simulation returning `(times, states)`. |
| `regulatory_network` | `() -> GeneRegulatoryNetwork` | `GRN` | Extract GRN topology. |
| `extract_submodel` | `(species_names: List[str]) -> BioModel` | `BioModel` | Extract a submodel for compositional analysis. |
| `conservation_laws` | `() -> List[Tuple[np.ndarray, float]]` | list | Detect conservation laws from stoichiometry. |
| `check_mass_balance` | `() -> Dict[str, bool]` | dict | Check mass balance per reaction. |
| `validate` | `() -> List[str]` | list | Return list of validation warnings. |
| `initial_state` | `() -> np.ndarray` | `ndarray` | Initial concentration vector. |

**Properties:** `species: List[Species]`, `reactions: List[Reaction]`, `parameters: ParameterSet`

```python
from bioprover import BioModel
from bioprover.models.species import Species
from bioprover.models.reactions import Reaction, HillRepression, LinearDegradation

model = BioModel("my_circuit")
model.add_species(Species("GFP", initial_concentration=1.0))
model.add_reaction(Reaction("deg", reactants={"GFP": 1}, products={},
    kinetic_law=LinearDegradation(rate=0.5)))
t, y = model.simulate((0, 100))
```

---

### `Species`

```python
class Species:
    def __init__(
        self,
        name: str,
        compartment: str = "default",
        initial_concentration: float = 0.0,
        units: str = "nM",
        species_type: SpeciesType = SpeciesType.PROTEIN,
        boundary_condition: BoundaryCondition = BoundaryCondition.FREE,
        metadata: Optional[SpeciesMetadata] = None,
        concentration_bounds: Optional[ConcentrationBounds] = None,
        copy_number: Optional[int] = None,
    ) -> None: ...
```

| Method | Returns | Description |
|--------|---------|-------------|
| `is_in_bounds(concentration)` | `bool` | Check if concentration is within bounds. |
| `validate()` | `List[str]` | Validation warnings. |
| `copy()` | `Species` | Deep copy. |
| `as_sympy_symbol()` | `sympy.Symbol` | SymPy symbol for symbolic computation. |

### `SpeciesType` (Enum)

`PROTEIN`, `MRNA`, `METABOLITE`, `COMPLEX`

---

### `Reaction`

```python
from bioprover.models.reactions import Reaction, StoichiometryEntry as SE

class Reaction:
    def __init__(
        self,
        name: str,
        reactants: List[StoichiometryEntry],  # NOT dicts
        products: List[StoichiometryEntry],    # NOT dicts
        kinetic_law: KineticLaw,
        reversible: bool = False,
        modifiers: List[str] = None,  # for Hill kinetics activator/repressor
    ) -> None: ...

# Example:
rxn = Reaction('production',
    reactants=[],
    products=[SE('gene_u', 1)],
    kinetic_law=HillRepression(Vmax=10.0, K=2.0, n=2),
    modifiers=['gene_v'])  # gene_v is the repressor
```

**Important API notes:**
- `reactants` and `products` take `List[StoichiometryEntry]`, NOT dictionaries.
  Use `StoichiometryEntry("species_name", stoichiometry)` or the alias `SE`.
- Species names MUST NOT be single-letter Bio-STL keywords: `G` (Globally),
  `F` (Finally), `U` (Until). Use multi-character names like `gene_u`, `lacI`.
- For Hill activation/repression with no consumed reactants, use
  `modifiers=["activator_species"]` to specify which species acts as the
  activator or repressor.

### Kinetic Law Classes

All kinetic laws inherit from `KineticLaw` (ABC) and implement
`rate_expression()`, `evaluate()`, `propensity()`, and `parameters`.

| Class | Constructor | Rate Law |
|-------|-------------|----------|
| `MassAction` | `(k_forward: float, k_reverse: float = 0.0)` | k_f · ∏[reactants] − k_r · ∏[products] |
| `HillActivation` | `(Vmax: float, K: float, n: float = 1.0)` | Vmax · x^n / (K^n + x^n) |
| `HillRepression` | `(Vmax: float, K: float, n: float = 1.0)` | Vmax · K^n / (K^n + x^n) |
| `MichaelisMenten` | `(Vmax: float, Km: float)` | Vmax · S / (Km + S) |
| `ConstitutiveProduction` | `(rate: float)` | k (constant) |
| `LinearDegradation` | `(rate: float)` | γ · x |

```python
from bioprover.models.reactions import HillRepression

hill = HillRepression(Vmax=10.0, K=2.0, n=2)
rate = hill.evaluate({"repressor": 3.0})  # Evaluates K^n / (K^n + x^n) * Vmax
```

---

### `Parameter`

```python
class Parameter:
    def __init__(
        self,
        name: str,
        value: float,
        lower: float = 0.0,
        upper: float = float("inf"),
        uncertainty_type: UncertaintyType = UncertaintyType.UNIFORM,
    ) -> None: ...
```

### `UncertaintyType` (Enum)

`UNIFORM`, `NORMAL`, `LOGNORMAL`

### `ParameterSet`

Container for named parameters.

| Method | Signature | Returns |
|--------|-----------|---------|
| `add_param` | `(param: Parameter) -> None` | `None` |
| `get_param` | `(name: str) -> Parameter` | `Parameter` |
| `to_dict` | `() -> Dict[str, float]` | `dict` |

---

### `SBMLImporter`

```python
class SBMLImporter:
    def __init__(self) -> None: ...
    def import_model(self, filepath: str) -> BioModel: ...
```

### `SBOLImporter`

Import SBOL v2/v3 files (XML/RDF) into BioProver's BioModel.
Extracts `ComponentDefinition` elements (genes, promoters, RBSs,
terminators), circuit topology from `ModuleDefinition` interactions,
and maps common genetic parts to standard kinetic models with default
parameters.

```python
from bioprover.models.sbol_import import SBOLImporter, parse_sbol_file

class SBOLImporter:
    def __init__(self) -> None: ...
    def import_file(self, filepath: str) -> BioModel: ...
    def import_string(self, xml_string: str, name: str = "sbol_model") -> BioModel: ...
```

| Method | Returns | Description |
|--------|---------|-------------|
| `import_file(filepath)` | `BioModel` | Parse an SBOL file and return a BioModel with auto-generated kinetics. |
| `import_string(xml_string, name)` | `BioModel` | Parse an SBOL XML string. |

**Convenience functions:**

```python
parse_sbol_file(filepath: str) -> BioModel
parse_sbol_string(xml_string: str, name: str = "sbol_model") -> BioModel
```

**Part type → kinetic model mapping:**

| SBOL Part Role | Kinetic Model | Default Parameters |
|----------------|---------------|--------------------|
| Promoter (constitutive) | `ConstitutiveProduction` | rate = 0.5 nM/min |
| Promoter (activated) | `HillActivation` | Vmax=10, K=2, n=2 |
| Promoter (repressed) | `HillRepression` | Vmax=10, K=2, n=2 |
| CDS | Protein species | initial_conc = 0 |
| All proteins | `LinearDegradation` | rate = 0.01 /min |

```python
from bioprover.models.sbol_import import parse_sbol_file
from bioprover import verify

model = parse_sbol_file("examples/inverter_circuit.sbol")
result = verify(model, "G[0,100](GFP_protein > 0.0)")
```

### `GenBankImporter`

Import GenBank flat files (.gb/.gbk/.genbank) into BioProver's BioModel.
Extracts FEATURES table entries (CDS, promoter, terminator, RBS, regulatory)
and maps them to kinetic models with default parameters. Only annotations
are used — the DNA sequence is not parsed.

```python
from bioprover.models.genbank_import import GenBankImporter, parse_genbank_file

class GenBankImporter:
    def __init__(self) -> None: ...
    def import_file(self, filepath: str) -> BioModel: ...
    def import_string(self, text: str, name: str = "genbank_model") -> BioModel: ...
```

| Method | Returns | Description |
|--------|---------|-------------|
| `import_file(filepath)` | `BioModel` | Parse a GenBank file and return a BioModel with auto-generated kinetics. |
| `import_string(text, name)` | `BioModel` | Parse a GenBank string. |

**Convenience functions:**

```python
parse_genbank_file(filepath: str) -> BioModel
parse_genbank_string(text: str, name: str = "genbank_model") -> BioModel
```

**GenBank feature → kinetic model mapping:**

| GenBank Feature | Kinetic Model | Default Parameters |
|-----------------|---------------|--------------------|
| CDS (unregulated) | `ConstitutiveProduction` | rate = 0.5 nM/min |
| CDS (activated) | `HillActivation` | Vmax=10, K=2, n=2 |
| CDS (repressed) | `HillRepression` | Vmax=10, K=2, n=2 |
| CDS | Protein species | initial_conc = 0 |
| All proteins | `LinearDegradation` | rate = 0.01 /min |

```python
from bioprover.models.genbank_import import parse_genbank_file
from bioprover import verify

model = parse_genbank_file("examples/toggle_switch.gb")
result = verify(model, "G[0,100](LacI_protein > 0.0)")
```

---

<a id="temporal"></a>
## 3. Temporal Logic — `bioprover.temporal`

### STL Formula AST

```python
class STLFormula(ABC): ...

# Atomic predicates
class Predicate(STLFormula):
    def __init__(self, var: str, op: ComparisonOp, threshold: float) -> None: ...

# Boolean connectives
class STLNot(STLFormula):
    def __init__(self, child: STLFormula) -> None: ...
class STLAnd(STLFormula):
    def __init__(self, left: STLFormula, right: STLFormula) -> None: ...
class STLOr(STLFormula):
    def __init__(self, left: STLFormula, right: STLFormula) -> None: ...
class STLImplies(STLFormula):
    def __init__(self, left: STLFormula, right: STLFormula) -> None: ...

# Temporal operators
class Always(STLFormula):
    def __init__(self, child: STLFormula, interval: Interval) -> None: ...
class Eventually(STLFormula):
    def __init__(self, child: STLFormula, interval: Interval) -> None: ...
class Until(STLFormula):
    def __init__(self, left: STLFormula, right: STLFormula, interval: Interval) -> None: ...
```

---

### `BioSTLParser`

Recursive-descent parser for Bio-STL formulas with domain-specific macros.

```python
class BioSTLParser:
    def __init__(self, custom_macros: Optional[Dict[str, MacroDef]] = None) -> None: ...
    def parse(self, text: str) -> STLFormula: ...
    def available_macros(self) -> List[str]: ...
```

**Built-in domain macros** (expanded during parsing):

| Macro | Expansion | Description |
|-------|-----------|-------------|
| `Oscillates(X, period, amp)` | `G[0,T](F[0,period](X > amp) & F[0,period](X < -amp))` | Sustained oscillation. |
| `Bistable(X, lo, hi)` | `G[0,T]((X < lo) \| (X > hi))` | Bimodal steady state. |
| `Adapts(X, stim, t_adapt)` | `F[0,t_stim](X > stim) & F[t_adapt,T](\|X - X_0\| < eps)` | Perfect adaptation. |
| `Switches(X, lo, hi, t)` | `(X < lo) U[0,t] G[t,T](X > hi)` | Irreversible switch. |
| `ReachesSteadyState(X, val, tol, t)` | `F[0,t](G[t,T](\|X - val\| < tol))` | Convergence to steady state. |
| `MonotoneResponse(in, out, delay)` | Monotone input-output | Monotone dose-response. |
| `Pulse(X, duration, amp)` | Pulse shape | Transient pulse behavior. |

```python
from bioprover.temporal.bio_stl_parser import BioSTLParser

parser = BioSTLParser()
formula = parser.parse("G[0,100](Bistable(U, 1.0, 5.0))")
formula = parser.parse("F[0,50](GFP > 1.0)")
formula = parser.parse("G[0,100](Oscillates(GFP, 20.0, 0.5))")
```

---

### `RobustnessComputer`

Quantitative STL robustness semantics.

```python
class RobustnessComputer:
    def __init__(self) -> None: ...
    def compute_robustness(
        self,
        trajectory: np.ndarray,
        times: np.ndarray,
        formula: STLFormula,
    ) -> float: ...
```

**Returns:** Positive if satisfied, negative if violated.

### `IntervalModelChecker`

Three-valued model checking over interval trajectories.

```python
class IntervalModelChecker:
    def __init__(self) -> None: ...
    def check(self, trajectory: IntervalTrajectory, formula: STLFormula) -> ThreeValuedResult: ...
```

`ThreeValuedResult`: `TRUE`, `FALSE`, `UNKNOWN`

### `StatisticalModelChecker`

```python
class StatisticalModelChecker:
    def __init__(self, alpha: float = 0.01, beta: float = 0.01, delta: float = 0.01) -> None: ...
    def sprt_check(self, verifier_fn: Callable[[], bool], threshold: float = 0.95) -> SPRTResult: ...
```

### `BMCEncoder`

Bounded model checking encoder.

```python
class BMCEncoder:
    def __init__(self, model: BioModel, formula: STLFormula) -> None: ...
    def encode_bmc(self, steps: int) -> ExprNode: ...
```

---

<a id="cegar"></a>
## 4. CEGAR Engine — `bioprover.cegar`

### `VerificationStatus` (Enum)

`VERIFIED`, `FALSIFIED`, `UNKNOWN`, `BOUNDED_GUARANTEE`

### `CEGARConfig`

```python
@dataclass
class CEGARConfig:
    max_iterations: int = 100
    timeout: float = 3600.0
    initial_grid_resolution: int = 4
    step_size: float = 0.01
    delta: float = 1e-3
    stagnation_window: int = 5
    coverage_target: float = 1.0
    strategy_name: str = "auto"
    enable_ai_heuristic: bool = False
    enable_bounded_guarantee: bool = True
    portfolio_configs: Optional[List[CEGARConfig]] = None
    max_workers: int = 1
```

| Method | Returns |
|--------|---------|
| `to_dict()` | `Dict[str, Any]` |

### `CEGAREngine`

Main CEGAR orchestrator for abstraction-refinement verification.

```python
class CEGAREngine:
    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        rhs: Dict[str, ExprNode],
        property_expr: ExprNode,
        property_name: str = "spec",
        initial_region: Optional[IntervalBox] = None,
        solver: Optional[Any] = None,
        model_checker: Optional[Callable] = None,
        refinement_strategy: Optional[RefinementStrategy] = None,
        ai_heuristic: Optional[Callable] = None,
        config: Optional[CEGARConfig] = None,
        hill_params: Optional[List[Dict[str, Any]]] = None,
        monotone_info: Optional[Dict[str, Dict[str, int]]] = None,
        steady_states: Optional[List[Dict[str, float]]] = None,
        jacobian: Optional[Callable] = None,
    ) -> None: ...

    def verify(self) -> VerificationResult: ...
```

The top-level `verify()` convenience function constructs a `CEGAREngine`
from a `BioModel` and `STLFormula` automatically.

### `VerificationResult`

```python
@dataclass
class VerificationResult:
    status: VerificationStatus
    property_name: str = ""
    counterexample: Optional[ConcreteCounterexample] = None
    abstract_counterexample: Optional[AbstractCounterexample] = None
    proof_certificate: Optional[Dict[str, Any]] = None
    statistics: Optional[CEGARStatistics] = None
    coverage: float = 0.0
    robustness: float = 0.0
    termination_reason: Optional[TerminationReason] = None
    message: str = ""
    soundness: Optional[SoundnessAnnotation] = None
```

| Property | Type | Description |
|----------|------|-------------|
| `is_verified` | `bool` | `True` if status is `VERIFIED`. |
| `is_falsified` | `bool` | `True` if status is `FALSIFIED`. |
| `to_dict()` | `Dict` | Serialize to dictionary. |

### `CEGARStatistics`

```python
@dataclass
class CEGARStatistics:
    iterations: int = 0
    total_time: float = 0.0
    abstraction_time: float = 0.0
    model_check_time: float = 0.0
    feasibility_time: float = 0.0
    refinement_time: float = 0.0
    peak_states: int = 0
    peak_predicates: int = 0
    spurious_count: int = 0
    genuine_count: int = 0
    final_coverage: float = 0.0
    strategies_used: Dict[str, int]
```

---

### Abstraction & Refinement

#### `IntervalAbstraction`

```python
class IntervalAbstraction:
    def __init__(self, model: BioModel, resolution: int) -> None: ...
    def initial_states(self) -> List[AbstractState]: ...
    def successors(self, state: AbstractState) -> List[AbstractTransition]: ...
    def refine(self, predicates: List[Predicate]) -> None: ...
```

#### Refinement Strategies

All implement `RefinementStrategy` (ABC):

```python
class RefinementStrategy(ABC):
    def refine(self, abstraction, counterexample, model) -> List[Predicate]: ...
```

| Strategy | Description |
|----------|-------------|
| `StructuralRefinement` | Hill-threshold predicates from network topology. |
| `MonotonicityRefinement` | Exploits GRN monotonicity for vertex-only checking. |
| `TimeScaleRefinement` | Temporal splitting via time-scale separation. |
| `InterpolationRefinement` | Craig interpolants from UNSAT proofs. |
| `SimulationGuidedRefinement` | Simulation-based predicate discovery. |

#### `ConvergenceMonitor`

```python
class ConvergenceMonitor:
    def __init__(self, patience: int = 10) -> None: ...
    def update(self, stats: IterationStats) -> None: ...
    def recommend_action(self) -> StrategySwitchAction: ...
```

#### `SpuriousnessChecker`

```python
class SpuriousnessChecker:
    def __init__(self, solver: AbstractSMTSolver = Z3Solver()) -> None: ...
    def check(self, counterexample: Counterexample) -> SpuriousnessResult: ...
```

---

<a id="repair"></a>
## 5. Repair & Synthesis — `bioprover.repair`

### `SynthesisMode` (Enum)

`FEASIBILITY`, `ROBUSTNESS`, `MINIMAL_PERTURBATION`, `MULTI_OBJECTIVE`

### `SynthesisConfig`

```python
@dataclass
class SynthesisConfig:
    mode: SynthesisMode = SynthesisMode.ROBUSTNESS
    max_outer_iterations: int = 10
    timeout: float = 600.0
    cmaes_config: Optional[CMAESConfig] = None
    cegis_config: Optional[CEGISConfig] = None
    n_cmaes_restarts: int = 3
    perturbation_weight: float = 0.1
    pareto_samples: int = 500
    check_realizability: bool = True
    warm_start_params: Optional[np.ndarray] = None
    verbose: bool = True
```

### `ParameterSynthesizer`

Orchestrator combining CEGIS + CMA-ES + robustness optimization.

```python
class ParameterSynthesizer:
    def __init__(
        self,
        param_set: ParameterSet,
        verifier: VerifierProtocol,
        robustness_fn: Callable[[np.ndarray], float],
        original_params: Optional[np.ndarray] = None,
        realizability_checker: Optional[RealizabilityChecker] = None,
        config: Optional[SynthesisConfig] = None,
    ) -> None: ...

    def synthesize(self) -> SynthesisResult: ...
    def warm_start(self, params: np.ndarray, robustness: float) -> None: ...
    def generate_report(self, result: SynthesisResult) -> RepairReport: ...
```

### `SynthesisResult`

```python
@dataclass
class SynthesisResult:
    success: bool = False
    mode: SynthesisMode = SynthesisMode.FEASIBILITY
    repair_result: Optional[RepairResult] = None
    pareto_frontier: Optional[ParetoFrontier] = None
    realizability_report: Optional[RealizabilityReport] = None
    cegis_result: Optional[CEGISResult] = None
    optimization_result: Optional[OptimizationResult] = None
    outer_iterations: int = 0
    total_time: float = 0.0
    history: List[Dict[str, Any]]
```

| Method | Returns |
|--------|---------|
| `summary()` | `str` — Human-readable summary. |

### `RepairResult`

```python
@dataclass
class RepairResult:
    original: np.ndarray
    repaired: np.ndarray
    parameter_names: List[str]
    robustness_before: float = float("-inf")
    robustness_after: float = float("-inf")
    verified: bool = False
    method: str = ""
    metadata: Dict[str, Any]
```

| Property | Type | Description |
|----------|------|-------------|
| `delta` | `np.ndarray` | `repaired - original` |
| `perturbation_l1` | `float` | L1 norm of parameter change. |
| `perturbation_l2` | `float` | L2 norm of parameter change. |
| `perturbation_linf` | `float` | L∞ norm of parameter change. |
| `relative_change` | `np.ndarray` | Relative change per parameter. |
| `robustness_improvement` | `float` | Robustness delta. |

| Method | Returns |
|--------|---------|
| `most_changed(k=5)` | `List[Tuple[str, float, float, float]]` — Top-k changed parameters. |
| `parameter_dict(repaired=True)` | `Dict[str, float]` — Named parameter values. |

### `RepairReport`

```python
class RepairReport:
    def __init__(
        self,
        primary: RepairResult,
        alternatives: Optional[List[RepairResult]] = None,
        sensitivity_rankings: Optional[List[Tuple[str, float]]] = None,
        confidence: float = 1.0,
        notes: Optional[str] = None,
    ) -> None: ...
```

| Property / Method | Returns |
|-------------------|---------|
| `success` | `bool` |
| `summary_text()` | `str` — Human-readable summary. |
| `biological_interpretation()` | `str` — Biological interpretation of repairs. |

### `CEGISLoop`

```python
class CEGISLoop:
    def __init__(self, config: CEGISConfig) -> None: ...
    def run(self, initial_guess, verifier, objective) -> CEGISResult: ...
```

### `RobustnessOptimizer`

```python
class RobustnessOptimizer:
    def __init__(self, config: CMAESConfig) -> None: ...
    def optimize(self, robustness_fn, bounds, x0) -> OptimizationResult: ...
```

### `RealizabilityChecker`

```python
class RealizabilityChecker:
    def __init__(self) -> None: ...
    def check(self, params: Dict[str, float]) -> RealizabilityReport: ...
```

### `DesignSpace`

```python
class DesignSpace:
    def __init__(self) -> None: ...
    def add_point(self, point: DesignPoint) -> None: ...
    def pareto_frontier(self) -> ParetoFrontier: ...
```

---

<a id="soundness"></a>
## 6. Soundness — `bioprover.soundness`

### `SoundnessLevel` (Enum)

```python
class SoundnessLevel(Enum):
    SOUND = auto()           # Full mathematical guarantee
    DELTA_SOUND = auto()     # Sound up to delta (dReal)
    BOUNDED = auto()         # Sound within bounded horizon
    APPROXIMATE = auto()     # Informative but not guaranteed
```

Supports comparison operators (`<`, `<=`, `>`, `>=`) and `SoundnessLevel.meet(a, b)` returns the weakest of two levels.

### `SoundnessAnnotation`

```python
@dataclass
class SoundnessAnnotation:
    level: SoundnessLevel
    assumptions: List[str] = field(default_factory=list)
    delta: Optional[float] = None
    time_bound: Optional[float] = None
    approximation_error: Optional[float] = None
```

| Method | Returns | Description |
|--------|---------|-------------|
| `weaken_to(level, reason)` | `SoundnessAnnotation` | Weaken to a new level with a reason. |
| `with_delta(delta)` | `SoundnessAnnotation` | Set delta-satisfiability. |
| `with_time_bound(t)` | `SoundnessAnnotation` | Set bounded time horizon. |

```python
from bioprover.soundness import SoundnessLevel, SoundnessAnnotation

ann = SoundnessAnnotation(level=SoundnessLevel.SOUND)
ann = ann.with_delta(1e-3)   # Now DELTA_SOUND
print(ann.level)             # SoundnessLevel.DELTA_SOUND
print(ann.assumptions)       # ['dReal delta-satisfiability with delta=0.001']
```

### `ErrorBudget`

Tracks individual error sources and computes the combined error via
root-sum-of-squares composition.

```python
@dataclass
class ErrorBudget:
    delta: float = 0.0              # dReal delta-satisfiability tolerance
    epsilon: float = 0.0            # ODE integration error bound
    truncation: float = 0.0         # Taylor truncation error
    discretization: float = 0.0     # time-discretization error
```

| Method | Returns | Description |
|--------|---------|-------------|
| `propagate_errors()` | `float` | Combined error (RSS of all sources). |

```python
from bioprover.soundness import ErrorBudget

budget = ErrorBudget(delta=1e-3, epsilon=1e-4, truncation=1e-5)
print(budget.propagate_errors())  # ~0.001005
```

`SoundnessAnnotation.with_error_budget(budget)` attaches an `ErrorBudget` to
an annotation and weakens the soundness level appropriately.

---

<a id="solver"></a>
## 7. Solver — `bioprover.solver`

### `Interval`

Rigorous interval arithmetic with outward-rounded operations.

```python
class Interval:
    def __init__(self, lo: float, hi: Optional[float] = None) -> None: ...
```

| Method | Returns | Description |
|--------|---------|-------------|
| `mid()` | `float` | Centre of the interval. |
| `width()` | `float` | `hi - lo`. |
| `radius()` | `float` | Half-width. |
| `is_empty()` | `bool` | Empty interval check. |
| `is_thin(tol=0.0)` | `bool` | Near-point interval. |
| `contains(x)` | `bool` | Point or interval containment. |
| `contains_zero()` | `bool` | Does interval contain 0? |
| `overlaps(other)` | `bool` | Do intervals overlap? |
| `bisect()` | `Tuple[Interval, Interval]` | Split at midpoint. |
| `subdivide(n)` | `List[Interval]` | Split into n sub-intervals. |
| `to_validated()` | `ValidatedInterval` | Convert to mpmath-backed interval. |

**Overloaded operators:** `+`, `-`, `*`, `/`, `**`, `abs`, `-` (unary)

**Elementary functions:** `sqrt()`, `exp()`, `log()`, `sin()`, `cos()`

### `ValidatedInterval`

mpmath-backed rigorous interval arithmetic with quad-precision guarantees.
Same interface as `Interval` but with higher precision.

```python
use_validated_arithmetic(enabled: bool = True)  # Module-level switch
```

### `IntervalVector`

```python
class IntervalVector:
    def __init__(self, intervals: List[Interval]) -> None: ...
    def dim(self) -> int: ...
    def midpoint(self) -> np.ndarray: ...
    def hull(self, other) -> IntervalVector: ...
    def volume(self) -> float: ...
```

### `ValidatedODEIntegrator`

```python
class ValidatedODEIntegrator:
    def __init__(self, order: int = 4, min_step: float = 1e-10, max_step: float = 0.1) -> None: ...
    def step(self, x, p, t, dt) -> StepResult: ...
    def integrate(self, x0, p, t_span) -> Flowpipe: ...
```

### `TaylorModel`

Taylor model arithmetic for tighter flowpipe enclosures.

```python
class TaylorModel:
    def __init__(self, coefficients: np.ndarray, remainder: Interval, order: int) -> None: ...
    def evaluate(self, point: np.ndarray) -> Interval: ...
    def bound(self) -> Interval: ...
```

### `Flowpipe`

Sequence of reachable-set enclosures over time.

| Method | Returns | Description |
|--------|---------|-------------|
| `enclose(point)` | `bool` | Does the flowpipe enclose the point? |
| `time_steps()` | `List[float]` | Time stamps of each segment. |
| `get_enclosure(t)` | `IntervalVector` | Enclosure at time t. |

<a id="proof-certificates"></a>
### Proof Certificates — `bioprover.solver.proof_certificate`

#### `ProofCertificate`

Base certificate with versioning, timestamps, and metadata.

```python
@dataclass
class ProofCertificate:
    version: str
    timestamp: str
    metadata: Dict[str, Any]
```

#### `FlowpipeCertificate`

Records the ODE system, initial conditions, integration parameters, and
flowpipe segments for independent re-verification.

```python
class FlowpipeCertificate(ProofCertificate):
    def __init__(self, ode_system, initial_conditions, segments, params) -> None: ...
    @classmethod
    def from_integration(cls, integration_result) -> "FlowpipeCertificate": ...
    def validate(self) -> bool: ...
```

#### `InvariantCertificate`

Proves that invariants (lower/upper bounds, linear constraints) hold
across all flowpipe segments.

| Method | Returns | Description |
|--------|---------|-------------|
| `check_lower_bound(species, bound)` | `bool` | Verify lower bound holds. |
| `check_upper_bound(species, bound)` | `bool` | Verify upper bound holds. |
| `check_linear_invariant(coefficients, bound)` | `bool` | Verify linear invariant. |

#### `SoundnessCertificate`

Wraps a proof certificate with a `SoundnessAnnotation` and `ErrorBudget`.

```python
class SoundnessCertificate:
    def __init__(self, certificate: ProofCertificate,
                 soundness: SoundnessAnnotation,
                 error_budget: ErrorBudget) -> None: ...
```

#### `validate_certificate(certificate) -> bool`

Standalone validator that re-checks a certificate without access to the
original verifier. Returns `True` if the certificate is self-consistent.

```python
from bioprover.solver.proof_certificate import validate_certificate

valid = validate_certificate(cert)
```

#### `compose_flowpipe_certificates(certs) -> FlowpipeCertificate`

Compose a temporal sequence of flowpipe certificates into a single certificate.

---

<a id="smt"></a>
## 8. SMT — `bioprover.smt`

### `SMTResult` (Enum)

`SAT`, `UNSAT`, `UNKNOWN`

### `Z3Solver`

```python
class Z3Solver(AbstractSMTSolver):
    def __init__(self, tactic: TacticPreset = TacticPreset.AUTO) -> None: ...
    def solve(self, assertions: List[ExprNode], timeout: float = 60.0) -> Tuple[SMTResult, Optional[Model]]: ...
```

### `DRealSolver`

```python
class DRealSolver(AbstractSMTSolver):
    def __init__(self, delta: float = 1e-3, binary_path: str = "dreal") -> None: ...
```

### `PortfolioSolver`

Run multiple solver strategies in parallel; return the first result.

```python
class PortfolioSolver(AbstractSMTSolver):
    def __init__(self, solvers: List[AbstractSMTSolver] | None = None) -> None: ...
```

### `CraigInterpolant`

Extract Craig interpolants from UNSAT proofs for CEGAR refinement.

```python
class CraigInterpolant:
    def extract(self, proof, partition_A, partition_B) -> Optional[ExprNode]: ...
```

---

<a id="encoding"></a>
## 9. Encoding — `bioprover.encoding`

### Expression IR

All SMT expressions are built from the `ExprNode` abstract base class.

| Category | Classes |
|----------|---------|
| **Leaf** | `Const(value)`, `Var(name)` |
| **Arithmetic** | `Add`, `Mul`, `Div`, `Pow`, `Neg` |
| **Functions** | `Exp`, `Log`, `Sin`, `Cos`, `Sqrt`, `Abs`, `Min`, `Max` |
| **Biology** | `HillAct(x, k, n)`, `HillRep(x, k, n)` |
| **Comparison** | `Lt`, `Le`, `Eq`, `Ge`, `Gt` |
| **Logic** | `And`, `Or`, `Not`, `Implies`, `ForAll`, `Exists` |

### `expr_to_smtlib(expr: ExprNode) -> str`

Serialize an expression tree to SMT-LIB 2.6 format.

### `ODEEncoding`

```python
class ODEEncoding:
    def __init__(self, system: ODESystem, method: DiscretizationMethod = DiscretizationMethod.FORWARD_EULER, steps: int = 100) -> None: ...
    def encode(self) -> List[ExprNode]: ...
```

### `DiscretizationMethod` (Enum)

`FORWARD_EULER`, `BACKWARD_EULER`, `MIDPOINT`, `RK4`

### `HillEncodingConfig`

```python
class HillEncodingConfig:
    def __init__(self, integer_hill_exact: bool = True, piecewise_linear_fallback: bool = True, approximation_segments: int = 8) -> None: ...
```

---

<a id="stochastic"></a>
## 10. Stochastic — `bioprover.stochastic`

### `DirectMethod`

Exact stochastic simulation (Gillespie SSA).

```python
class DirectMethod:
    def __init__(self, model: BioModel, seed: int = 42) -> None: ...
    def run_trajectory(self, x0: np.ndarray, t_end: float) -> StochasticTrajectory: ...
```

### `NextReactionMethod`

Gibson-Bruck optimised SSA.

### `EnsembleSimulator`

```python
class EnsembleSimulator:
    def __init__(self, method: str = "direct", n_workers: int = 4) -> None: ...
    def run_ensemble(self, model, n_trajectories, t_end) -> List[StochasticTrajectory]: ...
```

### `FSPSolver`

Finite State Projection for exact probability distributions.

```python
class FSPSolver:
    def __init__(self, model: BioModel, truncation: int = 100) -> None: ...
    def solve(self, t_end: float) -> MarginalDistribution: ...
```

### Tau-Leaping Variants

`ExplicitTauLeaping`, `ImplicitTauLeaping`, `MidpointTauLeaping`

### `MomentClosureSolver`

```python
class MomentClosureSolver:
    def __init__(self, model: BioModel, closure: str = "normal", max_order: int = 2) -> None: ...
    def solve(self, t_end: float) -> MomentTrajectory: ...
```

<a id="lna"></a>
### LNA — `bioprover.stochastic.lna`

Linear Noise Approximation with rigorous O(1/Ω) error bounds based on van Kampen's
system-size expansion. Decomposes molecular counts as X(t) = Ω·φ(t) + √Ω·ξ(t) where
φ(t) satisfies deterministic rate equations and ξ(t) has Gaussian fluctuations with
covariance Σ(t) satisfying dΣ/dt = A·Σ + Σ·Aᵀ + D.

#### `LNAResult`

```python
@dataclass
class LNAResult:
    mean_trajectory: np.ndarray       # shape (n_times, n_species)
    covariance_trajectory: np.ndarray # shape (n_times, n_species, n_species)
    time_points: np.ndarray
    error_bound_mean: float           # O(1/Ω) mean approximation error
    error_bound_covariance: float     # O(1/Ω) covariance approximation error
    error_budget: ErrorBudget
    steady_state_cov: Optional[np.ndarray]
```

#### `LNASolver`

```python
class LNASolver:
    def __init__(self, reactions: List[MomentReaction], num_species: int,
                 volume: float = 1.0) -> None: ...
    def jacobian(self, concentrations: np.ndarray) -> np.ndarray: ...
    def diffusion_matrix(self, concentrations: np.ndarray) -> np.ndarray: ...
    def compute_error_bound(self, initial: np.ndarray, T: float) -> Tuple[float, float]: ...
    def solve(self, initial: np.ndarray, t_span: Tuple[float, float],
              t_eval: Optional[np.ndarray] = None,
              compute_bounds: bool = True) -> LNAResult: ...
    def steady_state_covariance(self, ss: np.ndarray) -> np.ndarray: ...
    def find_steady_states(self, initial_guesses: Optional[List[np.ndarray]] = None,
                           n_random: int = 20) -> List[SteadyStateInfo]: ...
```

| Method | Description |
|--------|-------------|
| `jacobian(concentrations)` | Numerical Jacobian of macroscopic dynamics at given state |
| `diffusion_matrix(concentrations)` | Diffusion matrix D = S · diag(a(φ)) · Sᵀ |
| `compute_error_bound(initial, T)` | Returns (mean_bound, cov_bound) via Kurtz 1972 theory |
| `solve(initial, t_span, ...)` | Full LNA solve: mean trajectory + covariance + error bounds |
| `steady_state_covariance(ss)` | Solve continuous Lyapunov equation for steady-state Σ |
| `find_steady_states(...)` | Locate steady states via numerical root-finding |

#### `BimodalityDetector`

Detects when LNA is unreliable due to bistability or multimodality.

```python
class BimodalityDetector:
    def __init__(self, solver: LNASolver) -> None: ...
    def detect(self, initial_guesses: Optional[List[np.ndarray]] = None,
               n_random: int = 30) -> Tuple[StabilityType, List[SteadyStateInfo]]: ...
    def trace_determinant_test_2d(self, steady_state: np.ndarray) -> Dict[str, Any]: ...
    def validate_lna(self, initial_concentrations: np.ndarray,
                     t_span: Tuple[float, float]) -> Dict[str, Any]: ...
```

`StabilityType` enum: `MONOSTABLE`, `BISTABLE`, `MULTISTABLE`, `OSCILLATORY`, `UNKNOWN`.

#### `StochasticAnalysisPipeline`

Automatically selects the best stochastic analysis method for a given system.

```python
class StochasticAnalysisPipeline:
    def __init__(self, reactions: List[MomentReaction], num_species: int,
                 volume: float = 1.0,
                 state_space_bounds: Optional[List[int]] = None) -> None: ...
    def select_method(self, initial_concentrations: Optional[np.ndarray] = None
                      ) -> MethodSelection: ...
    def run(self, initial_concentrations: np.ndarray,
            t_span: Tuple[float, float],
            t_eval: Optional[np.ndarray] = None) -> Dict[str, Any]: ...
```

Decision logic:
1. Large Ω (>100) + monostable → **LNA** (fastest, rigorous O(1/Ω) bounds)
2. Small state space (<1000) → **FSP** (exact, no approximation error)
3. Moderate systems → **Moment closure** with bimodality check
4. Fallback → **Hybrid SSA/ODE** (Monte Carlo)

#### Convenience Functions

| Function | Description |
|----------|-------------|
| `lna_error_budget(reactions, num_species, volume, initial, T)` | Compute ErrorBudget for LNA truncation error |
| `validate_lna_applicability(reactions, num_species, volume, initial, T)` | Check LNA preconditions and return validity report |

---

<a id="ai"></a>
## 11. AI / ML — `bioprover.ai`

### `CircuitEncoder`

Converts a BioModel to a graph representation for GNN input.

```python
class CircuitEncoder:
    def __init__(self) -> None: ...
    def encode(self, model: BioModel) -> CircuitGraph: ...
```

### `GraphSAGEEncoder`

```python
class GraphSAGEEncoder:
    def __init__(self, embedding_dim: int = 64, num_layers: int = 3) -> None: ...
    def encode(self, graph: CircuitGraph) -> np.ndarray: ...
```

### `PredicatePredictor`

```python
class PredicatePredictor:
    def __init__(self, model_path: str | None = None) -> None: ...
    def predict(self, model, abstraction, counterexample) -> List[CandidatePredicate]: ...
```

### `RobustnessSurrogate`

Gaussian process surrogate for fast robustness estimation.

```python
class RobustnessSurrogate:
    def __init__(self) -> None: ...
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, params: np.ndarray) -> Tuple[float, float]: ...  # (mean, std)
```

### `PredictionQualityMonitor`

Sliding-window monitor that tracks prediction precision and F1 score.
Automatically disables ML heuristics when quality degrades below threshold.

```python
class PredictionQualityMonitor:
    def record(self, predicted, actual) -> None: ...
    def get_metrics(self) -> Dict[str, float]: ...
    def get_family_metrics(self) -> Dict[str, Dict[str, float]]: ...
    def get_report(self) -> str: ...
```

<a id="training-pipeline"></a>
### Training Pipeline — `bioprover.ai.training_pipeline`

#### `TrainingDataGenerator`

Collects CEGAR verification traces and generates labelled training examples
for the predicate predictor.

```python
class TrainingDataGenerator:
    def __init__(self, augmenter: DataAugmenter | None = None) -> None: ...
    def add_trace(self, trace: VerificationTrace) -> None: ...
    def generate(self) -> Dataset: ...
    def generate_from_engine(self, engine, models, formulas) -> Dataset: ...
    @property
    def n_traces(self) -> int: ...
    @property
    def predicate_stats(self) -> Dict: ...
```

#### `CrossValidator`

K-fold cross-validation for the predicate predictor MLP, returning a
`TrainingReport` with per-fold precision, recall, F1, and loss.

```python
class CrossValidator:
    def __init__(self, hidden_dims: List[int], config: TrainingConfig,
                 k_folds: int = 5) -> None: ...
    def run(self, dataset: Dataset) -> TrainingReport: ...
```

#### `TrainingReport`

```python
@dataclass
class TrainingReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    loss: float
    mrr_score: float
    ndcg_score: float
    n_train: int
    n_val: int
    n_epochs_run: int
    fold_reports: List[Dict]
    training_history: List[Dict]
    def summary(self) -> str: ...
```

#### Supporting Classes

| Class | Purpose |
|-------|---------|
| `CircuitData` | Graph embedding + structural/kinetic features for a circuit. |
| `VerificationTrace` | Complete CEGAR trace (circuit, steps, success, time). |
| `RefinementExampleRecord` | Context features + predicate index + effectiveness. |
| `DataAugmenter` | Species permutation, parameter perturbation, time rescaling. |
| `Dataset` | Container with `split()` and `k_fold()` methods. |
| `Trainer` | SGD trainer with early stopping and learning-rate schedules. |
| `MLEvaluator` | Cross-validation and hold-out evaluation wrapper. |

<a id="online-ml"></a>
### Online ML — `bioprover.ai.online_learner`

Online adaptation from CEGAR verification traces with out-of-distribution
detection and ablation experiment support.

#### `CEGARTraceEntry`

```python
@dataclass
class CEGARTraceEntry:
    circuit_features: np.ndarray   # feature vector for the circuit
    predicate_index: int           # which predicate was selected
    success: bool                  # whether refinement succeeded
    iterations: int                # CEGAR iteration number
    robustness_delta: float        # change in robustness
    timestamp: float               # wall-clock time
```

#### `OnlineLearner`

Incremental MLP learner with exponential moving average (EMA) weights,
priority replay buffer, and cosine-annealed learning rate.

```python
class OnlineLearner:
    def __init__(self, feature_dim: int, n_predicates: int = 50,
                 hidden_dims: Optional[List[int]] = None,
                 lr: float = 0.001, buffer_capacity: int = 5000) -> None: ...
    def record_trace(self, entry: CEGARTraceEntry) -> None: ...
    def update(self, batch_size: int = 16) -> float: ...
    def predict(self, circuit_features: np.ndarray,
                candidate_indices: Optional[List[int]] = None
                ) -> np.ndarray: ...
    def prediction_accuracy(self) -> float: ...
    def get_metrics(self) -> Dict[str, Any]: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "OnlineLearner": ...
```

#### `AblationController`

Controls ablation experiments across predicate selection strategies.

```python
class AblationMode(Enum):
    FULL = auto()              # ML predictor + domain heuristics
    RANDOM = auto()            # Uniform random selection
    DOMAIN_HEURISTIC = auto()  # Domain heuristics only (no ML)
    NO_ML = auto()             # Disable all ML, use fixed ordering
```

```python
class AblationController:
    def __init__(self, feature_dim: int, n_predicates: int = 50) -> None: ...
    def set_mode(self, mode: AblationMode) -> None: ...
    def begin_run(self) -> None: ...
    def record_step(self, success: bool) -> None: ...
    def end_run(self, converged: bool, iterations: int) -> AblationRunMetrics: ...
    def select_predicate(self, circuit_features: np.ndarray,
                         candidates: List[int]) -> int: ...
    def get_results(self) -> Dict[str, List[AblationRunMetrics]]: ...
    def generate_report(self) -> Dict[str, Any]: ...
```

#### `OutOfDistributionDetector`

Mahalanobis-distance-based OOD detector with incremental covariance updates.

```python
class OutOfDistributionDetector:
    def __init__(self, feature_dim: int, threshold_percentile: float = 95.0,
                 min_samples: int = 30) -> None: ...
    def fit_incremental(self, x: np.ndarray) -> None: ...
    def fit_batch(self, X: np.ndarray) -> None: ...
    def mahalanobis_distance(self, x: np.ndarray) -> float: ...
    def is_ood(self, x: np.ndarray) -> bool: ...
    def ood_detection_rate(self) -> float: ...
    def false_positive_rate(self) -> float: ...
    def get_metrics(self) -> Dict[str, Any]: ...
```

#### `OnlineCEGARIntegration`

End-to-end integration of online learning with the CEGAR loop.

```python
class OnlineCEGARIntegration:
    def __init__(self, feature_dim: int, n_predicates: int = 50,
                 ablation_mode: Optional[AblationMode] = None) -> None: ...
    def select_predicate(self, candidates: List[int],
                         circuit_features: np.ndarray) -> int: ...
    def report_outcome(self, predicate_index: int, success: bool,
                       robustness_delta: float = 0.0) -> None: ...
    def step(self, candidates: List[int], circuit_features: np.ndarray,
             success: bool) -> int: ...
    def cumulative_regret(self) -> float: ...
    def iteration_count(self) -> int: ...
    def ood_fallback_count(self) -> int: ...
```

---

<a id="compositional"></a>
## 12. Compositional — `bioprover.compositional`

### `verify_compositional(model, spec, *, timeout, max_module_size, strategy_name)`

**The recommended API for circuits with 5+ species.** Decomposes a model
into modules and verifies each independently. Scales to 50+ species.

```python
from bioprover.compositional.compositional_runner import verify_compositional

result = verify_compositional(
    model, spec_str,
    timeout=60.0,         # wall-clock seconds
    max_module_size=3,    # max species per module
    strategy_name="auto", # CEGAR strategy
)
result.status           # VerificationStatus.VERIFIED
result.n_modules        # e.g. 7
result.total_time       # e.g. 0.4
result.module_results   # dict of per-module VerificationResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `BioModel` | — | The biological model to verify. |
| `spec` | `str` | — | Bio-STL specification string. |
| `timeout` | `float` | `120.0` | Maximum wall-clock seconds. |
| `max_iterations` | `int` | `50` | Max CEGAR iterations per module. |
| `max_module_size` | `int` | `3` | Maximum species per decomposed module. |
| `strategy_name` | `str` | `"auto"` | CEGAR refinement strategy. |

Returns `CompositionalResult` with `to_verification_result()` for standard API.

### `CircularAG`

Circular assume-guarantee reasoning for compositional verification.

```python
class CircularAG:
    def __init__(self) -> None: ...
    def decompose(self, model: BioModel) -> List[Contract]: ...
    def verify_compositionally(self, model, formula) -> VerificationResult: ...
```

### `ContractSynthesis`

```python
class ContractSynthesis:
    def __init__(self) -> None: ...
    def synthesize(self, components: List[BioModel], property: STLFormula) -> List[Contract]: ...
```

<a id="circular-ag"></a>
### Circular AG — `bioprover.compositional.circular_ag`

#### `CircularAGChecker`

Co-inductive fixed-point algorithm for circular module dependencies.
Iterates assumptions and guarantees until convergence or divergence.

```python
class CircularAGChecker:
    def __init__(self, modules, contracts, dependency_graph) -> None: ...
    def check(self) -> CircularAGResult: ...
    def check_with_symmetry(self) -> CircularAGResult: ...
    def check_quantitative(self) -> CircularAGResult: ...
```

#### `CircularAGResult`

```python
@dataclass
class CircularAGResult:
    status: ConvergenceStatus      # CONVERGED, DIVERGED, FAILED, MAX_ITERATIONS
    proof_valid: bool
    history: List[FixedPointState]
    soundness_argument: str
```

#### `TopologyAnalyzer`

Analyzes the dependency graph to determine whether circular AG reasoning
is needed.

```python
class TopologyAnalyzer:
    def __init__(self, dependency_graph) -> None: ...
    def is_acyclic(self) -> bool: ...
    def find_cycles(self) -> List[List]: ...
    def topological_order(self) -> List: ...
    def needs_circular_ag(self) -> bool: ...
    def recommend_strategy(self) -> str: ...    # "sequential" or "circular"
    def strongly_connected_components(self) -> List[List]: ...
```

#### `WellFormednessChecker`

Validates that a set of assume-guarantee contracts is well-formed:
compatible, monotone, and contracting (spectral radius < 1).

```python
class WellFormednessChecker:
    def __init__(self, contracts, dependency_graph, modules=None) -> None: ...
    def check_all(self) -> AGWellFormedness: ...
    def check_contract_compatibility(self) -> bool: ...
    def check_progress_condition(self) -> bool: ...
    def recommend_strategy(self) -> str: ...
```

#### Supporting Classes

| Class | Purpose |
|-------|---------|
| `ConvergenceStatus` | Enum: `CONVERGED`, `DIVERGED`, `FAILED`, `MAX_ITERATIONS`, `INCOMPLETE`. |
| `FixedPointState` | Iteration snapshot with assumptions, guarantees, verification results. |
| `AGWellFormedness` | Result of well-formedness checks with coupling metrics. |
| `AGFailureDiagnostics` | Coupling matrix, spectral radius, failure suggestions. |

<a id="ag-soundness"></a>
### AG Soundness — `bioprover.compositional.ag_soundness`

Formal soundness proofs for assume-guarantee composition of ODE systems.
Implements three theorems with explicit sufficient conditions grounded in
differential inequality theory, Gronwall's inequality, and spectral analysis.

#### `SoundnessProver`

Unified orchestrator for the three composition theorems.

```python
class SoundnessProver:
    def __init__(self, modules: List[ModuleODE], contracts: List[Contract],
                 lipschitz_matrix: np.ndarray) -> None: ...
    @property
    def n_modules(self) -> int: ...
    @property
    def spectral_radius(self) -> float: ...
    @property
    def is_contractive(self) -> bool: ...
    def verify_lipschitz_bounds(self, test_points=None, n_samples=100
                                ) -> List[ConditionCheckResult]: ...
    def compute_coupling_error(self, time_horizon: float) -> float: ...
    def compute_composed_robustness(self, time_horizon: float) -> Tuple[float, float]: ...
    def prove_composition(self, time_horizon: float) -> AGSoundnessCertificate: ...
    def prove_convergence(self) -> AGSoundnessCertificate: ...
    def full_proof(self, time_horizon: float) -> AGSoundnessCertificate: ...
    def combined_annotation(self, time_horizon: float) -> SoundnessAnnotation: ...
```

```python
prover = SoundnessProver(
    modules=[m1, m2, m3],
    contracts=[c1, c2, c3],
    lipschitz_matrix=np.array([
        [0.0, 0.3, 0.1],
        [0.2, 0.0, 0.4],
        [0.1, 0.2, 0.0],
    ]),
)
cert = prover.prove_composition(time_horizon=10.0)
print(cert.summary())
```

#### `Theorem1_AGComposition`

AG composition rule for ODE systems with Lipschitz coupling.
Given n modules with ODE dynamics dx_i/dt = f_i(x_i, y_i), verifies that
isolated guarantees transfer to the composed system when ρ(L) < 1.

```python
class Theorem1_AGComposition:
    @staticmethod
    def check_conditions(modules: List[ModuleODE], lipschitz_matrix: np.ndarray,
                         isolation_verified: Optional[List[bool]] = None,
                         contracts_well_formed: bool = True
                         ) -> List[ConditionCheckResult]: ...
    @staticmethod
    def prove(modules: List[ModuleODE], lipschitz_matrix: np.ndarray, **kwargs
              ) -> AGSoundnessCertificate: ...
```

Conditions checked: (C1) isolation verification, (C2) Lipschitz coupling,
(C3) spectral radius ρ(L) < 1, (C4) well-formed contract network.

#### `Theorem2_RobustnessComposition`

Quantitative robustness bound: ρ_system ≥ min_i(ρ_i) − E_coupling(T).

```python
class Theorem2_RobustnessComposition:
    @staticmethod
    def compute_composed_robustness(modules: List[ModuleODE],
                                    coupling_analysis: CouplingAnalysis,
                                    time_horizon: float
                                    ) -> Tuple[float, float]: ...
    @staticmethod
    def check_conditions(modules, coupling_analysis, time_horizon
                         ) -> List[ConditionCheckResult]: ...
    @staticmethod
    def prove(modules, coupling_analysis, time_horizon
              ) -> AGSoundnessCertificate: ...
```

#### `Theorem3_CircularAGConvergence`

Convergence analysis for circular AG iteration. Convergence iff ρ(L) < 1,
with iteration bound K = ⌈log(C/ε) / (−log(ρ(L)))⌉.

```python
class Theorem3_CircularAGConvergence:
    @staticmethod
    def required_iterations(spectral_radius: float, initial_error: float,
                            tolerance: float) -> int: ...
    @staticmethod
    def error_after_k_iterations(spectral_radius: float, initial_error: float,
                                  k: int) -> float: ...
    @staticmethod
    def check_conditions(coupling_analysis: CouplingAnalysis,
                         tolerance: float = 1e-6) -> List[ConditionCheckResult]: ...
    @staticmethod
    def prove(coupling_analysis: CouplingAnalysis, tolerance: float = 1e-6
              ) -> AGSoundnessCertificate: ...
```

#### Supporting Classes

| Class | Purpose |
|-------|---------|
| `ProofStatus` | Enum: `PROVED`, `FAILED`, `PARTIAL`, `UNKNOWN`. |
| `ConditionCheckResult` | Result of checking a single theorem condition. |
| `CouplingAnalysis` | Spectral analysis: spectral_radius, dominant_eigenvalue, is_contractive. |
| `AGSoundnessCertificate` | Full proof certificate with conditions, theorem name, and summary. |
| `ModuleODE` | Module descriptor with dynamics function, species count, robustness margin. |

#### Utility Functions

| Function | Description |
|----------|-------------|
| `compute_spectral_radius(matrix)` | Spectral radius and dominant eigenvalue of a matrix |
| `analyze_coupling(lipschitz_matrix)` | Full spectral analysis of the coupling matrix |
| `gronwall_error_bound(L, e0, T)` | Gronwall's inequality error bound |
| `estimate_lipschitz_constant(f, domain, n_samples)` | Numerical Lipschitz constant estimation |
| `estimate_coupling_matrix(modules)` | Estimate coupling matrix from module dynamics |

---

<a id="library"></a>
## 13. Library — `bioprover.library`

### `PartsDatabase`

Registry of characterized biological parts.

```python
class PartsDatabase:
    def __init__(self, data_path: str | None = None) -> None: ...
    def get_part(self, part_id: str) -> Part: ...
    def list_parts(self, part_type: str | None = None) -> List[Part]: ...
```

### `MotifLibrary`

Common regulatory motifs with models and specifications.

```python
class MotifLibrary:
    def __init__(self) -> None: ...
    def get_motif(self, name: str) -> Motif: ...
    def list_motifs(self) -> List[str]: ...
```

### `ModelTemplates`

```python
class ModelTemplates:
    def toggle_switch(self, params: Dict | None = None) -> BioModel: ...
    def repressilator(self, params: Dict | None = None) -> BioModel: ...
    def feed_forward_loop(self, loop_type: str = "C1-FFL") -> BioModel: ...
```

### `ParameterDatabase`

Literature-sourced kinetic parameters with provenance.

```python
class ParameterDatabase:
    def __init__(self) -> None: ...
    def lookup(self, name: str) -> List[ParameterRecord]: ...
```

---

<a id="evaluation"></a>
## 14. Evaluation — `bioprover.evaluation`

### `BenchmarkSuite`

```python
class BenchmarkSuite:
    def __init__(self, suite_name: str = "full") -> None: ...
    def run(self, timeout_per_case: float = 300.0) -> BenchmarkResults: ...
    def list_cases(self) -> List[str]: ...
```

### `BenchmarkCircuit`

```python
@dataclass
class BenchmarkCircuit:
    model: BioModel
    specification: STLFormula
    expected_result: VerificationStatus
    difficulty: BenchmarkDifficulty  # EASY, MEDIUM, HARD, FRONTIER
    name: str
    description: str
    category: str
    metadata: Dict[str, Any]
    tags: List[str]
```

### `BaselineComparison`

```python
class BaselineComparison:
    def __init__(self, baselines: List[str]) -> None: ...
    def compare(self, suite: BenchmarkSuite) -> ComparisonTable: ...
```

### `MutationTester`

Fault injection for soundness testing.

```python
class MutationTester:
    def __init__(self) -> None: ...
    def mutate(self, model: BioModel, n_mutants: int = 100) -> List[BioModel]: ...
    def run(self, model: BioModel, formula: STLFormula) -> MutationReport: ...
```

<a id="ablation"></a>
### Ablation — `bioprover.evaluation.ablation`

#### `AblationConfig` (Enum)

Predefined ablation configurations:

```python
class AblationConfig(Enum):
    NO_AI = auto()               # Disable all ML heuristics
    AI_STRUCTURAL = auto()       # AI + structural refinement only
    AI_MONOTONICITY = auto()     # AI + monotonicity refinement only
    AI_ALL_STRATEGIES = auto()   # AI + all refinement strategies
```

#### `AblationRunner`

Runs ablation experiments across configurations and circuits.

```python
class AblationRunner:
    def run(self, circuits, configurations) -> AblationSummary: ...
```

#### `AblationRunResult`

```python
@dataclass
class AblationRunResult:
    circuit: str
    configuration: str
    iterations: int
    time_s: float
    predicates: int
    converged: bool
    soundness_level: str
```

#### `AblationSummary`

```python
@dataclass
class AblationSummary:
    results: List[AblationRunResult]
```

<a id="extended-benchmarks"></a>
### Extended Benchmarks — `bioprover.evaluation.extended_benchmarks`

11 additional benchmark circuits covering enzymatic cascades (Michaelis-Menten),
allosteric regulation (MWC model), and multi-feedback oscillators (3–12 species).

#### `ExtendedBenchmarkSuite`

```python
class ExtendedBenchmarkSuite:
    @classmethod
    def get_all_circuits(cls) -> List[BenchmarkCircuit]: ...
    @classmethod
    def get_by_kinetics(cls, kinetics_type: str) -> List[BenchmarkCircuit]: ...
    @classmethod
    def get_by_topology(cls, topology_type: str) -> List[BenchmarkCircuit]: ...
    @classmethod
    def get_by_difficulty(cls, min_difficulty: BenchmarkDifficulty = BenchmarkDifficulty.EASY,
                          max_difficulty: BenchmarkDifficulty = BenchmarkDifficulty.FRONTIER
                          ) -> List[BenchmarkCircuit]: ...
    @classmethod
    def get_by_tags(cls, tags: Sequence[str]) -> List[BenchmarkCircuit]: ...
    @classmethod
    def get_combined_suite(cls) -> List[BenchmarkCircuit]: ...
    @classmethod
    def coverage_summary(cls) -> Dict[str, Any]: ...
```

```python
from bioprover.evaluation.extended_benchmarks import ExtendedBenchmarkSuite

# List all extended circuits
for c in ExtendedBenchmarkSuite.get_all_circuits():
    print(f"{c.name}: {c.category}, {c.difficulty.name}")

# Filter by kinetics type
mm_circuits = ExtendedBenchmarkSuite.get_by_kinetics("michaelis_menten")

# Get combined suite (original + extended)
full_suite = ExtendedBenchmarkSuite.get_combined_suite()
```

#### Extended Circuit Generators

| Function | Kinetics | Species | Description |
|----------|----------|---------|-------------|
| `enzymatic_cascade_mm()` | Michaelis-Menten | 6 | Three-enzyme cascade with substrate/product |
| `competitive_inhibition_mm()` | Michaelis-Menten | 5 | Competitive inhibition of enzymatic reaction |
| `substrate_channeling_mm()` | Michaelis-Menten | 7 | Substrate channeling between enzymes |
| `mwc_allosteric_switch()` | MWC | 4 | Monod-Wyman-Changeux allosteric switch |
| `allosteric_transcription_factor()` | MWC | 6 | Allosteric TF with ligand binding |
| `dual_feedback_oscillator()` | Hill | 4 | Positive + negative feedback oscillator |
| `three_node_competitive_network()` | Hill | 6 | Three-node competitive inhibition |
| `iffl_adaptation()` | Hill | 3 | Incoherent feed-forward loop adaptation |
| `repressilator_with_reporters()` | Hill | 6 | Repressilator with fluorescent reporters |
| `signaling_cascade_10()` | Hill/MM | 10 | 10-species signaling cascade |
| `metabolic_pathway_regulated()` | MM/Hill | 12 | Regulated metabolic pathway |

---

<a id="visualization"></a>
## 15. Visualization — `bioprover.visualization`

### `ResultExporter`

```python
class ResultExporter:
    def __init__(self, export_format: ExportFormat = ExportFormat.TEXT) -> None: ...
    def export(self, result, output_path: str) -> None: ...
    def export_latex(self, results: List[VerificationResult]) -> str: ...
    def export_html(self, results, model_name: str = "", path: str = "report.html") -> None: ...
    def summary_report(self, results: List[VerificationResult]) -> str: ...
```

### `ExportFormat` (Enum)

`JSON`, `CSV`, `LATEX`, `HTML`, `TEXT`

### `CounterexampleVisualizer`

```python
class CounterexampleVisualizer:
    def __init__(self, width: int = 80, height: int = 25) -> None: ...
    def plot_trace_ascii(self, cex: Counterexample) -> str: ...
    def generate_text_report(self, cex, result) -> str: ...
```

### `ProgressReporter`

Real-time CEGAR iteration display.

```python
class ProgressReporter:
    def __init__(self, verbosity: int = 1) -> None: ...
    def update(self, iteration: int, stats: IterationStats) -> None: ...
    def finish(self, result: VerificationResult) -> None: ...
```

---

<a id="cli"></a>
## 16. CLI — `bioprover.cli`

Entry point: `bioprover.cli:main`

### `bioprover verify`

```
bioprover verify -m MODEL -s SPEC [--mode MODE] [--timeout T] [--format FMT] [-o PATH]
```

| Flag | Description |
|------|-------------|
| `-m, --model` | Path to model file (SBML or `.sbol`). Auto-detects SBOL by extension. |
| `-s, --spec` | Bio-STL specification (string or `.stl` file). |
| `--mode` | `full`, `bounded`, or `compositional`. |
| `--timeout` | Wall-clock timeout in seconds (default: 300). |
| `--format` | Output format: `text`, `json`, `csv`, `latex`, `html`. |
| `-o, --output` | Output file path. |

### `bioprover synthesize`

```
bioprover synthesize -m MODEL -s SPEC [--objective OBJ] [--timeout T] [-o PATH]
```

| Flag | Description |
|------|-------------|
| `--objective` | `feasibility`, `robustness`, or `minimal`. |

### `bioprover repair`

```
bioprover repair -m MODEL -s SPEC [--budget B] [--timeout T] [-o PATH]
```

| Flag | Description |
|------|-------------|
| `--budget` | Maximum fractional perturbation per parameter (0–1, default: 0.5). |

### `bioprover benchmark`

```
bioprover benchmark [--suite SUITE] [--baselines B1 B2 ...] [--format FMT] [-o PATH]
```

| Flag | Description |
|------|-------------|
| `--suite` | `toggle_switch`, `repressilator`, or `full`. |
| `--baselines` | Tools to compare: `bioprover`, `dreach`, `flow_star`, `spaceex`. |

### `bioprover info`

```
bioprover info -m MODEL [-s SPEC]
```

Prints model summary: species, reactions, parameters, compartments, and
optional specification parse.

### Global Flags

| Flag | Description |
|------|-------------|
| `-v` | Verbosity (`-v`=INFO, `-vv`=DEBUG, `-vvv`=TRACE). |
| `-q, --quiet` | Suppress progress output. |
| `--no-color` | Disable colour output. |
| `-c, --config` | Configuration file (JSON or YAML). |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Verified / success |
| 1 | Falsified (counterexample found) |
| 2 | Unknown (timeout or inconclusive) |
| 3 | Error (invalid input, crash) |

### Module-Level Functions

```python
def build_parser() -> argparse.ArgumentParser: ...
def load_config(path: str) -> Dict[str, Any]: ...
def load_model(path: str) -> BioModel: ...
def load_spec(spec_arg: str) -> STLFormula: ...
def main(argv: Optional[List[str]] = None) -> int: ...
```

---

## 17. Certificate Verifier — `bioprover.certificate_verifier`

Standalone certificate verification module (~800 LoC). Does NOT import
from `bioprover.solver`, Z3, dReal, or SymPy.

### `CertificateVerifier`

```python
from bioprover.certificate_verifier import CertificateVerifier

verifier = CertificateVerifier()
report = verifier.verify(certificate_dict)
print(report.passed, report.failed, report.warnings)
print(report.accepted)  # True if failed == 0
```

### Verifier classes

| Class | Purpose |
|-------|---------|
| `FlowpipeReplayVerifier` | Replays integration with independent Euler; checks enclosure containment |
| `InvariantReplayVerifier` | Checks invariant satisfaction over interval boxes |
| `ErrorBudgetVerifier` | Validates soundness level vs. error budget consistency |
| `CompositionalVerifier` | Checks module interface compatibility |

### `VInterval` / `VBox`

Independent interval arithmetic (verifier-only, does not share code with solver):

```python
from bioprover.certificate_verifier.verifier import VInterval, VBox

iv = VInterval(1.0, 2.0)
print(iv.contains(1.5))   # True
print(iv.width())         # 1.0

box = VBox([VInterval(0, 1), VInterval(0, 1)])
print(box.contains_point([0.5, 0.5]))  # True
```

### CLI

```bash
python -m bioprover.certificate_verifier.cli certificate.json
```

---

## 18. Error Propagation — `bioprover.soundness`

### `ErrorSource`

```python
from bioprover.soundness import ErrorSource

src = ErrorSource(
    name="discretization",
    magnitude=0.01,
    origin="ODE integration",
    lipschitz_factor=2.0,  # dynamics amplification
)
print(src.effective_magnitude)  # 0.02
```

### `ErrorBudget` (extended)

```python
from bioprover.soundness import ErrorBudget

budget = ErrorBudget(
    delta=0.001,            # SMT perturbation
    epsilon=0.0,            # CEGIS tolerance
    truncation=0.05,        # moment closure
    discretization=0.01,    # ODE integration
)
print(budget.combined)           # RSS bound
print(budget.combined_additive)  # additive bound
print(budget.is_sound)           # True if all finite
```

### Key functions

| Function | Description |
|----------|-------------|
| `propagate_errors_with_lipschitz(sources)` | Lipschitz-amplified error composition |
| `compute_moment_closure_bound(n, k, N, L_f)` | Theorem 4 truncation bound |
| `compute_discretization_bound(h, p, L, T, C)` | Grönwall discretization bound |

---

<a id="bio-stl-templates"></a>
## 19. Bio-STL Templates — `bioprover.spec.templates`

15 reusable Bio-STL specification templates covering common biological circuit
behaviours.

### `TemplateLibrary`

```python
class TemplateLibrary:
    def __init__(self, *, load_builtins: bool = True) -> None: ...
    def register(self, template: SpecificationTemplate) -> None: ...
    def get(self, name: str) -> Optional[SpecificationTemplate]: ...
    @property
    def names(self) -> List[str]: ...
    @property
    def all_templates(self) -> List[SpecificationTemplate]: ...
    def search_by_category(self, category: str) -> List[SpecificationTemplate]: ...
    def search_by_keyword(self, keyword: str) -> List[SpecificationTemplate]: ...
    @staticmethod
    def compose(*specs: STLFormula) -> STLFormula: ...
    def documentation(self) -> str: ...
```

```python
from bioprover.spec.templates import TemplateLibrary

lib = TemplateLibrary()
tmpl = lib.get("sustained_oscillation")
formula = tmpl.instantiate(species="lacI", period_min=20, period_max=60,
                           amplitude_high=8.0, amplitude_low=2.0)
```

### `SpecificationTemplate`

```python
@dataclass
class SpecificationTemplate:
    name: str
    description: str
    parameters: List[TemplateParameter]
    builder: Optional[Callable]
    category: str
    notes: str
    def instantiate(self, **kwargs) -> STLFormula: ...
    @property
    def parameter_names(self) -> List[str]: ...
    def documentation(self) -> str: ...
```

### Built-in Templates

| # | Name | Category | Description |
|---|------|----------|-------------|
| 1 | `correct_boolean_logic` | logic | Genetic gate implements correct Boolean logic (NOT, AND, OR) |
| 2 | `oscillation` | dynamic | Sustained periodic oscillation with period and amplitude |
| 3 | `bistability` | memory | Two stable steady states with switching |
| 4 | `adaptation` | dynamic | Perfect adaptation: transient response then return to baseline |
| 5 | `monotone_dose_response` | transfer_function | Monotone input-output relationship |
| 6 | `pulse_generation` | dynamic | Transient pulse: spike above level then return |
| 7 | `steady_state_convergence` | stability | SS[T,ε](φ): convergence to target within tolerance band |
| 8 | `rise_time` | performance | Output reaches threshold within specified rise time |
| 9 | `overshoot_constraint` | performance | Output must not exceed maximum concentration |
| 10 | `separation` | memory | Two signals maintain minimum separation (toggle-switch) |
| 11 | `damped_oscillation` | dynamic | Successive peak amplitudes decrease by decay factor α |
| 12 | `sustained_oscillation` | dynamic | G[T,T_end](F[0,P](x>A_hi) ∧ F[0,P](x<A_lo)) |
| 13 | `probability_threshold` | stochastic | P(G[0,T](x > θ)) ≥ p_min |
| 14 | `bimodal_steady_state` | stochastic | G[T,T_end](|x−m1|<sep/2 ∨ |x−m2|<sep/2) |
| 15 | `switching_rate` | stochastic | Bounded noise-induced switching between bistable states |
