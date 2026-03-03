# BioProver — Formal Verification for Synthetic Biology Circuits

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Z3](https://img.shields.io/badge/solver-Z3%204.8%2B-orange)
![Version](https://img.shields.io/badge/version-0.1.0-lightgrey)

BioProver is a **CEGAR-based formal verification, repair, and synthesis** tool for synthetic biology circuits. It accepts **SBML**, **SBOL** (v2/v3), **GenBank** (.gb/.gbk), and **programmatic BioModel** definitions. Compositional verification via assume-guarantee reasoning scales to **50+ species**, far beyond the ~5 species limit of monolithic approaches. The tool includes **Bio-STL temporal specifications** (15 templates), **Linear Noise Approximation** with automatic bimodality detection for stochastic analysis, and an **online ML-guided predicate selector** with out-of-distribution detection for efficient CEGAR refinement.

---

## Key Features

- **CEGAR verification loop** — counterexample-guided abstraction refinement with sound ODE/LNA backends
- **Bio-STL temporal logic specifications** — 15 ready-to-use templates (sustained oscillation, steady-state convergence, damped oscillation, probabilistic threshold, noise-induced switching, and more)
- **Compositional assume-guarantee verification** — scales to 50+ species; speedups from 24× to >175×
- **SBOL v2/v3 import** — auto-mapping of genetic parts (promoters, RBSs, CDSs, terminators) to kinetics
- **GenBank import** — extract annotated DNA features (CDS, promoter, terminator, regulatory) from `.gb`/`.gbk` files
- **SBML import** — load standard systems biology models directly
- **Parameter synthesis** — feasibility, robustness, or minimal-perturbation objectives via CEGIS
- **Parameter repair** — fix failing circuits within a perturbation budget
- **Linear Noise Approximation (LNA)** — rigorous O(1/Ω) stochastic bounds with bimodality detection
- **Online ML predicate predictor** — learns from CEGAR traces with OOD detection
- **Standalone certificate verifier** — ~800 LoC trusted computing base (no Z3 dependency)
- **AG soundness proofs** — 3 theorems: AG Composition, Robustness Composition, Circular AG Convergence
- **21 benchmark circuits** — toggle switches, repressilators, gates, cascades, enzymatic (MM), allosteric (MWC)
- **Multiple output formats** — JSON, CSV, LaTeX, HTML, and plain text

---

## Installation

```bash
pip install -e .           # core dependencies
pip install -e ".[dev]"    # + pytest, mypy, ruff
pip install -e ".[viz]"    # + matplotlib
```

### Dependencies

| Required         | Version  | Optional              | Version  |
| ---------------- | -------- | --------------------- | -------- |
| Python           | ≥ 3.9   | matplotlib            | ≥ 3.4   |
| numpy            | ≥ 1.21  | pytest                | ≥ 7.0   |
| scipy            | ≥ 1.7   | pytest-cov            | ≥ 3.0   |
| sympy            | ≥ 1.9   | mypy                  | ≥ 0.950 |
| networkx         | ≥ 2.6   | ruff                  | ≥ 0.1   |
| z3-solver        | ≥ 4.8   | dReal binary (opt.)   | —       |

---

## Quickstart (30 seconds)

```python
from bioprover.models.bio_model import BioModel
from bioprover.models.species import Species
from bioprover.models.reactions import (
    Reaction, HillRepression, LinearDegradation, StoichiometryEntry as SE,
)
from bioprover import verify

model = BioModel("toggle_switch")
model.add_species(Species("gene_u", initial_concentration=10.0))
model.add_species(Species("gene_v", initial_concentration=0.1))
model.add_species(Species("reporter", initial_concentration=0.0))

model.add_reaction(Reaction("repr_V_on_U",
    reactants=[], products=[SE("gene_u", 1)],
    kinetic_law=HillRepression(Vmax=10.0, K=2.0, n=2), modifiers=["gene_v"]))
model.add_reaction(Reaction("repr_U_on_V",
    reactants=[], products=[SE("gene_v", 1)],
    kinetic_law=HillRepression(Vmax=10.0, K=2.0, n=2), modifiers=["gene_u"]))
model.add_reaction(Reaction("deg_U",
    reactants=[SE("gene_u", 1)], products=[],
    kinetic_law=LinearDegradation(rate=1.0)))
model.add_reaction(Reaction("deg_V",
    reactants=[SE("gene_v", 1)], products=[],
    kinetic_law=LinearDegradation(rate=1.0)))

result = verify(model, "G[0,100](gene_u > 1.0)")
print(result.status)  # => VerificationStatus.VERIFIED
```

---

## Full CLI Reference

### `bioprover verify`

| Flag        | Short | Description                                    | Default |
| ----------- | ----- | ---------------------------------------------- | ------- |
| `--model`   | `-m`  | Model file (SBML `.xml`, `.sbol`, or `.gb`/`.gbk`) | —       |
| `--spec`    | `-s`  | Bio-STL spec (inline string or file)           | —       |
| `--mode`    |       | `full` or `compositional`                      | `full`  |
| `--timeout` | `-t`  | Timeout in seconds                             | `300`   |
| `--output`  | `-o`  | Output file path                               | stdout  |
| `--format`  | `-f`  | `json\|csv\|latex\|html\|text`                 | `text`  |

```bash
bioprover verify -m circuit.xml -s "G[0,100](protein_A > 0.5)" --format json
```

### `bioprover synthesize`

| Flag          | Short | Description                        | Default       |
| ------------- | ----- | ---------------------------------- | ------------- |
| `--model`     | `-m`  | Model file                         | —             |
| `--spec`      | `-s`  | Bio-STL specification              | —             |
| `--objective` |       | `feasibility\|robustness\|minimal` | `feasibility` |
| `--timeout`   | `-t`  | Timeout in seconds                 | `600`         |
| `--output`    | `-o`  | Output file path                   | stdout        |

```bash
bioprover synthesize -m circuit.xml -s spec.biostl --objective robustness
```

### `bioprover repair`

| Flag        | Short | Description                           | Default |
| ----------- | ----- | ------------------------------------- | ------- |
| `--model`   | `-m`  | Model file                            | —       |
| `--spec`    | `-s`  | Bio-STL specification                 | —       |
| `--budget`  | `-b`  | Max fractional parameter perturbation | `0.5`   |
| `--timeout` | `-t`  | Timeout in seconds                    | `600`   |
| `--output`  | `-o`  | Output file path                      | stdout  |

```bash
bioprover repair -m broken.xml -s "G[0,50](gene_u > 1.0)" --budget 0.3
```

### `bioprover benchmark`

| Flag         | Short | Description                                             | Default     |
| ------------ | ----- | ------------------------------------------------------- | ----------- |
| `--suite`    |       | `toggle_switch\|repressilator\|full`                    | `full`      |
| `--baselines`|       | `bioprover\|dreach\|flow_star\|spaceex` (comma-sep)     | `bioprover` |
| `--output`   | `-o`  | Output file path                                        | stdout      |
| `--format`   | `-f`  | Output format                                           | `csv`       |

```bash
bioprover benchmark --suite full --baselines bioprover,dreach --format latex
```

### `bioprover info`

| Flag      | Short | Description              | Default |
| --------- | ----- | ------------------------ | ------- |
| `--model` | `-m`  | Model file (required)    | —       |
| `--spec`  | `-s`  | Bio-STL spec (optional)  | —       |

### `bioprover certificate_verifier`

```bash
python -m bioprover.certificate_verifier.cli certificate.json
```

### Global Flags

`--config/-c FILE` · `--verbose/-v` (`-v`/`-vv`/`-vvv`) · `--quiet/-q` · `--no-color` · `--version`

### Exit Codes

| Code | Meaning  |
| ---- | -------- |
| `0`  | Verified |
| `1`  | Falsified|
| `2`  | Unknown  |
| `3`  | Error    |

---

## Supported Input Formats

| Format                | Extension / Type       | Notes                                              |
| --------------------- | ---------------------- | -------------------------------------------------- |
| SBML                  | `.xml`                 | Systems Biology Markup Language                    |
| SBOL v2/v3            | `.sbol`                | Parts auto-mapped to kinetic models                |
| GenBank               | `.gb`, `.gbk`, `.genbank` | Annotated DNA features mapped to kinetic models |
| Programmatic BioModel | Python API             | `BioModel`, `Species`, `Reaction`                  |
| Bio-STL specification | Inline or `.biostl`    | Temporal logic specs with time bounds              |
| Configuration         | `.yaml` or `.json`     | Tool settings, solver parameters                   |

---

## Python API Overview

### Core functions

```python
from bioprover import verify, synthesize, repair

result = verify(model, spec="G[0,100](gene_u > 0.5)", mode="full", timeout=300)
synth  = synthesize(model, spec="G[0,100](gene_u > 0.5)", objective="robustness")
rep    = repair(model, spec="G[0,50](gene_u > 1.0)", budget=0.5)
```

### BioModel, Species, Reaction

```python
from bioprover.models.bio_model import BioModel
from bioprover.models.species import Species
from bioprover.models.reactions import (
    Reaction, HillRepression, HillActivation, LinearDegradation, MassAction,
    StoichiometryEntry as SE,
)

model = BioModel("my_circuit")
model.add_species(Species("protein_A", initial_concentration=5.0))
model.add_reaction(Reaction("prod_A",
    reactants=[], products=[SE("protein_A", 1)],
    kinetic_law=HillActivation(Vmax=8.0, K=1.5, n=2), modifiers=["inducer"]))
```

### Compositional verification — `verify_compositional`

```python
from bioprover.compositional.compositional_runner import verify_compositional
result = verify_compositional(model, spec_str, timeout=60.0, max_module_size=3)
```

### Importers — `SBOLImporter`, `SBMLImporter`, `GenBankImporter`

```python
from bioprover.models.sbol_import import parse_sbol_file
from bioprover.models.sbml_import import SBMLImporter
from bioprover.models.genbank_import import parse_genbank_file
model = parse_sbol_file("design.sbol")   # or SBMLImporter.load("model.xml")
model = parse_genbank_file("circuit.gb")
```

### TemplateLibrary — 15 Bio-STL templates

```python
from bioprover.spec.templates import TemplateLibrary
lib = TemplateLibrary()
spec = lib.get("sustained_oscillation").instantiate(
    species="gene_u", period_min=20.0, period_max=60.0,
    amplitude_high=8.0, amplitude_low=2.0)
```

### OnlineCEGARIntegration — ML-guided predicate selection

```python
from bioprover.ai.online_learner import OnlineCEGARIntegration
integration = OnlineCEGARIntegration(feature_dim=32)
predicate = integration.select_predicate(candidates, circuit_features)
integration.report_outcome(predicate, success=True)
```

### CertificateVerifier — standalone certificate verification

```python
from bioprover.certificate_verifier import CertificateVerifier
report = CertificateVerifier().verify(certificate_dict)
```

### ErrorBudget — error propagation

```python
from bioprover.soundness import ErrorBudget
budget = ErrorBudget(delta=0.001, discretization=0.01, truncation=0.05)
print(budget.combined)  # RSS bound
```

---

## Architecture Overview

```
  Model (SBML / SBOL / GenBank / BioModel)   Bio-STL Spec
          │                                  │
          ▼                                  ▼
  ┌───────────────────────────────────────────────────┐
  │                  CEGAR Engine                      │
  │  Abstraction ──▶ ODE/LNA ──▶ Bio-STL Check        │
  │       ▲                           │                │
  │       └──── Refinement (+ML) ◀────┘                │
  └───────────────┬──────────────┬────────────────────┘
                  │              │
          ┌───────┘     ┌───────┘
          ▼             ▼
    Verdict (V/F/U)   Certificate (.json)   Repair / Synthesis

  ── Compositional Branch ──
  Model ──▶ Decomposer ──▶ Module₁ ─┐
                           Module₂ ─┼──▶ AG Composer ──▶ Global Verdict
                           Module_n ─┘     (Thms 1–3)
```

---

## Bio-STL Specification Language

### Operators

| Operator       | Syntax           | Meaning                                     |
| -------------- | ---------------- | ------------------------------------------- |
| **Globally**   | `G[t1,t2](φ)`   | φ holds at every time in [t1, t2]           |
| **Eventually** | `F[t1,t2](φ)`   | φ holds at some time in [t1, t2]            |
| **Until**      | `φ U[t1,t2] ψ`  | φ holds until ψ becomes true in [t1, t2]    |
| **And / Or**   | `φ & ψ`, `φ \| ψ` | conjunction / disjunction                 |
| **Not**        | `!φ`             | negation                                    |

```
G[0,100](gene_u > 1.0)                            # always above 1.0
F[0,50](reporter > 5.0)                            # eventually reaches 5.0
G[0,100](gene_u > 1.0) & F[10,50](gene_v < 0.5)   # conjunction
```

### Template Library (15 templates)

| Template                    | Description                                  |
| --------------------------- | -------------------------------------------- |
| `sustained_oscillation`     | Repeated peaks within period/amplitude bounds|
| `damped_oscillation`        | Peaks decrease over time                     |
| `steady_state_convergence`  | Species converges to target value            |
| `monotone_increase`         | Species only increases over interval         |
| `monotone_decrease`         | Species only decreases over interval         |
| `bounded_concentration`     | Species stays within [lo, hi]                |
| `threshold_crossing`        | Species crosses a threshold                  |
| `mutual_exclusion`          | Two species never simultaneously high        |
| `sequential_activation`     | Species A peaks before species B             |
| `bistable_switch`           | System settles to one of two steady states   |
| `probabilistic_threshold`   | P(species > threshold) ≥ p                  |
| `bimodal_steady_state`      | Stochastic bimodal distribution detected     |
| `noise_induced_switching`   | Switching rate between attractors            |
| `pulse_response`            | Transient pulse after stimulus               |
| `adaptation`                | Returns to baseline after perturbation       |

---

## Examples

### Verify an SBOL circuit

```python
from bioprover.models.sbol_import import parse_sbol_file
from bioprover import verify

model = parse_sbol_file("examples/inverter_circuit.sbol")
result = verify(model, "G[0,100](GFP_protein > 0.0)")
print(result.status)  # VerificationStatus.VERIFIED
```

```bash
bioprover verify -m examples/inverter_circuit.sbol -s "G[0,100](GFP_protein > 0.0)"
```

### Compositional verification of a large circuit

```python
from bioprover.compositional.compositional_runner import verify_compositional

result = verify_compositional(large_model, "G[0,200](output > 0.1)",
    timeout=120.0, max_module_size=3)
print(result.status, result.n_modules, result.total_time)
```

| Species | Monolithic | Compositional | Speedup |
| ------- | ---------- | ------------- | ------- |
| 5       | 2.4 s      | 0.1 s         | 24×     |
| 8       | T/O        | 0.2 s         | >175×   |
| 20      | T/O        | 0.8 s         | >44×    |
| 50      | T/O        | 1.1 s         | >32×    |

### Parameter repair

```python
from bioprover import repair
result = repair(broken_model, "G[0,100](gene_u > 1.0)", budget=0.3, timeout=600)
print(result.repaired_model)
```

```bash
bioprover repair -m broken.xml -s "G[0,100](gene_u > 1.0)" -b 0.3
```

### Use Bio-STL templates

```python
from bioprover.spec.templates import TemplateLibrary
from bioprover import verify

spec = TemplateLibrary().get("steady_state_convergence").instantiate(
    species="gene_u", target=5.0, tolerance=0.5, settle_time=80.0, horizon=100.0)
result = verify(model, str(spec))
```

---

## FAQ / Troubleshooting

**Species naming — avoid `G`, `F`, `U`.**
These single letters are reserved Bio-STL operators (Globally, Eventually, Until). Use descriptive names like `gene_u`, `protein_GFP`, or `reporter`.

**When to use compositional mode?**
Whenever your model exceeds ~5 species. Monolithic CEGAR scales exponentially; compositional mode decomposes the circuit into modules of `max_module_size` species each, yielding 24×–175× speedups.

**What if verification times out?**
1. Switch to compositional mode. 2. Increase `--timeout`. 3. Shorten the time horizon in the specification. 4. Reduce `max_module_size` for faster per-module solving.

**How does the certificate verifier work?**
The standalone verifier (`python -m bioprover.certificate_verifier.cli cert.json`) is an ~800 LoC TCB that validates proof certificates without importing Z3 or solver modules — it checks internal consistency of every abstraction/refinement step.

**Can I use custom kinetic laws?**
Yes. Subclass `KineticLaw` and implement `rate(concentrations)`. The CEGAR engine will automatically discretize and encode it for SMT solving. Built-in laws: `HillRepression`, `HillActivation`, `LinearDegradation`, `MassAction`.

---

## License

MIT — see [LICENSE](LICENSE) for details.
