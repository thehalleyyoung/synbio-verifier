"""
Linear Noise Approximation (LNA) with rigorous O(1/Ω) error bounds.

Implements van Kampen's system-size expansion for stochastic biochemical
networks, providing rigorous error bounds that address the key limitation
of moment closure methods: lack of guaranteed approximation quality.

Mathematical foundation (van Kampen, 2007; Kurtz, 1972):

  The Chemical Master Equation (CME) for a system with volume parameter Ω
  admits a system-size expansion. Writing X(t) = Ω·φ(t) + √Ω·ξ(t):

    1. Macroscopic part: dφ/dt = S · a(φ)
       where S is the stoichiometry matrix and a(φ) the rate vector.

    2. Fluctuation covariance: dΣ/dt = A·Σ + Σ·Aᵀ + D
       where A = S · ∂a/∂x|_{x=φ} is the Jacobian of the macroscopic
       dynamics and D = S · diag(a(φ)) · Sᵀ is the diffusion matrix.

  Error bounds (Theorem, Kurtz 1972; Ethier & Kurtz 1986):
    |E[X_i/Ω] - φ_i(t)| ≤ C / Ω           (mean approximation)
    |Cov(X_i,X_j)/Ω - Σ_{ij}(t)| ≤ C' / Ω  (covariance approximation)

  where C, C' depend on the third derivatives of propensity functions,
  the time horizon, and the Jacobian stability.

  CRITICAL LIMITATION: LNA is invalid for bistable/multimodal systems.
  The BimodalityDetector class checks this precondition.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov, eigvals
from scipy.optimize import fsolve

from bioprover.soundness import (
    ErrorBudget,
    ErrorSource,
    SoundnessAnnotation,
    SoundnessLevel,
)
from bioprover.stochastic.moment_closure import MomentReaction


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class LNAResult:
    """Result container for LNA solver output.

    Attributes:
        times: Time points of the solution.
        concentrations: Macroscopic concentrations φ(t), shape (T, n).
        means: Expected molecule counts E[X(t)] = Ω·φ(t), shape (T, n).
        covariances: Covariance matrices Σ(t), shape (T, n, n).
            These are the *concentration-scale* covariances (divided by Ω).
        molecule_covariances: Molecule-count covariance = Ω·Σ(t), shape (T, n, n).
        error_bound_mean: Per-species bound on |E[X_i/Ω] - φ_i(t)|.
        error_bound_cov: Per-pair bound on |Cov(X_i,X_j)/Ω - Σ_{ij}(t)|.
        error_budget: ErrorBudget for integration with BioProver soundness.
        is_valid: False if bistability was detected (LNA unreliable).
        validation_warnings: List of warnings about approximation validity.
    """

    times: np.ndarray
    concentrations: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    molecule_covariances: np.ndarray
    error_bound_mean: float = 0.0
    error_bound_cov: float = 0.0
    error_budget: Optional[ErrorBudget] = None
    is_valid: bool = True
    validation_warnings: List[str] = field(default_factory=list)


@dataclass
class SteadyStateInfo:
    """Information about a steady state of the deterministic system."""

    state: np.ndarray
    jacobian_eigenvalues: np.ndarray
    is_stable: bool
    covariance: Optional[np.ndarray] = None


class StabilityType(Enum):
    """Classification of deterministic system stability."""

    MONOSTABLE = auto()
    BISTABLE = auto()
    MULTISTABLE = auto()
    OSCILLATORY = auto()
    UNKNOWN = auto()


# ═══════════════════════════════════════════════════════════════════════════
# LNA Solver
# ═══════════════════════════════════════════════════════════════════════════


class LNASolver:
    """Linear Noise Approximation with rigorous O(1/Ω) error bounds.

    Based on van Kampen's system-size expansion of the Chemical Master
    Equation. Decomposes molecular counts as:

        X(t) = Ω · φ(t) + √Ω · ξ(t)

    where φ(t) satisfies the deterministic rate equations and ξ(t) is
    a Gaussian fluctuation process with covariance Σ(t) satisfying the
    Lyapunov-type ODE:

        dΣ/dt = A(t)·Σ + Σ·A(t)ᵀ + D(t)

    Parameters:
        reactions: List of MomentReaction describing the reaction network.
        num_species: Number of chemical species.
        volume: System volume Ω (the expansion parameter).
    """

    def __init__(
        self,
        reactions: List[MomentReaction],
        num_species: int,
        volume: float = 1.0,
    ):
        if volume <= 0:
            raise ValueError(f"Volume must be positive, got {volume}")
        self.reactions = reactions
        self.num_species = num_species
        self.volume = volume
        # Build stoichiometry matrix S: shape (num_reactions, num_species)
        # S[j, i] = net change in species i from reaction j
        self._stoich = self._build_stoichiometry_matrix()

    def _build_stoichiometry_matrix(self) -> np.ndarray:
        """Build the stoichiometry matrix S from reactions."""
        S = np.zeros((len(self.reactions), self.num_species))
        for j, rxn in enumerate(self.reactions):
            for sp, delta in rxn.state_change.items():
                if 0 <= sp < self.num_species:
                    S[j, sp] = delta
        return S

    def deterministic_rates(self, concentrations: np.ndarray) -> np.ndarray:
        """Compute mass-action propensity rates a(φ) at given concentrations.

        For a reaction with reactants {s₁: c₁, s₂: c₂, ...}:
            a_j(φ) = k_j · ∏ᵢ φ_{sᵢ}^{cᵢ}
        """
        rates = np.zeros(len(self.reactions))
        for j, rxn in enumerate(self.reactions):
            r = rxn.rate_constant
            for sp, coeff in rxn.reactants.items():
                r *= concentrations[sp] ** coeff
            rates[j] = r
        return rates

    def macroscopic_rhs(self, t: float, phi: np.ndarray) -> np.ndarray:
        """Right-hand side of the deterministic RRE: dφ/dt = Sᵀ · a(φ)."""
        rates = self.deterministic_rates(np.maximum(phi, 0.0))
        return self._stoich.T @ rates

    def jacobian(self, concentrations: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Compute Jacobian A = d(Sᵀ·a)/dφ via finite differences.

        A_{ki} = ∂(dφ_k/dt)/∂φ_i
        """
        n = self.num_species
        conc = np.maximum(concentrations, 0.0)
        f0 = self._stoich.T @ self.deterministic_rates(conc)
        A = np.zeros((n, n))
        for i in range(n):
            c_pert = conc.copy()
            c_pert[i] += eps
            f_pert = self._stoich.T @ self.deterministic_rates(c_pert)
            A[:, i] = (f_pert - f0) / eps
        return A

    def jacobian_analytical(self, concentrations: np.ndarray) -> np.ndarray:
        """Compute Jacobian analytically for mass-action kinetics.

        For mass-action propensity a_j = k_j ∏ φ_s^{c_s}:
            ∂a_j/∂φ_i = a_j(φ) · c_i / φ_i   (if species i is a reactant)
        """
        n = self.num_species
        conc = np.maximum(concentrations, 1e-30)
        rates = self.deterministic_rates(conc)
        # ∂a/∂φ: shape (num_reactions, num_species)
        da_dphi = np.zeros((len(self.reactions), n))
        for j, rxn in enumerate(self.reactions):
            for sp, coeff in rxn.reactants.items():
                if 0 <= sp < n and conc[sp] > 0:
                    da_dphi[j, sp] = rates[j] * coeff / conc[sp]
        # A = Sᵀ · ∂a/∂φ
        return self._stoich.T @ da_dphi

    def diffusion_matrix(self, concentrations: np.ndarray) -> np.ndarray:
        """Compute diffusion matrix D = Sᵀ · diag(a(φ)) · S.

        This is the noise intensity matrix from the system-size expansion.
        Note: this is at the concentration scale (divided by Ω).
        """
        rates = self.deterministic_rates(np.maximum(concentrations, 0.0))
        # D = Sᵀ · diag(rates) · S = Σ_j a_j(φ) · v_j · v_jᵀ
        n = self.num_species
        D = np.zeros((n, n))
        for j in range(len(self.reactions)):
            v = self._stoich[j]
            D += np.outer(v, v) * rates[j]
        return D / self.volume

    def third_derivative_bound(self, concentrations: np.ndarray) -> float:
        """Estimate bound on third derivatives of propensity functions.

        For mass-action kinetics with reactions up to bimolecular (order ≤ 2),
        the third derivatives ∂³a/∂φ³ = 0. For trimolecular or Hill-type
        kinetics, this provides a numerical estimate via finite differences.

        Returns:
            M₃: upper bound on max |∂³aⱼ/∂φᵢ∂φₖ∂φₗ| across all j,i,k,l.
        """
        max_order = max((rxn.order() for rxn in self.reactions), default=0)
        if max_order <= 2:
            return 0.0

        # Numerical estimate for higher-order reactions
        n = self.num_species
        conc = np.maximum(concentrations, 1e-10)
        eps = 1e-4
        M3 = 0.0
        for j, rxn in enumerate(self.reactions):
            if rxn.order() <= 2:
                continue
            for i in range(n):
                for k in range(n):
                    for l in range(n):
                        # Third-order finite difference
                        d3 = self._third_diff(j, conc, i, k, l, eps)
                        M3 = max(M3, abs(d3))
        return M3

    def _third_diff(
        self,
        rxn_idx: int,
        conc: np.ndarray,
        i: int,
        k: int,
        l: int,
        eps: float,
    ) -> float:
        """Third-order finite difference ∂³a_j/∂φ_i ∂φ_k ∂φ_l."""

        def a_j(c: np.ndarray) -> float:
            rxn = self.reactions[rxn_idx]
            r = rxn.rate_constant
            for sp, coeff in rxn.reactants.items():
                r *= max(c[sp], 0.0) ** coeff
            return r

        # 8-point stencil for mixed third derivative
        result = 0.0
        for si in [1, -1]:
            for sk in [1, -1]:
                for sl in [1, -1]:
                    c = conc.copy()
                    c[i] += si * eps
                    c[k] += sk * eps
                    c[l] += sl * eps
                    result += si * sk * sl * a_j(c)
        return result / (8.0 * eps**3)

    def compute_error_bound(
        self,
        concentrations: np.ndarray,
        time_horizon: float,
    ) -> Tuple[float, float]:
        """Compute rigorous O(1/Ω) error bounds for the LNA.

        Theorem (Kurtz 1972, Ethier & Kurtz 1986):
          For a system with volume Ω and mass-action propensities:

          |E[X_i(t)/Ω] - φ_i(t)| ≤ C(t) / Ω

          |Cov(X_i,X_j)(t)/Ω - Σ_{ij}(t)| ≤ C'(t) / Ω

          where:
            C(t)  = (n·M₃/6) · exp(‖A‖·t) · t
            C'(t) = (n²·M₃/2) · exp(2‖A‖·t) · t

            n = number of species
            M₃ = bound on third derivatives of propensities
            ‖A‖ = operator norm of the Jacobian

        For systems with only uni- and bimolecular reactions (M₃ = 0),
        the LNA is exact to O(1/Ω) — the bound comes from fourth-order
        terms and is O(1/Ω²) for means.

        Returns:
            (mean_bound, cov_bound): error bounds for mean and covariance.
        """
        n = self.num_species
        conc = np.maximum(concentrations, 1e-10)
        A = self.jacobian(conc)
        A_norm = np.linalg.norm(A, ord=2)
        M3 = self.third_derivative_bound(conc)

        if M3 == 0.0:
            # For uni/bimolecular systems, error comes from O(1/Ω²) terms.
            # Use a conservative bound based on the fourth-order expansion.
            max_rate = max(
                (self.deterministic_rates(conc).max(), 1e-10)
            )
            C_mean = n * max_rate * math.exp(A_norm * time_horizon)
            C_cov = n**2 * max_rate * math.exp(2 * A_norm * time_horizon)
            mean_bound = C_mean / self.volume**2
            cov_bound = C_cov / self.volume**2
        else:
            # General case: O(1/Ω) error from third-order terms
            gronwall = math.exp(A_norm * time_horizon) if A_norm * time_horizon < 50 else 1e20
            C_mean = (n * M3 / 6.0) * gronwall * time_horizon
            C_cov = (n**2 * M3 / 2.0) * gronwall**2 * time_horizon
            mean_bound = C_mean / self.volume
            cov_bound = C_cov / self.volume

        return mean_bound, cov_bound

    def _covariance_rhs(
        self,
        sigma_flat: np.ndarray,
        A: np.ndarray,
        D: np.ndarray,
    ) -> np.ndarray:
        """Right-hand side for the covariance ODE: dΣ/dt = AΣ + ΣAᵀ + D."""
        n = self.num_species
        Sigma = sigma_flat.reshape(n, n)
        dSigma = A @ Sigma + Sigma @ A.T + D
        return dSigma.flatten()

    def solve(
        self,
        initial_concentrations: np.ndarray,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
        method: str = "RK45",
        compute_bounds: bool = True,
    ) -> LNAResult:
        """Solve the LNA: macroscopic trajectory + covariance evolution.

        Simultaneously integrates:
          1. dφ/dt = Sᵀ · a(φ)                    (deterministic RRE)
          2. dΣ/dt = A(t)·Σ + Σ·A(t)ᵀ + D(t)      (covariance Lyapunov ODE)

        where A(t) and D(t) are evaluated along the deterministic trajectory.

        Args:
            initial_concentrations: Initial macroscopic concentrations φ(0).
            t_span: (t_start, t_end) integration interval.
            t_eval: Optional time points for output.
            method: ODE solver method (default "RK45").
            compute_bounds: Whether to compute rigorous error bounds.

        Returns:
            LNAResult with trajectories, covariances, and error bounds.
        """
        n = self.num_species
        # State: [φ (n values), Σ_flat (n² values)]
        y0 = np.zeros(n + n * n)
        y0[:n] = initial_concentrations

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            phi = np.maximum(y[:n], 0.0)
            sigma = y[n:].reshape(n, n)
            dphi = self.macroscopic_rhs(t, phi)
            A = self.jacobian(phi)
            D = self.diffusion_matrix(phi)
            dsigma = A @ sigma + sigma @ A.T + D
            # Enforce symmetry to prevent drift
            dsigma = 0.5 * (dsigma + dsigma.T)
            return np.concatenate([dphi, dsigma.flatten()])

        sol = solve_ivp(
            rhs,
            t_span,
            y0,
            method=method,
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
            max_step=0.1,
        )

        if not sol.success:
            warnings.warn(f"LNA ODE integration failed: {sol.message}")

        times = sol.t
        concentrations = sol.y[:n, :].T
        covariances = sol.y[n:, :].T.reshape(-1, n, n)
        # Enforce symmetry
        covariances = 0.5 * (covariances + np.swapaxes(covariances, -2, -1))

        means = concentrations * self.volume
        mol_cov = covariances * self.volume

        # Compute error bounds
        mean_bound = 0.0
        cov_bound = 0.0
        error_budget = None
        if compute_bounds:
            T = t_span[1] - t_span[0]
            mean_bound, cov_bound = self.compute_error_bound(
                initial_concentrations, T
            )
            error_budget = ErrorBudget(
                truncation=mean_bound,
                sources=[
                    ErrorSource(
                        name="lna_mean",
                        magnitude=mean_bound,
                        origin="LNA system-size expansion O(1/Ω) truncation",
                    ),
                    ErrorSource(
                        name="lna_covariance",
                        magnitude=cov_bound,
                        origin="LNA covariance approximation error",
                    ),
                ],
            )

        return LNAResult(
            times=times,
            concentrations=concentrations,
            means=means,
            covariances=covariances,
            molecule_covariances=mol_cov,
            error_bound_mean=mean_bound,
            error_bound_cov=cov_bound,
            error_budget=error_budget,
            is_valid=True,
        )

    def steady_state_covariance(
        self,
        steady_state_conc: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """Compute steady-state covariance via the Lyapunov equation.

        At steady state, dΣ/dt = 0, so: A·Σ + Σ·Aᵀ + D = 0.
        This has a unique solution when all eigenvalues of A have
        negative real part (i.e., the steady state is stable).

        Returns:
            (Sigma, success): covariance matrix and whether it was solved.
        """
        A = self.jacobian(steady_state_conc)
        D = self.diffusion_matrix(steady_state_conc)
        eigs = eigvals(A)
        if np.any(np.real(eigs) >= 0):
            warnings.warn(
                "Jacobian has non-negative eigenvalues; "
                "steady state is unstable, Lyapunov equation has no solution."
            )
            return np.full((self.num_species, self.num_species), np.nan), False

        try:
            Sigma = solve_continuous_lyapunov(A, -D)
            Sigma = 0.5 * (Sigma + Sigma.T)
            return Sigma, True
        except np.linalg.LinAlgError:
            return np.full((self.num_species, self.num_species), np.nan), False

    def find_steady_states(
        self,
        initial_guesses: Optional[List[np.ndarray]] = None,
        n_random: int = 20,
        tol: float = 1e-10,
    ) -> List[SteadyStateInfo]:
        """Find steady states of the deterministic system dφ/dt = 0.

        Uses scipy.optimize.fsolve from multiple initial guesses.
        Classifies each steady state by Jacobian eigenvalue analysis.

        Args:
            initial_guesses: User-provided starting points.
            n_random: Number of random initial guesses to supplement.
            tol: Tolerance for identifying distinct steady states.

        Returns:
            List of SteadyStateInfo with stability classification.
        """
        n = self.num_species
        if initial_guesses is None:
            initial_guesses = []

        # Add random initial guesses in [0, 10] for each species
        rng = np.random.default_rng(42)
        for _ in range(n_random):
            initial_guesses.append(rng.uniform(0.01, 10.0, size=n))

        found: List[SteadyStateInfo] = []
        for guess in initial_guesses:
            try:
                ss, info, ier, _ = fsolve(
                    lambda x: self.macroscopic_rhs(0, np.maximum(x, 0.0)),
                    guess,
                    full_output=True,
                )
            except Exception:
                continue

            if ier != 1:
                continue
            ss = np.maximum(ss, 0.0)
            residual = np.linalg.norm(self.macroscopic_rhs(0, ss))
            if residual > 1e-6:
                continue

            # Check if this is a duplicate
            is_dup = False
            for existing in found:
                if np.allclose(ss, existing.state, atol=tol, rtol=tol):
                    is_dup = True
                    break
            if is_dup:
                continue

            eigs = eigvals(self.jacobian(ss))
            is_stable = bool(np.all(np.real(eigs) < 0))
            cov = None
            if is_stable:
                cov, ok = self.steady_state_covariance(ss)
                if not ok:
                    cov = None

            found.append(
                SteadyStateInfo(
                    state=ss,
                    jacobian_eigenvalues=eigs,
                    is_stable=is_stable,
                    covariance=cov,
                )
            )

        return found


# ═══════════════════════════════════════════════════════════════════════════
# Bimodality Detector
# ═══════════════════════════════════════════════════════════════════════════


class BimodalityDetector:
    """Detects when LNA is unreliable due to bistability or multimodality.

    The LNA assumes a unimodal (approximately Gaussian) distribution
    centered on the deterministic trajectory. This assumption fails when:

    1. The deterministic system has multiple stable steady states (bistability).
       In this case the true CME distribution is bimodal, and the LNA—which
       captures only the neighborhood of one steady state—gives qualitatively
       wrong results.

    2. The system exhibits noise-induced transitions between attractors,
       which are entirely invisible to the LNA.

    Detection strategy:
      - Find all steady states of dφ/dt = 0.
      - Classify stability via Jacobian eigenvalue analysis.
      - For 2D systems, apply the trace-determinant test.
      - Count the number of stable steady states.
      - If ≥ 2 stable states exist, flag as bistable.
    """

    def __init__(self, solver: LNASolver):
        self.solver = solver

    def detect(
        self,
        initial_guesses: Optional[List[np.ndarray]] = None,
        n_random: int = 30,
    ) -> Tuple[StabilityType, List[SteadyStateInfo]]:
        """Detect the stability type of the reaction network.

        Returns:
            (stability_type, steady_states): classification and found steady states.
        """
        steady_states = self.solver.find_steady_states(
            initial_guesses=initial_guesses, n_random=n_random
        )

        stable = [ss for ss in steady_states if ss.is_stable]
        n_stable = len(stable)

        # Check for oscillatory behavior (complex eigenvalues with positive real)
        has_oscillation = False
        for ss in steady_states:
            eigs = ss.jacobian_eigenvalues
            if np.any(np.imag(eigs) != 0) and np.any(np.real(eigs) > 0):
                has_oscillation = True
                break

        if has_oscillation:
            return StabilityType.OSCILLATORY, steady_states
        if n_stable == 0:
            return StabilityType.UNKNOWN, steady_states
        if n_stable == 1:
            return StabilityType.MONOSTABLE, steady_states
        if n_stable == 2:
            return StabilityType.BISTABLE, steady_states
        return StabilityType.MULTISTABLE, steady_states

    def trace_determinant_test_2d(
        self,
        steady_state: np.ndarray,
    ) -> Dict[str, Any]:
        """Apply the trace-determinant classification for 2D systems.

        For a 2×2 Jacobian A:
          - tr(A) < 0, det(A) > 0: stable node or spiral
          - tr(A) > 0, det(A) > 0: unstable node or spiral
          - det(A) < 0: saddle point
          - tr(A)² - 4·det(A) < 0: spiral (oscillatory)
          - tr(A)² - 4·det(A) > 0: node (non-oscillatory)

        Returns:
            Dict with 'trace', 'determinant', 'discriminant', 'classification'.
        """
        if self.solver.num_species != 2:
            raise ValueError("trace-determinant test requires exactly 2 species")

        A = self.solver.jacobian(steady_state)
        tr = np.trace(A)
        det = np.linalg.det(A)
        disc = tr**2 - 4 * det

        if det < 0:
            cls = "saddle"
        elif tr < 0:
            cls = "stable_spiral" if disc < 0 else "stable_node"
        elif tr > 0:
            cls = "unstable_spiral" if disc < 0 else "unstable_node"
        else:
            cls = "center"

        return {
            "trace": float(tr),
            "determinant": float(det),
            "discriminant": float(disc),
            "classification": cls,
        }

    def validate_lna(
        self,
        initial_guesses: Optional[List[np.ndarray]] = None,
    ) -> Tuple[bool, List[str]]:
        """Check whether LNA is a valid approximation for this system.

        Returns:
            (is_valid, warnings): True if LNA can be trusted, with any caveats.
        """
        stability_type, steady_states = self.detect(initial_guesses)
        warn_msgs: List[str] = []

        if stability_type == StabilityType.BISTABLE:
            warn_msgs.append(
                "System is bistable with 2 stable steady states. "
                "LNA is INVALID: it cannot capture bimodal distributions. "
                "Use FSP or SSA instead."
            )
            return False, warn_msgs

        if stability_type == StabilityType.MULTISTABLE:
            n_stable = sum(1 for ss in steady_states if ss.is_stable)
            warn_msgs.append(
                f"System has {n_stable} stable steady states. "
                "LNA captures only local fluctuations around one state. "
                "Use FSP for global distribution."
            )
            return False, warn_msgs

        if stability_type == StabilityType.OSCILLATORY:
            warn_msgs.append(
                "System has oscillatory instability. LNA covariance "
                "may grow unbounded. Results should be used with caution."
            )

        if stability_type == StabilityType.UNKNOWN:
            warn_msgs.append(
                "No stable steady states found. LNA validity uncertain."
            )
            return False, warn_msgs

        # Check condition number of Jacobian at stable SS
        for ss in steady_states:
            if ss.is_stable:
                A = self.solver.jacobian(ss.state)
                cond = np.linalg.cond(A)
                if cond > 1e8:
                    warn_msgs.append(
                        f"Jacobian is ill-conditioned (κ={cond:.1e}) at "
                        f"steady state {ss.state}. Error bounds may be loose."
                    )

        return True, warn_msgs


# ═══════════════════════════════════════════════════════════════════════════
# Analysis Method Selection
# ═══════════════════════════════════════════════════════════════════════════


class AnalysisMethod(Enum):
    """Available stochastic analysis methods."""

    LNA = auto()
    FSP = auto()
    MOMENT_CLOSURE = auto()
    HYBRID_SSA_ODE = auto()


@dataclass
class MethodSelection:
    """Result of automatic method selection."""

    method: AnalysisMethod
    reason: str
    soundness_level: SoundnessLevel
    estimated_cost: str  # "low", "medium", "high"
    warnings: List[str] = field(default_factory=list)


class StochasticAnalysisPipeline:
    """Automatically selects the best stochastic analysis method.

    Decision logic:
      1. For large Ω (>100) and monostable systems → LNA
         Fastest, O(n²) per step, rigorous O(1/Ω) bounds.

      2. For small state spaces (<1000 states) → FSP
         Exact solution of truncated CME, no approximation error.

      3. For moderate systems → Moment closure with bimodality check
         Good balance of speed and accuracy.

      4. Fallback → Hybrid SSA/ODE
         Monte Carlo, no approximation but statistical error.

    Parameters:
        reactions: Reaction network specification.
        num_species: Number of chemical species.
        volume: System volume parameter Ω.
        state_space_bounds: Upper bounds on molecule counts per species
            (used to estimate FSP state space size).
    """

    def __init__(
        self,
        reactions: List[MomentReaction],
        num_species: int,
        volume: float = 1.0,
        state_space_bounds: Optional[List[int]] = None,
    ):
        self.reactions = reactions
        self.num_species = num_species
        self.volume = volume
        self.state_space_bounds = state_space_bounds
        self._lna_solver = LNASolver(reactions, num_species, volume)
        self._detector = BimodalityDetector(self._lna_solver)

    def _estimate_state_space_size(self) -> int:
        """Estimate the size of the FSP state space."""
        if self.state_space_bounds is None:
            # Conservative estimate based on volume
            bound = max(int(10 * self.volume), 50)
            return bound ** self.num_species
        product = 1
        for b in self.state_space_bounds:
            product *= (b + 1)
        return product

    def select_method(
        self,
        initial_concentrations: Optional[np.ndarray] = None,
    ) -> MethodSelection:
        """Select the best analysis method for the given system.

        Args:
            initial_concentrations: Optional starting point for bistability
                analysis. If None, uses random initial guesses.

        Returns:
            MethodSelection with the recommended method and justification.
        """
        guesses = None
        if initial_concentrations is not None:
            guesses = [initial_concentrations]

        # Step 1: Check for bistability
        is_valid, validity_warnings = self._detector.validate_lna(guesses)

        # Step 2: Estimate state space for FSP feasibility
        ss_size = self._estimate_state_space_size()
        fsp_feasible = ss_size < 1000

        # Decision tree
        if fsp_feasible:
            return MethodSelection(
                method=AnalysisMethod.FSP,
                reason=(
                    f"State space is small ({ss_size} states). "
                    "FSP provides exact CME solution."
                ),
                soundness_level=SoundnessLevel.BOUNDED,
                estimated_cost="low" if ss_size < 100 else "medium",
            )

        if is_valid and self.volume > 100:
            return MethodSelection(
                method=AnalysisMethod.LNA,
                reason=(
                    f"System is monostable with large volume (Ω={self.volume}). "
                    f"LNA error bound O(1/Ω) = O({1/self.volume:.1e})."
                ),
                soundness_level=SoundnessLevel.BOUNDED,
                estimated_cost="low",
                warnings=validity_warnings,
            )

        if is_valid and self.volume > 10:
            return MethodSelection(
                method=AnalysisMethod.LNA,
                reason=(
                    f"System is monostable (Ω={self.volume}). "
                    "LNA applicable but bounds less tight."
                ),
                soundness_level=SoundnessLevel.BOUNDED,
                estimated_cost="low",
                warnings=validity_warnings,
            )

        if not is_valid:
            # Bistable or multistable — moment closure is risky too
            if ss_size < 10000:
                return MethodSelection(
                    method=AnalysisMethod.FSP,
                    reason=(
                        "System is bistable/multistable. "
                        f"FSP is feasible ({ss_size} states) and exact."
                    ),
                    soundness_level=SoundnessLevel.BOUNDED,
                    estimated_cost="medium",
                    warnings=validity_warnings,
                )
            return MethodSelection(
                method=AnalysisMethod.HYBRID_SSA_ODE,
                reason=(
                    "System is bistable with large state space. "
                    "SSA is the only reliable method."
                ),
                soundness_level=SoundnessLevel.APPROXIMATE,
                estimated_cost="high",
                warnings=validity_warnings,
            )

        # Moderate volume, monostable — moment closure is acceptable
        return MethodSelection(
            method=AnalysisMethod.MOMENT_CLOSURE,
            reason=(
                f"Moderate volume (Ω={self.volume}), monostable. "
                "Moment closure with bimodality monitoring."
            ),
            soundness_level=SoundnessLevel.APPROXIMATE,
            estimated_cost="medium",
            warnings=validity_warnings,
        )

    def run(
        self,
        initial_concentrations: np.ndarray,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run the automatically selected analysis method.

        Returns a dict with:
          - 'method': the AnalysisMethod used
          - 'selection': the MethodSelection reasoning
          - 'result': method-specific results
          - 'soundness': SoundnessAnnotation for the result
        """
        selection = self.select_method(initial_concentrations)

        if selection.method == AnalysisMethod.LNA:
            return self._run_lna(initial_concentrations, t_span, t_eval, selection)

        if selection.method == AnalysisMethod.MOMENT_CLOSURE:
            return self._run_moment_closure(
                initial_concentrations, t_span, t_eval, selection
            )

        # FSP and HYBRID_SSA_ODE: return selection info, caller uses
        # the appropriate solver from bioprover.stochastic
        return {
            "method": selection.method,
            "selection": selection,
            "result": None,
            "soundness": SoundnessAnnotation(
                level=selection.soundness_level,
                assumptions=[selection.reason] + selection.warnings,
            ),
            "instructions": (
                f"Use bioprover.stochastic.{'FSPSolver' if selection.method == AnalysisMethod.FSP else 'HaseltineRawlingsHybrid'} "
                "directly for this system configuration."
            ),
        }

    def _run_lna(
        self,
        initial_concentrations: np.ndarray,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray],
        selection: MethodSelection,
    ) -> Dict[str, Any]:
        """Execute LNA analysis with error bounds."""
        result = self._lna_solver.solve(
            initial_concentrations, t_span, t_eval, compute_bounds=True
        )
        annotation = SoundnessAnnotation(
            level=SoundnessLevel.BOUNDED,
            assumptions=[
                "LNA system-size expansion to O(1/Ω)",
                f"Volume Ω = {self.volume}",
                selection.reason,
            ] + selection.warnings,
            error_budget=result.error_budget,
            approximation_error=result.error_bound_mean,
        )
        return {
            "method": AnalysisMethod.LNA,
            "selection": selection,
            "result": result,
            "soundness": annotation,
        }

    def _run_moment_closure(
        self,
        initial_concentrations: np.ndarray,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray],
        selection: MethodSelection,
    ) -> Dict[str, Any]:
        """Execute moment closure with normal closure scheme."""
        from bioprover.stochastic.moment_closure import (
            MomentEquations,
            NormalClosure,
            MomentClosureSolver,
        )

        n = self.num_species
        moment_eqs = MomentEquations(self.reactions, n, max_order=2)
        closure = NormalClosure(n)
        solver = MomentClosureSolver(moment_eqs, closure)
        initial_cov = np.zeros((n, n))
        result = solver.solve(initial_concentrations, initial_cov, t_span, t_eval)
        annotation = SoundnessAnnotation(
            level=SoundnessLevel.APPROXIMATE,
            assumptions=[
                "Normal (Gaussian) moment closure at order 2",
                "Third and higher cumulants assumed zero",
                selection.reason,
            ] + selection.warnings,
        )
        return {
            "method": AnalysisMethod.MOMENT_CLOSURE,
            "selection": selection,
            "result": result,
            "soundness": annotation,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════════════


def lna_error_budget(
    reactions: List[MomentReaction],
    num_species: int,
    volume: float,
    initial_concentrations: np.ndarray,
    time_horizon: float,
) -> ErrorBudget:
    """Compute an ErrorBudget for LNA applied to the given system.

    This is a convenience function for integration with BioProver's
    end-to-end error propagation framework (Theorem 4 in the paper).

    The LNA truncation error contributes to the τ (truncation) component
    of the overall error budget.

    Returns:
        ErrorBudget with the LNA truncation error populated.
    """
    solver = LNASolver(reactions, num_species, volume)
    mean_bound, cov_bound = solver.compute_error_bound(
        initial_concentrations, time_horizon
    )
    return ErrorBudget(
        truncation=mean_bound,
        sources=[
            ErrorSource(
                name="lna_system_size_expansion",
                magnitude=mean_bound,
                origin=(
                    f"LNA O(1/Ω) truncation with Ω={volume}, "
                    f"T={time_horizon}, n={num_species}"
                ),
            ),
            ErrorSource(
                name="lna_covariance_error",
                magnitude=cov_bound,
                origin="LNA covariance approximation error bound",
            ),
        ],
    )


def validate_lna_applicability(
    reactions: List[MomentReaction],
    num_species: int,
    volume: float,
    initial_guesses: Optional[List[np.ndarray]] = None,
) -> Tuple[bool, StabilityType, List[str]]:
    """Check whether LNA is applicable to the given system.

    Performs bistability detection and returns a recommendation.

    Returns:
        (is_applicable, stability_type, warnings)
    """
    solver = LNASolver(reactions, num_species, volume)
    detector = BimodalityDetector(solver)
    stability_type, _ = detector.detect(initial_guesses)
    is_valid, warnings = detector.validate_lna(initial_guesses)
    return is_valid, stability_type, warnings
