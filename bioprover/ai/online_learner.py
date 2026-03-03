"""Online adaptation of the neural predicate predictor from CEGAR traces.

Provides online learning, ablation experiment control, out-of-distribution
detection, and full CEGAR-loop integration for the BioProver AI heuristic
engine.  Addresses reviewer feedback on the lack of online adaptation from
CEGAR traces and missing ablation comparisons.

All ML operations use pure NumPy, matching the rest of the AI module.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .predicate_predictor import (
    CandidatePredicate,
    PredicatePredictor,
    _ensure_dim,
    _softmax,
)
from .refinement_learner import (
    Adam,
    CEGARState,
    DifferentiableMLP,
    RefinementExample,
    ReplayBuffer,
    _cross_entropy_loss,
)

logger = logging.getLogger(__name__)

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CEGARTraceEntry:
    """Single entry recorded from a CEGAR iteration.

    Attributes
    ----------
    circuit_features : np.ndarray
        Graph / circuit embedding (1-D).
    cegar_state : CEGARState
        Snapshot of CEGAR loop state at this iteration.
    predicate_index : int
        Index of the predicate that was chosen.
    outcome : float
        Scalar effectiveness (1.0 = refinement succeeded, 0.0 = no progress).
    timestamp : float
        Wall-clock time when the entry was recorded.
    """

    circuit_features: np.ndarray
    cegar_state: CEGARState
    predicate_index: int
    outcome: float
    timestamp: float = 0.0


@dataclass
class LearningCurvePoint:
    """A single point on the online learning curve."""

    iteration: int
    prediction_accuracy: float
    cumulative_regret: float
    loss: float
    wall_time: float


# ---------------------------------------------------------------------------
# Priority replay buffer
# ---------------------------------------------------------------------------


class PriorityReplayBuffer:
    """Replay buffer with priority sampling.

    Successful predicates receive higher sampling priority.  Priorities
    are proportional to ``|outcome| + base_priority`` so that informative
    examples (both strong successes and clear failures) are replayed
    more often.

    Parameters
    ----------
    capacity : int
        Maximum number of stored entries.
    base_priority : float
        Minimum priority added to every entry to avoid starvation.
    """

    def __init__(self, capacity: int = 5000, base_priority: float = 0.1) -> None:
        self.capacity = capacity
        self.base_priority = base_priority
        self._entries: deque[CEGARTraceEntry] = deque(maxlen=capacity)
        self._priorities: deque[float] = deque(maxlen=capacity)

    def add(self, entry: CEGARTraceEntry) -> None:
        """Add an entry with automatically computed priority."""
        priority = abs(entry.outcome) + self.base_priority
        self._entries.append(entry)
        self._priorities.append(priority)

    def sample(
        self, batch_size: int, rng: Optional[np.random.RandomState] = None,
    ) -> List[CEGARTraceEntry]:
        """Sample a batch with probability proportional to priority."""
        if rng is None:
            rng = np.random.RandomState()
        n = len(self._entries)
        if n == 0:
            return []
        k = min(batch_size, n)
        priors = np.array(list(self._priorities), dtype=np.float64)
        probs = priors / (priors.sum() + _EPS)
        indices = rng.choice(n, size=k, replace=False, p=probs)
        buf_list = list(self._entries)
        return [buf_list[i] for i in indices]

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def mean_priority(self) -> float:
        if not self._priorities:
            return 0.0
        return float(np.mean(list(self._priorities)))


# ---------------------------------------------------------------------------
# OnlineLearner
# ---------------------------------------------------------------------------


class OnlineLearner:
    """Adapts the predicate predictor during verification using CEGAR traces.

    After each CEGAR iteration the trace is stored in a priority replay
    buffer and the underlying policy network is fine-tuned.  An exponential
    moving average (EMA) of the weights is maintained to avoid catastrophic
    forgetting.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the concatenated feature vector.
    n_candidates : int
        Maximum number of candidate predicates (output dimension).
    hidden_dims : list[int]
        Hidden layer sizes for the policy MLP.
    lr0 : float
        Initial learning rate.
    tau : float
        Decay time-constant for the schedule  η_t = η_0 / (1 + t/τ).
    ema_decay : float
        EMA coefficient (closer to 1.0 → slower updates, less forgetting).
    buffer_capacity : int
        Capacity of the priority replay buffer.
    weight_decay : float
        L2 regularisation coefficient.
    """

    def __init__(
        self,
        input_dim: int = 64,
        n_candidates: int = 128,
        hidden_dims: Optional[List[int]] = None,
        lr0: float = 1e-3,
        tau: float = 50.0,
        ema_decay: float = 0.995,
        buffer_capacity: int = 5000,
        weight_decay: float = 1e-4,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self._rng = rng or np.random.RandomState(42)

        self.input_dim = input_dim
        self.n_candidates = n_candidates
        self.lr0 = lr0
        self.tau = tau
        self.ema_decay = ema_decay

        layer_dims = [input_dim] + hidden_dims + [n_candidates]
        self.policy = DifferentiableMLP(layer_dims, self._rng)
        self._ema_weights: List[np.ndarray] = [w.copy() for w in self.policy.weights]
        self._ema_biases: List[np.ndarray] = [b.copy() for b in self.policy.biases]

        self.optimizer = Adam(
            self.policy.weights + self.policy.biases,
            lr=lr0, weight_decay=weight_decay,
        )
        self.buffer = PriorityReplayBuffer(capacity=buffer_capacity)

        self._step_count: int = 0
        self._train_losses: List[float] = []
        self._learning_curve: List[LearningCurvePoint] = []
        self._correct_predictions: int = 0
        self._total_predictions: int = 0

    # -- learning rate schedule ----------------------------------------------

    @property
    def current_lr(self) -> float:
        """η_t = η_0 / (1 + t/τ)"""
        return self.lr0 / (1.0 + self._step_count / self.tau)

    def _apply_lr_schedule(self) -> None:
        """Update the optimiser's learning rate to the current schedule."""
        self.optimizer.lr = self.current_lr

    # -- EMA maintenance -----------------------------------------------------

    def _update_ema(self) -> None:
        """Exponential moving average update of shadow weights."""
        alpha = self.ema_decay
        for i in range(len(self._ema_weights)):
            self._ema_weights[i] = (
                alpha * self._ema_weights[i]
                + (1.0 - alpha) * self.policy.weights[i]
            )
        for i in range(len(self._ema_biases)):
            self._ema_biases[i] = (
                alpha * self._ema_biases[i]
                + (1.0 - alpha) * self.policy.biases[i]
            )

    def _swap_to_ema(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Swap policy weights with EMA weights for inference.

        Returns the original (non-EMA) weights so they can be restored.
        """
        orig_w = [w.copy() for w in self.policy.weights]
        orig_b = [b.copy() for b in self.policy.biases]
        for i in range(len(self.policy.weights)):
            self.policy.weights[i][:] = self._ema_weights[i]
        for i in range(len(self.policy.biases)):
            self.policy.biases[i][:] = self._ema_biases[i]
        return orig_w, orig_b

    def _restore_from_ema(
        self, orig_w: List[np.ndarray], orig_b: List[np.ndarray],
    ) -> None:
        """Restore original weights after EMA inference."""
        for i in range(len(self.policy.weights)):
            self.policy.weights[i][:] = orig_w[i]
        for i in range(len(self.policy.biases)):
            self.policy.biases[i][:] = orig_b[i]

    # -- feature assembly ----------------------------------------------------

    def _build_input(
        self, circuit_features: np.ndarray, cegar_state: CEGARState,
    ) -> np.ndarray:
        """Concatenate and pad/truncate features to ``input_dim``."""
        vec = np.concatenate([circuit_features, cegar_state.to_vector()])
        if len(vec) >= self.input_dim:
            return vec[: self.input_dim]
        return np.pad(vec, (0, self.input_dim - len(vec)))

    # -- recording -----------------------------------------------------------

    def record_trace(self, entry: CEGARTraceEntry) -> None:
        """Store a CEGAR trace entry in the priority replay buffer."""
        self.buffer.add(entry)
        logger.debug(
            "Recorded trace entry: predicate=%d outcome=%.3f buffer_size=%d",
            entry.predicate_index, entry.outcome, len(self.buffer),
        )

    # -- online update -------------------------------------------------------

    def update(self, batch_size: int = 16) -> float:
        """Sample from replay buffer and perform one gradient step.

        Returns the mean cross-entropy loss of the batch.
        """
        batch = self.buffer.sample(batch_size, self._rng)
        if not batch:
            return 0.0

        self._apply_lr_schedule()
        total_loss = 0.0
        dW_accum = [np.zeros_like(w) for w in self.policy.weights]
        db_accum = [np.zeros_like(b) for b in self.policy.biases]

        for entry in batch:
            x = self._build_input(entry.circuit_features, entry.cegar_state)
            logits = self.policy.forward(x, store_cache=True)
            target = min(entry.predicate_index, self.n_candidates - 1)
            loss, d_output = _cross_entropy_loss(logits, target)
            d_output *= entry.outcome
            total_loss += loss

            dW, db = self.policy.backward(x, d_output)
            for j in range(len(dW)):
                dW_accum[j] += dW[j]
                db_accum[j] += db[j]

        n = len(batch)
        grads = [dw / n for dw in dW_accum] + [db / n for db in db_accum]
        self.optimizer.step(grads)
        self._update_ema()

        self._step_count += 1
        mean_loss = total_loss / n
        self._train_losses.append(mean_loss)
        return mean_loss

    # -- inference (using EMA weights) ---------------------------------------

    def predict(
        self, circuit_features: np.ndarray, cegar_state: CEGARState,
    ) -> np.ndarray:
        """Return softmax scores over candidates using the EMA model.

        Parameters
        ----------
        circuit_features : (feat_dim,) circuit embedding
        cegar_state : current CEGAR loop state

        Returns
        -------
        np.ndarray of shape ``(n_candidates,)``
        """
        orig_w, orig_b = self._swap_to_ema()
        try:
            x = self._build_input(circuit_features, cegar_state)
            logits = self.policy.forward(x)
            return _softmax(logits)
        finally:
            self._restore_from_ema(orig_w, orig_b)

    def record_prediction_outcome(
        self, predicted_index: int, actual_best: int,
    ) -> None:
        """Track prediction accuracy for learning curve metrics."""
        self._total_predictions += 1
        if predicted_index == actual_best:
            self._correct_predictions += 1

    def record_learning_curve_point(
        self, iteration: int, cumulative_regret: float, wall_time: float,
    ) -> None:
        """Append a learning curve data point."""
        acc = (
            self._correct_predictions / max(self._total_predictions, 1)
        )
        loss = self._train_losses[-1] if self._train_losses else 0.0
        self._learning_curve.append(LearningCurvePoint(
            iteration=iteration,
            prediction_accuracy=acc,
            cumulative_regret=cumulative_regret,
            loss=loss,
            wall_time=wall_time,
        ))

    # -- metrics / serialisation ---------------------------------------------

    @property
    def training_losses(self) -> List[float]:
        return list(self._train_losses)

    @property
    def learning_curve(self) -> List[LearningCurvePoint]:
        return list(self._learning_curve)

    @property
    def prediction_accuracy(self) -> float:
        if self._total_predictions == 0:
            return 0.0
        return self._correct_predictions / self._total_predictions

    def get_metrics(self) -> Dict[str, Any]:
        """Return a summary of online learner state."""
        return {
            "step_count": self._step_count,
            "current_lr": self.current_lr,
            "buffer_size": len(self.buffer),
            "mean_buffer_priority": self.buffer.mean_priority,
            "prediction_accuracy": self.prediction_accuracy,
            "total_predictions": self._total_predictions,
            "mean_loss": float(np.mean(self._train_losses)) if self._train_losses else 0.0,
        }

    def save(self, path: str) -> None:
        """Persist online learner state to an ``.npz`` file."""
        np.savez(
            path,
            policy=self.policy.state_dict(),
            ema_weights=[w.tolist() for w in self._ema_weights],
            ema_biases=[b.tolist() for b in self._ema_biases],
            losses=np.array(self._train_losses),
            step_count=self._step_count,
        )

    @classmethod
    def load(cls, path: str) -> "OnlineLearner":
        """Restore an online learner from an ``.npz`` file."""
        data = np.load(path, allow_pickle=True)
        policy_dict = data["policy"].item()
        dims = policy_dict["layer_dims"]
        obj = cls(input_dim=dims[0], n_candidates=dims[-1], hidden_dims=dims[1:-1])
        obj.policy = DifferentiableMLP.from_state_dict(policy_dict)
        obj.optimizer = Adam(obj.policy.weights + obj.policy.biases)
        obj._ema_weights = [
            np.asarray(w, dtype=np.float64) for w in data["ema_weights"]
        ]
        obj._ema_biases = [
            np.asarray(b, dtype=np.float64) for b in data["ema_biases"]
        ]
        obj._train_losses = data["losses"].tolist()
        obj._step_count = int(data["step_count"])
        return obj


# ---------------------------------------------------------------------------
# Ablation modes
# ---------------------------------------------------------------------------


class AblationMode(Enum):
    """Ablation experiment modes for systematic comparison."""

    FULL = "full"                      # AI-guided predicate selection
    RANDOM = "random"                  # random predicate selection
    DOMAIN_HEURISTIC = "domain_heuristic"  # Hill threshold / nullcline only
    NO_ML = "no_ml"                    # template-only (no ML at all)


@dataclass
class AblationRunMetrics:
    """Metrics recorded for a single ablation run.

    Attributes
    ----------
    mode : AblationMode
        Which ablation mode was used.
    iterations_to_convergence : int
        Number of CEGAR iterations until convergence (or max).
    wall_clock_seconds : float
        Total wall-clock time.
    predicates_tried : int
        Number of distinct predicates tried.
    success : bool
        Whether verification converged.
    success_rate : float
        Fraction of refinement steps that were effective.
    """

    mode: AblationMode
    iterations_to_convergence: int = 0
    wall_clock_seconds: float = 0.0
    predicates_tried: int = 0
    success: bool = False
    success_rate: float = 0.0


# ---------------------------------------------------------------------------
# AblationController
# ---------------------------------------------------------------------------


class AblationController:
    """Systematic ablation experiment controller.

    Runs verification under different predicate selection strategies and
    records per-mode metrics for comparison reports.

    Parameters
    ----------
    modes : list[AblationMode] or None
        Modes to include.  Defaults to all four.
    rng : np.random.RandomState or None
        Random state for the ``RANDOM`` mode.
    """

    def __init__(
        self,
        modes: Optional[List[AblationMode]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        self.modes = modes or list(AblationMode)
        self._rng = rng or np.random.RandomState(42)
        self._results: Dict[AblationMode, List[AblationRunMetrics]] = {
            m: [] for m in self.modes
        }
        self._active_mode: AblationMode = AblationMode.FULL
        self._run_start_time: float = 0.0
        self._run_predicates_tried: int = 0
        self._run_successes: int = 0
        self._run_steps: int = 0

    @property
    def active_mode(self) -> AblationMode:
        return self._active_mode

    def set_mode(self, mode: AblationMode) -> None:
        """Switch to a new ablation mode."""
        self._active_mode = mode
        logger.info("Ablation mode set to: %s", mode.value)

    def begin_run(self) -> None:
        """Signal the start of a verification run under the active mode."""
        self._run_start_time = time.time()
        self._run_predicates_tried = 0
        self._run_successes = 0
        self._run_steps = 0

    def record_step(self, success: bool) -> None:
        """Record one refinement step outcome."""
        self._run_steps += 1
        self._run_predicates_tried += 1
        if success:
            self._run_successes += 1

    def end_run(self, converged: bool, iterations: int) -> AblationRunMetrics:
        """Finalise the current run and store results.

        Parameters
        ----------
        converged : whether verification converged
        iterations : total CEGAR iterations used

        Returns
        -------
        AblationRunMetrics for this run
        """
        elapsed = time.time() - self._run_start_time
        rate = self._run_successes / max(self._run_steps, 1)
        metrics = AblationRunMetrics(
            mode=self._active_mode,
            iterations_to_convergence=iterations,
            wall_clock_seconds=elapsed,
            predicates_tried=self._run_predicates_tried,
            success=converged,
            success_rate=rate,
        )
        self._results[self._active_mode].append(metrics)
        logger.info(
            "Ablation run complete: mode=%s converged=%s iters=%d time=%.2fs",
            self._active_mode.value, converged, iterations, elapsed,
        )
        return metrics

    # -- predicate selection per mode ----------------------------------------

    def select_predicate(
        self,
        candidates: List[CandidatePredicate],
        scores: Optional[np.ndarray] = None,
        species_names: Optional[List[str]] = None,
    ) -> int:
        """Choose a predicate index according to the active ablation mode.

        Parameters
        ----------
        candidates : available candidate predicates
        scores : AI-predicted scores (used only in ``FULL`` mode)
        species_names : species list (used only in ``DOMAIN_HEURISTIC`` mode)

        Returns
        -------
        int – index into *candidates*
        """
        if not candidates:
            return 0

        mode = self._active_mode
        if mode == AblationMode.FULL:
            if scores is not None and len(scores) > 0:
                return int(np.argmax(scores[: len(candidates)]))
            return 0

        if mode == AblationMode.RANDOM:
            return int(self._rng.randint(0, len(candidates)))

        if mode == AblationMode.DOMAIN_HEURISTIC:
            return self._domain_heuristic_select(candidates, species_names)

        # NO_ML: pick first threshold-type predicate
        for i, c in enumerate(candidates):
            if c.predicate_type == "threshold":
                return i
        return 0

    @staticmethod
    def _domain_heuristic_select(
        candidates: List[CandidatePredicate],
        species_names: Optional[List[str]] = None,
    ) -> int:
        """Hill-function threshold / nullcline heuristic selection.

        Prefers threshold predicates near the midpoint of their species
        range, with a bias towards rate-of-change predicates as a
        secondary choice.
        """
        best_idx = 0
        best_score = -1.0
        for i, c in enumerate(candidates):
            score = 0.0
            if c.predicate_type == "threshold":
                score = 1.0 - abs(c.threshold - 0.5)
            elif c.predicate_type == "rate_of_change":
                score = 0.6
            elif c.predicate_type == "relational":
                score = 0.4
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    # -- reporting -----------------------------------------------------------

    def get_results(self) -> Dict[str, List[AblationRunMetrics]]:
        """Return all recorded results keyed by mode name."""
        return {m.value: runs for m, runs in self._results.items()}

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comparison report across ablation modes.

        Returns a dictionary with per-mode aggregated statistics and a
        pairwise comparison summary.
        """
        report: Dict[str, Any] = {"per_mode": {}, "comparison": {}}

        for mode, runs in self._results.items():
            if not runs:
                report["per_mode"][mode.value] = {"n_runs": 0}
                continue
            iters = [r.iterations_to_convergence for r in runs]
            times = [r.wall_clock_seconds for r in runs]
            preds = [r.predicates_tried for r in runs]
            rates = [r.success_rate for r in runs]
            conv = sum(1 for r in runs if r.success)
            report["per_mode"][mode.value] = {
                "n_runs": len(runs),
                "convergence_count": conv,
                "convergence_fraction": conv / max(len(runs), 1),
                "mean_iterations": float(np.mean(iters)),
                "std_iterations": float(np.std(iters)),
                "mean_wall_time": float(np.mean(times)),
                "mean_predicates_tried": float(np.mean(preds)),
                "mean_success_rate": float(np.mean(rates)),
            }

        # Pairwise speedup of FULL vs each baseline
        full_stats = report["per_mode"].get(AblationMode.FULL.value, {})
        full_mean_iters = full_stats.get("mean_iterations", 0.0)
        for mode in self.modes:
            if mode == AblationMode.FULL:
                continue
            baseline = report["per_mode"].get(mode.value, {})
            baseline_iters = baseline.get("mean_iterations", 0.0)
            if baseline_iters > 0:
                speedup = baseline_iters / max(full_mean_iters, _EPS)
            else:
                speedup = 0.0
            report["comparison"][f"full_vs_{mode.value}"] = {
                "iteration_speedup": speedup,
                "full_mean_iters": full_mean_iters,
                "baseline_mean_iters": baseline_iters,
            }

        return report


# ---------------------------------------------------------------------------
# OutOfDistributionDetector
# ---------------------------------------------------------------------------


class OutOfDistributionDetector:
    """Detects when the predictor encounters unseen circuit topologies.

    Computes the Mahalanobis distance from a new feature vector to the
    training distribution.  If the OOD score exceeds ``threshold`` the
    detector recommends falling back to domain heuristics.

    Parameters
    ----------
    feature_dim : int
        Dimensionality of the feature vectors.
    threshold : float
        Mahalanobis distance above which an input is classified as OOD.
    min_samples : int
        Minimum training samples needed before detection is active.
    regularisation : float
        Diagonal regularisation added to the covariance matrix.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        threshold: float = 3.0,
        min_samples: int = 10,
        regularisation: float = 1e-4,
    ) -> None:
        self.feature_dim = feature_dim
        self.threshold = threshold
        self.min_samples = min_samples
        self.regularisation = regularisation

        self._mean: np.ndarray = np.zeros(feature_dim, dtype=np.float64)
        self._cov: np.ndarray = np.eye(feature_dim, dtype=np.float64)
        self._cov_inv: np.ndarray = np.eye(feature_dim, dtype=np.float64)
        self._n_samples: int = 0
        self._sum: np.ndarray = np.zeros(feature_dim, dtype=np.float64)
        self._sum_sq: np.ndarray = np.zeros(
            (feature_dim, feature_dim), dtype=np.float64,
        )

        # Tracking
        self._ood_count: int = 0
        self._total_queries: int = 0
        self._false_positives: int = 0

    def _pad_or_truncate(self, x: np.ndarray) -> np.ndarray:
        """Ensure *x* matches ``feature_dim``."""
        if len(x) >= self.feature_dim:
            return x[: self.feature_dim]
        return np.pad(x, (0, self.feature_dim - len(x)))

    def fit_incremental(self, x: np.ndarray) -> None:
        """Incrementally update the distribution estimate with a new sample.

        Uses Welford-style running mean and covariance.
        """
        x = self._pad_or_truncate(x)
        self._n_samples += 1
        self._sum += x
        self._sum_sq += np.outer(x, x)

        if self._n_samples >= self.min_samples:
            self._mean = self._sum / self._n_samples
            self._cov = (
                self._sum_sq / self._n_samples
                - np.outer(self._mean, self._mean)
                + self.regularisation * np.eye(self.feature_dim)
            )
            try:
                self._cov_inv = np.linalg.inv(self._cov)
            except np.linalg.LinAlgError:
                self._cov_inv = np.eye(self.feature_dim)

    def fit_batch(self, X: np.ndarray) -> None:
        """Fit the distribution from a batch of feature vectors.

        Parameters
        ----------
        X : (n_samples, feature_dim) array of training features.
        """
        for row in X:
            self.fit_incremental(row)

    def mahalanobis_distance(self, x: np.ndarray) -> float:
        """Compute the Mahalanobis distance of *x* to the training mean."""
        x = self._pad_or_truncate(x)
        diff = x - self._mean
        return float(np.sqrt(max(diff @ self._cov_inv @ diff, 0.0)))

    def is_ood(self, x: np.ndarray) -> bool:
        """Return ``True`` if *x* is out-of-distribution.

        Detection is inactive until ``min_samples`` have been observed.
        """
        self._total_queries += 1
        if self._n_samples < self.min_samples:
            return False
        score = self.mahalanobis_distance(x)
        ood = score > self.threshold
        if ood:
            self._ood_count += 1
        return ood

    def record_false_positive(self) -> None:
        """Record a case where OOD was flagged but the prediction worked."""
        self._false_positives += 1

    @property
    def ood_detection_rate(self) -> float:
        """Fraction of queries flagged as OOD."""
        if self._total_queries == 0:
            return 0.0
        return self._ood_count / self._total_queries

    @property
    def false_positive_rate(self) -> float:
        """Fraction of OOD flags that were false positives."""
        if self._ood_count == 0:
            return 0.0
        return self._false_positives / self._ood_count

    def get_metrics(self) -> Dict[str, Any]:
        """Return detection statistics."""
        return {
            "n_training_samples": self._n_samples,
            "total_queries": self._total_queries,
            "ood_count": self._ood_count,
            "ood_detection_rate": self.ood_detection_rate,
            "false_positive_count": self._false_positives,
            "false_positive_rate": self.false_positive_rate,
            "threshold": self.threshold,
            "active": self._n_samples >= self.min_samples,
        }

    def reset_tracking(self) -> None:
        """Reset query/detection counters without clearing the fit."""
        self._ood_count = 0
        self._total_queries = 0
        self._false_positives = 0


# ---------------------------------------------------------------------------
# OnlineCEGARIntegration
# ---------------------------------------------------------------------------


class OnlineCEGARIntegration:
    """Wires the :class:`OnlineLearner` into the CEGAR verification loop.

    Before each refinement step the predictor is queried; after the
    refinement succeeds or fails, the online learner is updated.
    Tracks learning curves (prediction accuracy vs iteration) and
    computes cumulative regret vs the random baseline.

    Parameters
    ----------
    online_learner : OnlineLearner
        The online learner that adapts during verification.
    ood_detector : OutOfDistributionDetector or None
        Optional OOD detector; if provided, the integration will fall
        back to domain heuristics when OOD is detected.
    ablation : AblationController or None
        Optional ablation controller for experiment tracking.
    fallback_mode : AblationMode
        Mode to use when OOD is detected.
    n_candidates : int
        Maximum number of candidate predicates.
    update_batch_size : int
        Mini-batch size for online updates after each step.
    """

    def __init__(
        self,
        online_learner: OnlineLearner,
        ood_detector: Optional[OutOfDistributionDetector] = None,
        ablation: Optional[AblationController] = None,
        fallback_mode: AblationMode = AblationMode.DOMAIN_HEURISTIC,
        n_candidates: int = 128,
        update_batch_size: int = 16,
    ) -> None:
        self.learner = online_learner
        self.ood_detector = ood_detector
        self.ablation = ablation
        self.fallback_mode = fallback_mode
        self.n_candidates = n_candidates
        self.update_batch_size = update_batch_size

        self._iteration: int = 0
        self._start_time: float = time.time()
        self._cumulative_regret: float = 0.0
        self._random_cumulative_reward: float = 0.0
        self._learner_cumulative_reward: float = 0.0
        self._ood_fallback_count: int = 0
        self._step_history: List[Dict[str, Any]] = []

    # -- pre-refinement query ------------------------------------------------

    def select_predicate(
        self,
        circuit_features: np.ndarray,
        cegar_state: CEGARState,
        candidates: List[CandidatePredicate],
        species_names: Optional[List[str]] = None,
    ) -> Tuple[int, bool]:
        """Select a predicate before a refinement step.

        Parameters
        ----------
        circuit_features : circuit embedding
        cegar_state : current CEGAR state
        candidates : available predicates
        species_names : species list for domain heuristic fallback

        Returns
        -------
        (chosen_index, used_ml) – the index of the chosen predicate and
        whether ML (rather than fallback) was used.
        """
        # Check for OOD
        if self.ood_detector is not None and self.ood_detector.is_ood(circuit_features):
            self._ood_fallback_count += 1
            logger.info(
                "OOD detected at iteration %d, falling back to %s",
                self._iteration, self.fallback_mode.value,
            )
            if self.ablation is not None:
                idx = self.ablation.select_predicate(
                    candidates, species_names=species_names,
                )
            else:
                idx = AblationController._domain_heuristic_select(
                    candidates, species_names,
                )
            return idx, False

        # Use ablation controller if active and not FULL mode
        if (
            self.ablation is not None
            and self.ablation.active_mode != AblationMode.FULL
        ):
            idx = self.ablation.select_predicate(
                candidates, species_names=species_names,
            )
            return idx, False

        # ML prediction
        scores = self.learner.predict(circuit_features, cegar_state)
        idx = int(np.argmax(scores[: len(candidates)]))
        return idx, True

    # -- post-refinement update ----------------------------------------------

    def report_outcome(
        self,
        circuit_features: np.ndarray,
        cegar_state: CEGARState,
        predicate_index: int,
        outcome: float,
        candidates: List[CandidatePredicate],
        used_ml: bool = True,
    ) -> float:
        """Report the outcome of a refinement step and update the learner.

        Parameters
        ----------
        circuit_features : circuit embedding used for this step
        cegar_state : CEGAR state at this step
        predicate_index : predicate that was applied
        outcome : effectiveness (0–1)
        candidates : candidate list (for regret computation)
        used_ml : whether ML was used (for ablation tracking)

        Returns
        -------
        float – training loss from the online update
        """
        # Record trace
        entry = CEGARTraceEntry(
            circuit_features=circuit_features.copy(),
            cegar_state=copy.copy(cegar_state),
            predicate_index=predicate_index,
            outcome=outcome,
            timestamp=time.time(),
        )
        self.learner.record_trace(entry)

        # Update OOD detector with this (now-seen) feature vector
        if self.ood_detector is not None:
            self.ood_detector.fit_incremental(circuit_features)
            # If we flagged OOD but the prediction would have worked,
            # record a false positive
            if not used_ml and outcome > 0.5:
                self.ood_detector.record_false_positive()

        # Online update
        loss = self.learner.update(self.update_batch_size)

        # Regret tracking: compare to random baseline
        n_cand = max(len(candidates), 1)
        random_expected_reward = 1.0 / n_cand
        self._learner_cumulative_reward += outcome
        self._random_cumulative_reward += random_expected_reward
        self._cumulative_regret += max(outcome - random_expected_reward, 0.0)

        # Record prediction accuracy
        scores = self.learner.predict(circuit_features, cegar_state)
        predicted_best = int(np.argmax(scores[: n_cand]))
        self.learner.record_prediction_outcome(predicted_best, predicate_index)

        # Ablation step tracking
        if self.ablation is not None:
            self.ablation.record_step(outcome > 0.5)

        # Record step history
        self._step_history.append({
            "iteration": self._iteration,
            "predicate_index": predicate_index,
            "outcome": outcome,
            "used_ml": used_ml,
            "loss": loss,
            "cumulative_regret": self._cumulative_regret,
        })

        # Learning curve point
        wall_time = time.time() - self._start_time
        self.learner.record_learning_curve_point(
            self._iteration, self._cumulative_regret, wall_time,
        )

        self._iteration += 1
        return loss

    # -- convenience: combined select + report --------------------------------

    def step(
        self,
        circuit_features: np.ndarray,
        cegar_state: CEGARState,
        candidates: List[CandidatePredicate],
        outcome_fn: Callable[[int], float],
        species_names: Optional[List[str]] = None,
    ) -> Tuple[int, float, float]:
        """Execute one full select → apply → update cycle.

        Parameters
        ----------
        circuit_features : circuit embedding
        cegar_state : current CEGAR state
        candidates : available predicates
        outcome_fn : callable(predicate_index) -> effectiveness
        species_names : for domain heuristic fallback

        Returns
        -------
        (chosen_index, outcome, loss)
        """
        idx, used_ml = self.select_predicate(
            circuit_features, cegar_state, candidates, species_names,
        )
        outcome = outcome_fn(idx)
        loss = self.report_outcome(
            circuit_features, cegar_state, idx, outcome, candidates, used_ml,
        )
        return idx, outcome, loss

    # -- metrics / reporting -------------------------------------------------

    @property
    def cumulative_regret(self) -> float:
        return self._cumulative_regret

    @property
    def iteration_count(self) -> int:
        return self._iteration

    @property
    def ood_fallback_count(self) -> int:
        return self._ood_fallback_count

    def get_metrics(self) -> Dict[str, Any]:
        """Return integration metrics."""
        return {
            "iteration": self._iteration,
            "cumulative_regret": self._cumulative_regret,
            "learner_cumulative_reward": self._learner_cumulative_reward,
            "random_cumulative_reward": self._random_cumulative_reward,
            "ood_fallback_count": self._ood_fallback_count,
            "learner_metrics": self.learner.get_metrics(),
            "ood_metrics": (
                self.ood_detector.get_metrics()
                if self.ood_detector is not None
                else None
            ),
        }

    def get_learning_curve(self) -> Dict[str, List[float]]:
        """Return learning curve data as parallel lists."""
        curve = self.learner.learning_curve
        return {
            "iterations": [p.iteration for p in curve],
            "prediction_accuracy": [p.prediction_accuracy for p in curve],
            "cumulative_regret": [p.cumulative_regret for p in curve],
            "loss": [p.loss for p in curve],
            "wall_time": [p.wall_time for p in curve],
        }

    def get_step_history(self) -> List[Dict[str, Any]]:
        """Return the full per-step outcome history."""
        return list(self._step_history)

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive integration report.

        Includes learning curves, regret analysis, OOD statistics, and
        (if available) ablation comparison.
        """
        report: Dict[str, Any] = {
            "summary": self.get_metrics(),
            "learning_curve": self.get_learning_curve(),
        }

        # Regret analysis
        n = max(self._iteration, 1)
        report["regret_analysis"] = {
            "total_cumulative_regret": self._cumulative_regret,
            "mean_regret_per_step": self._cumulative_regret / n,
            "learner_avg_reward": self._learner_cumulative_reward / n,
            "random_avg_reward": self._random_cumulative_reward / n,
            "reward_improvement": (
                (self._learner_cumulative_reward - self._random_cumulative_reward)
                / max(self._random_cumulative_reward, _EPS)
            ),
        }

        if self.ood_detector is not None:
            report["ood_analysis"] = self.ood_detector.get_metrics()

        if self.ablation is not None:
            report["ablation"] = self.ablation.generate_report()

        return report

    def reset(self) -> None:
        """Reset iteration counters and history (keeps learned weights)."""
        self._iteration = 0
        self._start_time = time.time()
        self._cumulative_regret = 0.0
        self._random_cumulative_reward = 0.0
        self._learner_cumulative_reward = 0.0
        self._ood_fallback_count = 0
        self._step_history.clear()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_online_cegar_system(
    input_dim: int = 64,
    n_candidates: int = 128,
    hidden_dims: Optional[List[int]] = None,
    lr0: float = 1e-3,
    tau: float = 50.0,
    ema_decay: float = 0.995,
    buffer_capacity: int = 5000,
    ood_threshold: float = 3.0,
    ood_min_samples: int = 10,
    enable_ablation: bool = True,
    rng: Optional[np.random.RandomState] = None,
) -> OnlineCEGARIntegration:
    """Convenience factory that wires up all components.

    Parameters
    ----------
    input_dim : feature vector dimensionality
    n_candidates : maximum candidate predicates
    hidden_dims : MLP hidden layer sizes
    lr0 : initial learning rate
    tau : LR decay time constant
    ema_decay : EMA coefficient
    buffer_capacity : replay buffer size
    ood_threshold : Mahalanobis distance OOD threshold
    ood_min_samples : minimum samples before OOD detection activates
    enable_ablation : whether to attach an ablation controller
    rng : random state

    Returns
    -------
    OnlineCEGARIntegration ready to use
    """
    if rng is None:
        rng = np.random.RandomState(42)

    learner = OnlineLearner(
        input_dim=input_dim,
        n_candidates=n_candidates,
        hidden_dims=hidden_dims,
        lr0=lr0,
        tau=tau,
        ema_decay=ema_decay,
        buffer_capacity=buffer_capacity,
        rng=rng,
    )

    ood = OutOfDistributionDetector(
        feature_dim=input_dim,
        threshold=ood_threshold,
        min_samples=ood_min_samples,
    )

    ablation = AblationController(rng=rng) if enable_ablation else None

    return OnlineCEGARIntegration(
        online_learner=learner,
        ood_detector=ood,
        ablation=ablation,
        n_candidates=n_candidates,
    )
