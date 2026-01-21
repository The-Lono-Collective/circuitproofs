"""
BlockCert Certifier

Computes certification metrics for surrogate blocks:
- Per-token residual error
- Local error bound (epsilon)
- Activation coverage
- Loss coverage
- Lipschitz constants (K_attn via spectral norm, K_MLP via auto-LiRPA)
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Literal
from pathlib import Path

from .ir import BlockIR, AttentionIR, MLPIR
from .interpreter import BlockInterpreter
from .tracer import TraceDataset


# Certification thresholds from BlockCert spec
TAU_ACT = 1e-2  # Activation error threshold
TAU_LOSS = 1e-3  # Loss error threshold
ALPHA_ACT = 0.94  # Required activation coverage
ALPHA_LOSS = 0.90  # Required loss coverage
L2_BALL_RADIUS = 1.0  # Radius for local Lipschitz computation


@dataclass
class LipschitzBounds:
    """Lipschitz constant bounds for a block."""

    K_attn: float  # Attention Lipschitz (spectral norm)
    K_mlp: float  # MLP Lipschitz (auto-LiRPA or analytic)
    L_block: float  # Combined block Lipschitz: (1 + K_attn) * K_mlp

    # Metadata
    K_attn_method: Literal["spectral_norm"] = "spectral_norm"
    K_mlp_method: Literal["auto_lirpa", "analytic_estimate"] = "auto_lirpa"
    K_mlp_certified: bool = True  # False if fell back to analytic


@dataclass
class CertificationMetrics:
    """Complete certification metrics for a block."""

    block_idx: int

    # Error metrics
    epsilon: float  # Local error bound (max error)
    mae: float  # Mean absolute error
    per_token_errors: np.ndarray  # All per-token errors

    # Coverage metrics
    activation_coverage: float  # Fraction with error < tau_act
    loss_coverage: Optional[float] = None  # Fraction with loss delta < tau_loss

    # Lipschitz bounds
    lipschitz: Optional[LipschitzBounds] = None

    # Thresholds used
    tau_act: float = TAU_ACT
    tau_loss: float = TAU_LOSS

    # Certification status
    is_certified: bool = False

    def check_certification(self) -> bool:
        """Check if block meets certification requirements."""
        meets_coverage = self.activation_coverage >= ALPHA_ACT
        if self.loss_coverage is not None:
            meets_coverage = meets_coverage and self.loss_coverage >= ALPHA_LOSS
        self.is_certified = meets_coverage
        return self.is_certified


class BlockCertifier:
    """
    Certifier for transformer blocks.

    Computes all metrics required for BlockCert certification.
    """

    def __init__(
        self,
        use_auto_lirpa: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize the certifier.

        Args:
            use_auto_lirpa: Whether to use auto-LiRPA for K_MLP (recommended)
            device: Device for computation
        """
        self.use_auto_lirpa = use_auto_lirpa
        self.device = device
        self.interpreter = BlockInterpreter()

        # Check auto-LiRPA availability
        self._auto_lirpa_available = False
        if use_auto_lirpa:
            try:
                from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
                self._auto_lirpa_available = True
            except ImportError:
                import warnings
                warnings.warn(
                    "auto-LiRPA not available. Falling back to analytic estimates for K_MLP. "
                    "Install with: pip install auto-LiRPA"
                )

    def certify_block(
        self,
        block_ir: BlockIR,
        trace_dataset: TraceDataset,
        model: Optional[nn.Module] = None,
        compute_loss_coverage: bool = False,
    ) -> CertificationMetrics:
        """
        Compute certification metrics for a block.

        Args:
            block_ir: The block's intermediate representation
            trace_dataset: Dataset of (input, output) traces
            model: Original model (required for loss coverage)
            compute_loss_coverage: Whether to compute loss coverage

        Returns:
            CertificationMetrics with all computed values
        """
        # Compute per-token errors
        per_token_errors = self._compute_per_token_errors(block_ir, trace_dataset)

        # Local error bound (max)
        epsilon = float(np.max(per_token_errors))

        # Mean error
        mae = float(np.mean(per_token_errors))

        # Activation coverage
        activation_coverage = float(np.mean(per_token_errors < TAU_ACT))

        # Loss coverage (if requested)
        loss_coverage = None
        if compute_loss_coverage and model is not None:
            loss_coverage = self._compute_loss_coverage(block_ir, trace_dataset, model)

        # Lipschitz bounds
        lipschitz = self._compute_lipschitz_bounds(block_ir, trace_dataset)

        metrics = CertificationMetrics(
            block_idx=block_ir.block_idx,
            epsilon=epsilon,
            mae=mae,
            per_token_errors=per_token_errors,
            activation_coverage=activation_coverage,
            loss_coverage=loss_coverage,
            lipschitz=lipschitz,
        )

        metrics.check_certification()
        return metrics

    def _compute_per_token_errors(
        self,
        block_ir: BlockIR,
        trace_dataset: TraceDataset,
    ) -> np.ndarray:
        """
        Compute per-token residual errors.

        e_l(p, t) = ||hat{x}^(l+1)_t - x^(l+1)_t||_2
        """
        errors = []

        for record in trace_dataset.records:
            # Get input/output from trace
            input_act = record.input_activations
            target_output = record.output_activations

            # Ensure 3D shape [batch=1, seq_len, d_model]
            if input_act.ndim == 2:
                input_act = input_act[np.newaxis, ...]
            if target_output.ndim == 2:
                target_output = target_output[np.newaxis, ...]

            # Run surrogate interpreter
            surrogate_output = self.interpreter.interpret_block(block_ir, input_act)

            # Per-token L2 error
            # Shape: [batch, seq_len]
            token_errors = np.linalg.norm(
                surrogate_output - target_output,
                ord=2,
                axis=-1,
            )

            errors.append(token_errors.flatten())

        return np.concatenate(errors)

    def _compute_loss_coverage(
        self,
        block_ir: BlockIR,
        trace_dataset: TraceDataset,
        model: nn.Module,
    ) -> float:
        """
        Compute loss coverage by stitching surrogate into model.

        Measures fraction of tokens where loss delta < tau_loss.
        """
        # This requires replacing the block in the model and comparing losses
        # For now, return a placeholder that indicates it wasn't computed
        # Full implementation would require model surgery

        import warnings
        warnings.warn(
            "Loss coverage computation requires model surgery (not yet implemented). "
            "Returning activation coverage as proxy."
        )

        # Return activation coverage as a conservative proxy
        per_token_errors = self._compute_per_token_errors(block_ir, trace_dataset)
        return float(np.mean(per_token_errors < TAU_LOSS))

    def _compute_lipschitz_bounds(
        self,
        block_ir: BlockIR,
        trace_dataset: TraceDataset,
    ) -> LipschitzBounds:
        """
        Compute Lipschitz bounds for the block.

        K_attn: Spectral norm of attention weight matrices
        K_mlp: Certified local bound via auto-LiRPA (or analytic fallback)
        L_block: (1 + K_attn) * K_mlp
        """
        # Compute K_attn via spectral norm
        K_attn = self._compute_attention_lipschitz(block_ir.attention)

        # Compute K_mlp
        if self._auto_lirpa_available and self.use_auto_lirpa:
            K_mlp, certified = self._compute_mlp_lipschitz_lirpa(
                block_ir.mlp, trace_dataset
            )
            method = "auto_lirpa" if certified else "analytic_estimate"
        else:
            K_mlp = self._compute_mlp_lipschitz_analytic(block_ir.mlp)
            certified = False
            method = "analytic_estimate"

        # Combined block Lipschitz
        L_block = (1 + K_attn) * K_mlp

        return LipschitzBounds(
            K_attn=K_attn,
            K_mlp=K_mlp,
            L_block=L_block,
            K_attn_method="spectral_norm",
            K_mlp_method=method,
            K_mlp_certified=certified,
        )

    def _compute_attention_lipschitz(self, attn_ir: AttentionIR) -> float:
        """
        Compute Lipschitz constant for attention via spectral norms.

        K_attn = ||W_Q||_2 * ||W_K||_2 * ||W_V||_2 * ||W_O||_2
        (simplified upper bound)
        """
        weights = attn_ir.get_masked_weights()

        # Compute spectral norms (largest singular value)
        def spectral_norm(W: np.ndarray) -> float:
            return float(np.linalg.svd(W, compute_uv=False)[0])

        norm_Q = spectral_norm(weights["W_Q"])
        norm_K = spectral_norm(weights["W_K"])
        norm_V = spectral_norm(weights["W_V"])
        norm_O = spectral_norm(weights["W_O"])

        # Upper bound for attention Lipschitz
        # This is a simplification; true bound depends on softmax behavior
        K_attn = norm_Q * norm_K * norm_V * norm_O

        return K_attn

    def _compute_mlp_lipschitz_lirpa(
        self,
        mlp_ir: MLPIR,
        trace_dataset: TraceDataset,
    ) -> Tuple[float, bool]:
        """
        Compute certified local Lipschitz bound for MLP using auto-LiRPA.

        Computes bound on L2 ball of radius 1.0 around calibration inputs.

        Returns:
            (K_mlp, certified): Lipschitz constant and whether it's certified
        """
        try:
            from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

            # Build PyTorch MLP module from IR
            mlp_module = self._build_mlp_module(mlp_ir)
            mlp_module.eval()

            # Get sample inputs from trace dataset
            inputs, _ = trace_dataset.get_flat_tokens()

            # Use a subset for efficiency
            max_samples = min(100, len(inputs))
            sample_inputs = torch.tensor(
                inputs[:max_samples], dtype=torch.float32, device=self.device
            )

            # Create bounded module
            bounded_mlp = BoundedModule(mlp_module, sample_inputs)

            # Define L2 perturbation ball
            ptb = PerturbationLpNorm(norm=2, eps=L2_BALL_RADIUS)
            bounded_input = BoundedTensor(sample_inputs, ptb)

            # Compute bounds
            lb, ub = bounded_mlp.compute_bounds(x=(bounded_input,), method="backward")

            # Lipschitz constant from output range
            output_range = (ub - lb).abs().max().item()
            K_mlp = output_range / (2 * L2_BALL_RADIUS)

            return K_mlp, True

        except Exception as e:
            import warnings
            warnings.warn(f"auto-LiRPA failed: {e}. Falling back to analytic estimate.")
            return self._compute_mlp_lipschitz_analytic(mlp_ir), False

    def _compute_mlp_lipschitz_analytic(self, mlp_ir: MLPIR) -> float:
        """
        Compute analytic Lipschitz estimate for MLP.

        K_mlp = ||W_1||_2 * L_act * ||W_2||_2

        Note: This is NOT a certified bound, just an estimate.
        """
        weights = mlp_ir.get_masked_weights()

        def spectral_norm(W: np.ndarray) -> float:
            return float(np.linalg.svd(W, compute_uv=False)[0])

        norm_1 = spectral_norm(weights["W_1"])
        norm_2 = spectral_norm(weights["W_2"])

        # Activation Lipschitz constants
        L_act_map = {
            "relu": 1.0,
            "gelu": 1.13,  # Approximate
            "silu": 1.1,  # Approximate
            "swiglu": 1.1,  # Approximate
        }
        L_act = L_act_map.get(mlp_ir.activation, 1.5)

        if mlp_ir.is_gated:
            norm_gate = spectral_norm(weights["W_gate"])
            # Gated MLP: output = W_2 @ (W_1 @ x * act(W_gate @ x))
            # Simplified bound
            K_mlp = norm_2 * (norm_1 * norm_gate * L_act)
        else:
            K_mlp = norm_1 * L_act * norm_2

        return K_mlp

    def _build_mlp_module(self, mlp_ir: MLPIR) -> nn.Module:
        """Build a PyTorch MLP module from IR for auto-LiRPA."""

        class MLPModule(nn.Module):
            def __init__(self, mlp_ir: MLPIR):
                super().__init__()
                weights = mlp_ir.get_masked_weights()

                self.W_1 = nn.Parameter(
                    torch.tensor(weights["W_1"].T, dtype=torch.float32),
                    requires_grad=False,
                )
                self.W_2 = nn.Parameter(
                    torch.tensor(weights["W_2"].T, dtype=torch.float32),
                    requires_grad=False,
                )

                self.b_1 = None
                self.b_2 = None
                if mlp_ir.b_1 is not None:
                    self.b_1 = nn.Parameter(
                        torch.tensor(mlp_ir.b_1, dtype=torch.float32),
                        requires_grad=False,
                    )
                if mlp_ir.b_2 is not None:
                    self.b_2 = nn.Parameter(
                        torch.tensor(mlp_ir.b_2, dtype=torch.float32),
                        requires_grad=False,
                    )

                self.is_gated = mlp_ir.is_gated
                if self.is_gated:
                    self.W_gate = nn.Parameter(
                        torch.tensor(weights["W_gate"].T, dtype=torch.float32),
                        requires_grad=False,
                    )
                    self.b_gate = None
                    if mlp_ir.b_gate is not None:
                        self.b_gate = nn.Parameter(
                            torch.tensor(mlp_ir.b_gate, dtype=torch.float32),
                            requires_grad=False,
                        )

                self.activation = mlp_ir.activation

            def forward(self, x):
                if self.is_gated:
                    up = torch.matmul(x, self.W_1)
                    if self.b_1 is not None:
                        up = up + self.b_1

                    gate = torch.matmul(x, self.W_gate)
                    if self.b_gate is not None:
                        gate = gate + self.b_gate

                    gate = self._activation(gate)
                    hidden = up * gate
                else:
                    hidden = torch.matmul(x, self.W_1)
                    if self.b_1 is not None:
                        hidden = hidden + self.b_1
                    hidden = self._activation(hidden)

                output = torch.matmul(hidden, self.W_2)
                if self.b_2 is not None:
                    output = output + self.b_2

                return output

            def _activation(self, x):
                if self.activation == "relu":
                    return torch.relu(x)
                elif self.activation == "gelu":
                    return torch.nn.functional.gelu(x)
                elif self.activation in ("silu", "swiglu"):
                    return torch.nn.functional.silu(x)
                else:
                    return x

        return MLPModule(mlp_ir).to(self.device)


def compute_global_error_bound(
    block_metrics: List[CertificationMetrics],
) -> float:
    """
    Compute global error bound using Lipschitz composition theorem.

    ||F_hat(x) - F(x)|| <= sum_i (epsilon_i * prod_{j>i} L_j)

    Args:
        block_metrics: List of certification metrics for each block (in order)

    Returns:
        Global error bound
    """
    n_blocks = len(block_metrics)
    global_bound = 0.0

    for i, metrics in enumerate(block_metrics):
        # Local error for this block
        epsilon_i = metrics.epsilon

        # Product of Lipschitz constants for subsequent blocks
        lipschitz_product = 1.0
        for j in range(i + 1, n_blocks):
            if block_metrics[j].lipschitz is not None:
                lipschitz_product *= block_metrics[j].lipschitz.L_block

        global_bound += epsilon_i * lipschitz_product

    return global_bound
