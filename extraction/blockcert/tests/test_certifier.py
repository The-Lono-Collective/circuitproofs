"""
Tests for BlockCert Certifier.

Tests the BlockCertifier class, LipschitzBounds, CertificationMetrics,
and related functions.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from extraction.blockcert.ir import (
    AttentionIR,
    MLPIR,
    BlockIR,
    TraceRecord,
    TraceDataset,
)
from extraction.blockcert.certifier import (
    BlockCertifier,
    LipschitzBounds,
    CertificationMetrics,
    compute_global_error_bound,
    get_available_memory_gb,
    estimate_lirpa_memory_gb,
    check_memory_for_lirpa,
    TAU_ACT,
    TAU_LOSS,
    ALPHA_ACT,
    ALPHA_LOSS,
)


# =============================================================================
# Memory Estimation Tests
# =============================================================================


class TestMemoryEstimation:
    """Tests for memory estimation functions."""

    def test_estimate_lirpa_memory_scales_quadratically(self):
        """Test memory estimation scales with d_ff^2."""
        mem_small = estimate_lirpa_memory_gb(d_ff=2816)  # Half of reference
        mem_ref = estimate_lirpa_memory_gb(d_ff=5632)    # Reference
        mem_large = estimate_lirpa_memory_gb(d_ff=11264) # Double reference

        # Quadratic scaling: doubling d_ff should 4x memory
        assert mem_ref > mem_small
        assert mem_large > mem_ref
        assert 3.5 < (mem_large / mem_ref) < 4.5  # Approximately 4x

    def test_estimate_includes_sample_overhead(self):
        """Test memory estimation includes sample overhead."""
        mem_100 = estimate_lirpa_memory_gb(d_ff=5632, n_samples=100)
        mem_200 = estimate_lirpa_memory_gb(d_ff=5632, n_samples=200)

        # More samples should require more memory
        assert mem_200 > mem_100

    def test_check_memory_sufficient(self, mock_psutil_high_memory):
        """Test check_memory returns sufficient=True with high memory."""
        sufficient, available, required = check_memory_for_lirpa(d_ff=5632)

        assert sufficient
        assert available == 64.0
        assert required > 0

    def test_check_memory_insufficient(self, mock_psutil_low_memory):
        """Test check_memory returns sufficient=False with low memory."""
        sufficient, available, required = check_memory_for_lirpa(d_ff=5632)

        assert not sufficient
        assert available == 8.0
        assert required > available

    def test_check_memory_unavailable(self, mock_psutil_unavailable):
        """Test check_memory handles unavailable memory detection."""
        sufficient, available, required = check_memory_for_lirpa(d_ff=5632)

        # Should assume sufficient but warn
        assert sufficient
        assert available == -1.0


# =============================================================================
# LipschitzBounds Tests
# =============================================================================


class TestLipschitzBounds:
    """Tests for LipschitzBounds dataclass."""

    def test_creation(self):
        """Test LipschitzBounds creation."""
        bounds = LipschitzBounds(
            K_attn=2.5,
            K_mlp=3.0,
            L_block=10.5,
            K_attn_method="spectral_norm",
            K_mlp_method="auto_lirpa",
            K_mlp_certified=True,
        )

        assert bounds.K_attn == 2.5
        assert bounds.K_mlp == 3.0
        assert bounds.L_block == 10.5
        assert bounds.K_mlp_certified

    def test_defaults(self):
        """Test LipschitzBounds default values."""
        bounds = LipschitzBounds(
            K_attn=1.0,
            K_mlp=1.0,
            L_block=2.0,
        )

        assert bounds.K_attn_method == "spectral_norm"
        assert bounds.K_mlp_method == "auto_lirpa"
        assert bounds.K_mlp_certified


# =============================================================================
# CertificationMetrics Tests
# =============================================================================


class TestCertificationMetrics:
    """Tests for CertificationMetrics dataclass."""

    def test_creation(self):
        """Test CertificationMetrics creation."""
        errors = np.array([0.001, 0.005, 0.01, 0.02])
        metrics = CertificationMetrics(
            block_idx=0,
            epsilon=0.02,
            mae=0.009,
            per_token_errors=errors,
            activation_coverage=0.75,  # 3/4 below TAU_ACT
        )

        assert metrics.block_idx == 0
        assert metrics.epsilon == 0.02
        assert metrics.mae == 0.009
        assert len(metrics.per_token_errors) == 4
        assert metrics.activation_coverage == 0.75

    def test_check_certification_passes(self):
        """Test check_certification returns True when coverage >= threshold."""
        metrics = CertificationMetrics(
            block_idx=0,
            epsilon=0.001,
            mae=0.0005,
            per_token_errors=np.array([0.001]),
            activation_coverage=0.95,  # > ALPHA_ACT (0.94)
            loss_coverage=0.92,  # > ALPHA_LOSS (0.90)
        )

        result = metrics.check_certification()

        assert result
        assert metrics.is_certified

    def test_check_certification_fails_activation(self):
        """Test check_certification fails with low activation coverage."""
        metrics = CertificationMetrics(
            block_idx=0,
            epsilon=0.001,
            mae=0.0005,
            per_token_errors=np.array([0.001]),
            activation_coverage=0.90,  # < ALPHA_ACT (0.94)
        )

        result = metrics.check_certification()

        assert not result
        assert not metrics.is_certified

    def test_check_certification_fails_loss(self):
        """Test check_certification fails with low loss coverage."""
        metrics = CertificationMetrics(
            block_idx=0,
            epsilon=0.001,
            mae=0.0005,
            per_token_errors=np.array([0.001]),
            activation_coverage=0.95,  # > ALPHA_ACT
            loss_coverage=0.85,  # < ALPHA_LOSS (0.90)
        )

        result = metrics.check_certification()

        assert not result
        assert not metrics.is_certified

    def test_check_certification_ignores_none_loss(self):
        """Test check_certification ignores loss coverage when None."""
        metrics = CertificationMetrics(
            block_idx=0,
            epsilon=0.001,
            mae=0.0005,
            per_token_errors=np.array([0.001]),
            activation_coverage=0.95,
            loss_coverage=None,  # Not computed
        )

        result = metrics.check_certification()

        assert result
        assert metrics.is_certified


# =============================================================================
# BlockCertifier Initialization Tests
# =============================================================================


class TestBlockCertifierInit:
    """Tests for BlockCertifier initialization."""

    def test_init_default(self):
        """Test default initialization."""
        certifier = BlockCertifier()

        assert certifier.use_auto_lirpa
        assert certifier.device == "cpu"
        assert certifier.interpreter is not None

    def test_init_without_auto_lirpa(self):
        """Test initialization with auto_lirpa disabled."""
        certifier = BlockCertifier(use_auto_lirpa=False)

        assert not certifier.use_auto_lirpa

    def test_init_auto_lirpa_unavailable(self):
        """Test initialization when auto-LiRPA not installed.

        Note: The warning is emitted during __init__ based on import availability.
        Since auto-LiRPA may or may not be installed in the test environment,
        we just verify the certifier initializes correctly with use_auto_lirpa=True.
        """
        # Create certifier with auto-LiRPA requested
        certifier = BlockCertifier(use_auto_lirpa=True)

        # Should have set _auto_lirpa_available based on import check
        assert hasattr(certifier, "_auto_lirpa_available")
        # Should work regardless of whether auto-LiRPA is installed
        assert certifier.use_auto_lirpa == True


# =============================================================================
# Per-Token Error Computation Tests
# =============================================================================


class TestPerTokenErrors:
    """Tests for per-token error computation."""

    def test_error_computation_shape(self, block_ir, trace_dataset, small_dims):
        """Test per-token errors have correct shape."""
        certifier = BlockCertifier(use_auto_lirpa=False)
        errors = certifier._compute_per_token_errors(block_ir, trace_dataset)

        # 5 records * seq_len tokens
        expected_tokens = 5 * small_dims["seq_len"]
        assert errors.shape == (expected_tokens,)

    def test_errors_are_non_negative(self, block_ir, trace_dataset):
        """Test all per-token errors are non-negative (L2 norms)."""
        certifier = BlockCertifier(use_auto_lirpa=False)
        errors = certifier._compute_per_token_errors(block_ir, trace_dataset)

        assert np.all(errors >= 0)

    def test_epsilon_is_max_error(self, block_ir, trace_dataset):
        """Test epsilon equals maximum per-token error."""
        certifier = BlockCertifier(use_auto_lirpa=False)
        metrics = certifier.certify_block(block_ir, trace_dataset)

        assert metrics.epsilon == np.max(metrics.per_token_errors)

    def test_mae_is_mean_error(self, block_ir, trace_dataset):
        """Test MAE equals mean of per-token errors."""
        certifier = BlockCertifier(use_auto_lirpa=False)
        metrics = certifier.certify_block(block_ir, trace_dataset)

        np.testing.assert_almost_equal(metrics.mae, np.mean(metrics.per_token_errors))


# =============================================================================
# Lipschitz Computation Tests
# =============================================================================


class TestLipschitzComputation:
    """Tests for Lipschitz constant computation."""

    def test_attention_lipschitz_positive(self, mha_attention_ir):
        """Test attention Lipschitz constant is positive."""
        certifier = BlockCertifier(use_auto_lirpa=False)
        K_attn = certifier._compute_attention_lipschitz(mha_attention_ir)

        assert K_attn > 0

    def test_attention_lipschitz_uses_spectral_norm(self, small_dims):
        """Test attention Lipschitz uses spectral norm of weight matrices."""
        d_model = small_dims["d_model"]
        num_heads = small_dims["num_heads"]
        head_dim = small_dims["head_dim"]

        # Create attention with known spectral norms
        # Identity matrices have spectral norm 1
        attn = AttentionIR(
            W_Q=np.eye(d_model, num_heads * head_dim, dtype=np.float32),
            W_K=np.eye(d_model, num_heads * head_dim, dtype=np.float32),
            W_V=np.eye(d_model, num_heads * head_dim, dtype=np.float32),
            W_O=np.eye(d_model, num_heads * head_dim, dtype=np.float32),
            num_heads=num_heads,
            head_dim=head_dim,
        )

        certifier = BlockCertifier(use_auto_lirpa=False)
        K_attn = certifier._compute_attention_lipschitz(attn)

        # Product of spectral norms = 1 * 1 * 1 * 1 = 1
        np.testing.assert_almost_equal(K_attn, 1.0, decimal=5)

    def test_mlp_lipschitz_analytic_positive(self, standard_mlp_ir):
        """Test analytic MLP Lipschitz is positive."""
        certifier = BlockCertifier(use_auto_lirpa=False)
        K_mlp = certifier._compute_mlp_lipschitz_analytic(standard_mlp_ir)

        assert K_mlp > 0

    def test_mlp_lipschitz_gated(self, gated_mlp_ir):
        """Test analytic MLP Lipschitz handles gated architecture."""
        certifier = BlockCertifier(use_auto_lirpa=False)
        K_mlp = certifier._compute_mlp_lipschitz_analytic(gated_mlp_ir)

        assert K_mlp > 0

    def test_lipschitz_bounds_method_analytic_fallback(self, block_ir, trace_dataset):
        """Test Lipschitz bounds use analytic fallback when auto-LiRPA disabled."""
        certifier = BlockCertifier(use_auto_lirpa=False)
        bounds = certifier._compute_lipschitz_bounds(block_ir, trace_dataset)

        assert bounds.K_mlp_method == "analytic_estimate"
        assert not bounds.K_mlp_certified

    def test_lipschitz_l_block_formula(self, block_ir, trace_dataset):
        """Test L_block = (1 + K_attn) * K_mlp."""
        certifier = BlockCertifier(use_auto_lirpa=False)
        bounds = certifier._compute_lipschitz_bounds(block_ir, trace_dataset)

        expected_L = (1 + bounds.K_attn) * bounds.K_mlp
        np.testing.assert_almost_equal(bounds.L_block, expected_L)


# =============================================================================
# Full Certification Tests
# =============================================================================


class TestFullCertification:
    """Tests for full block certification."""

    def test_certify_block_returns_metrics(self, block_ir, trace_dataset):
        """Test certify_block returns CertificationMetrics."""
        certifier = BlockCertifier(use_auto_lirpa=False)
        metrics = certifier.certify_block(block_ir, trace_dataset)

        assert isinstance(metrics, CertificationMetrics)
        assert metrics.block_idx == block_ir.block_idx
        assert metrics.epsilon > 0
        assert metrics.lipschitz is not None

    def test_certify_block_activation_coverage(self, block_ir, trace_dataset):
        """Test activation coverage is fraction below threshold."""
        certifier = BlockCertifier(use_auto_lirpa=False)
        metrics = certifier.certify_block(block_ir, trace_dataset)

        # Manually compute expected coverage
        expected = np.mean(metrics.per_token_errors < TAU_ACT)
        np.testing.assert_almost_equal(metrics.activation_coverage, expected)

    def test_certify_block_thresholds_stored(self, block_ir, trace_dataset):
        """Test certification thresholds are stored in metrics."""
        certifier = BlockCertifier(use_auto_lirpa=False)
        metrics = certifier.certify_block(block_ir, trace_dataset)

        assert metrics.tau_act == TAU_ACT
        assert metrics.tau_loss == TAU_LOSS


# =============================================================================
# Global Error Bound Tests
# =============================================================================


class TestGlobalErrorBound:
    """Tests for global error bound computation."""

    def test_single_block(self):
        """Test global bound with single block equals epsilon."""
        metrics = CertificationMetrics(
            block_idx=0,
            epsilon=0.01,
            mae=0.005,
            per_token_errors=np.array([0.01]),
            activation_coverage=0.95,
            lipschitz=LipschitzBounds(
                K_attn=1.0,
                K_mlp=2.0,
                L_block=6.0,
            ),
        )

        global_bound = compute_global_error_bound([metrics])

        # Single block: epsilon * 1 (no subsequent blocks)
        assert global_bound == 0.01

    def test_two_blocks(self):
        """Test global bound with two blocks."""
        metrics1 = CertificationMetrics(
            block_idx=0,
            epsilon=0.01,
            mae=0.005,
            per_token_errors=np.array([0.01]),
            activation_coverage=0.95,
            lipschitz=LipschitzBounds(K_attn=1.0, K_mlp=2.0, L_block=4.0),
        )
        metrics2 = CertificationMetrics(
            block_idx=1,
            epsilon=0.02,
            mae=0.01,
            per_token_errors=np.array([0.02]),
            activation_coverage=0.95,
            lipschitz=LipschitzBounds(K_attn=1.0, K_mlp=2.0, L_block=4.0),
        )

        global_bound = compute_global_error_bound([metrics1, metrics2])

        # Block 0: 0.01 * L_block_1 = 0.01 * 4 = 0.04
        # Block 1: 0.02 * 1 = 0.02
        # Total: 0.04 + 0.02 = 0.06
        expected = 0.01 * 4.0 + 0.02 * 1.0
        np.testing.assert_almost_equal(global_bound, expected)

    def test_three_blocks(self):
        """Test global bound with three blocks."""
        blocks = []
        for i in range(3):
            metrics = CertificationMetrics(
                block_idx=i,
                epsilon=0.01,
                mae=0.005,
                per_token_errors=np.array([0.01]),
                activation_coverage=0.95,
                lipschitz=LipschitzBounds(K_attn=1.0, K_mlp=1.0, L_block=2.0),
            )
            blocks.append(metrics)

        global_bound = compute_global_error_bound(blocks)

        # Block 0: 0.01 * 2 * 2 = 0.04
        # Block 1: 0.01 * 2 = 0.02
        # Block 2: 0.01 * 1 = 0.01
        # Total: 0.07
        expected = 0.01 * 4 + 0.01 * 2 + 0.01 * 1
        np.testing.assert_almost_equal(global_bound, expected)

    def test_handles_none_lipschitz(self):
        """Test global bound handles blocks without Lipschitz bounds."""
        metrics1 = CertificationMetrics(
            block_idx=0,
            epsilon=0.01,
            mae=0.005,
            per_token_errors=np.array([0.01]),
            activation_coverage=0.95,
            lipschitz=None,  # No Lipschitz
        )
        metrics2 = CertificationMetrics(
            block_idx=1,
            epsilon=0.02,
            mae=0.01,
            per_token_errors=np.array([0.02]),
            activation_coverage=0.95,
            lipschitz=LipschitzBounds(K_attn=1.0, K_mlp=2.0, L_block=4.0),
        )

        global_bound = compute_global_error_bound([metrics1, metrics2])

        # Block 0: 0.01 * L_block_1 = 0.01 * 4 = 0.04
        # Block 1: 0.02 * 1 = 0.02
        expected = 0.01 * 4.0 + 0.02
        np.testing.assert_almost_equal(global_bound, expected)


# =============================================================================
# Memory Check Integration Tests
# =============================================================================


class TestMemoryCheckIntegration:
    """Tests for memory check integration with certification."""

    def test_low_memory_triggers_analytic_fallback(
        self, block_ir, trace_dataset, mock_psutil_low_memory
    ):
        """Test low memory triggers analytic fallback for K_mlp."""
        certifier = BlockCertifier(use_auto_lirpa=True)
        # Force auto-LiRPA to be "available" for this test
        certifier._auto_lirpa_available = True

        bounds = certifier._compute_lipschitz_bounds(block_ir, trace_dataset)

        # Should fall back to analytic due to low memory
        assert bounds.K_mlp_method == "analytic_estimate"
        assert not bounds.K_mlp_certified


# =============================================================================
# MLPModule Builder Tests
# =============================================================================


class TestMLPModuleBuilder:
    """Tests for PyTorch MLP module building."""

    def test_build_standard_mlp(self, standard_mlp_ir):
        """Test building standard MLP module."""
        certifier = BlockCertifier(use_auto_lirpa=False)
        module = certifier._build_mlp_module(standard_mlp_ir)

        # Check module has expected structure
        assert hasattr(module, "W_1")
        assert hasattr(module, "W_2")
        assert not module.is_gated

    def test_build_gated_mlp(self, gated_mlp_ir):
        """Test building gated MLP module."""
        certifier = BlockCertifier(use_auto_lirpa=False)
        module = certifier._build_mlp_module(gated_mlp_ir)

        assert module.is_gated
        assert hasattr(module, "W_gate")

    def test_module_forward_shape(self, standard_mlp_ir, small_dims):
        """Test built module produces correct output shape."""
        import torch

        certifier = BlockCertifier(use_auto_lirpa=False)
        module = certifier._build_mlp_module(standard_mlp_ir)

        # Create input tensor
        x = torch.randn(10, small_dims["d_model"])
        output = module(x)

        assert output.shape == (10, small_dims["d_model"])

    def test_module_with_biases(self, mlp_ir_with_biases, small_dims):
        """Test module with biases."""
        import torch

        certifier = BlockCertifier(use_auto_lirpa=False)
        module = certifier._build_mlp_module(mlp_ir_with_biases)

        assert module.b_1 is not None
        assert module.b_2 is not None

        x = torch.randn(10, small_dims["d_model"])
        output = module(x)

        assert output.shape == (10, small_dims["d_model"])
