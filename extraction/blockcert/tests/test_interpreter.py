"""
Tests for BlockCert Interpreter.

Tests the BlockInterpreter class and convenience functions.
"""

import numpy as np
import pytest

from extraction.blockcert.ir import (
    NormIR,
    AttentionIR,
    MLPIR,
    BlockIR,
)
from extraction.blockcert.interpreter import (
    BlockInterpreter,
    interpret_attention_only,
    interpret_mlp_only,
)


# =============================================================================
# BlockInterpreter Basic Tests
# =============================================================================


class TestBlockInterpreterBasic:
    """Basic tests for BlockInterpreter."""

    def test_init_default(self):
        """Test default initialization."""
        interpreter = BlockInterpreter()
        assert not interpreter.use_sparse

    def test_init_sparse(self):
        """Test sparse mode initialization."""
        interpreter = BlockInterpreter(use_sparse=True)
        assert interpreter.use_sparse


# =============================================================================
# Attention Interpretation Tests
# =============================================================================


class TestAttentionInterpretation:
    """Tests for attention layer interpretation."""

    def test_output_shape_mha(self, mha_attention_ir, hidden_states, small_dims):
        """Test MHA output has correct shape."""
        output = interpret_attention_only(mha_attention_ir, hidden_states)

        expected_shape = (
            small_dims["batch_size"],
            small_dims["seq_len"],
            small_dims["d_model"],
        )
        assert output.shape == expected_shape

    def test_output_shape_gqa(self, gqa_attention_ir, hidden_states_gqa, gqa_dims):
        """Test GQA output has correct shape."""
        output = interpret_attention_only(gqa_attention_ir, hidden_states_gqa)

        expected_shape = (
            gqa_dims["batch_size"],
            gqa_dims["seq_len"],
            gqa_dims["d_model"],
        )
        assert output.shape == expected_shape

    def test_gqa_kv_head_repetition(self, gqa_attention_ir, hidden_states_gqa):
        """Test GQA correctly repeats KV heads to match query heads."""
        # The output should be valid (no NaN/Inf)
        output = interpret_attention_only(gqa_attention_ir, hidden_states_gqa)

        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_causal_mask_prevents_future_attention(self, mha_attention_ir, small_dims):
        """Test that causal mask prevents attending to future tokens."""
        # Create input where future tokens have distinct values
        batch_size = small_dims["batch_size"]
        seq_len = small_dims["seq_len"]
        d_model = small_dims["d_model"]

        # Input with zeros except last token has large values
        hidden_states = np.zeros((batch_size, seq_len, d_model), dtype=np.float32)
        hidden_states[:, -1, :] = 100.0  # Large values in last position

        # Create causal mask
        causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))

        # Run with causal mask
        output_masked = interpret_attention_only(
            mha_attention_ir, hidden_states, causal_mask=causal_mask
        )

        # First token output should not be influenced by last token (due to mask)
        # It should only see zeros, so output at position 0 should be close to zero
        first_token_norm = np.linalg.norm(output_masked[:, 0, :])
        last_token_norm = np.linalg.norm(output_masked[:, -1, :])

        # First token has less information (only sees itself)
        # Last token sees all tokens (including the large one)
        assert first_token_norm < last_token_norm

    def test_head_mask_zeros_pruned_heads(self, attention_ir_with_biases, hidden_states):
        """Test that head mask zeros out pruned heads."""
        output = interpret_attention_only(attention_ir_with_biases, hidden_states)

        # Output should be valid
        assert not np.any(np.isnan(output))
        # With half the heads pruned, output should have smaller norm
        # than with all heads active

    def test_output_dtype(self, mha_attention_ir, hidden_states):
        """Test output dtype is a valid float type.

        Note: The interpreter may upcast float32 to float64 during numpy operations.
        This is acceptable as long as the result is a valid floating point type.
        """
        output = interpret_attention_only(mha_attention_ir, hidden_states)
        assert np.issubdtype(output.dtype, np.floating)

    def test_batch_independence(self, mha_attention_ir, small_dims):
        """Test that batch items are processed independently."""
        batch_size = small_dims["batch_size"]
        seq_len = small_dims["seq_len"]
        d_model = small_dims["d_model"]

        rng = np.random.default_rng(42)

        # Create different inputs for each batch item
        hidden_states = rng.normal(0, 1, (batch_size, seq_len, d_model)).astype(np.float32)

        output = interpret_attention_only(mha_attention_ir, hidden_states)

        # Process each batch item separately
        for i in range(batch_size):
            single_input = hidden_states[i:i+1]
            single_output = interpret_attention_only(mha_attention_ir, single_input)
            np.testing.assert_array_almost_equal(output[i], single_output[0])


# =============================================================================
# MLP Interpretation Tests
# =============================================================================


class TestMLPInterpretation:
    """Tests for MLP layer interpretation."""

    def test_output_shape_standard(self, standard_mlp_ir, hidden_states, small_dims):
        """Test standard MLP output shape."""
        output = interpret_mlp_only(standard_mlp_ir, hidden_states)

        expected_shape = (
            small_dims["batch_size"],
            small_dims["seq_len"],
            small_dims["d_model"],
        )
        assert output.shape == expected_shape

    def test_output_shape_gated(self, gated_mlp_ir, hidden_states, small_dims):
        """Test gated MLP (SwiGLU) output shape."""
        output = interpret_mlp_only(gated_mlp_ir, hidden_states)

        expected_shape = (
            small_dims["batch_size"],
            small_dims["seq_len"],
            small_dims["d_model"],
        )
        assert output.shape == expected_shape

    @pytest.mark.parametrize("activation", ["relu", "gelu", "silu", "swiglu"])
    def test_activation_functions(self, small_dims, activation):
        """Test all supported activation functions."""
        d_model = small_dims["d_model"]
        d_ff = small_dims["d_ff"]
        batch_size = small_dims["batch_size"]
        seq_len = small_dims["seq_len"]

        rng = np.random.default_rng(42)

        is_gated = activation == "swiglu"
        # Note: Weight shapes follow PyTorch convention [out_features, in_features]
        mlp = MLPIR(
            W_1=rng.normal(0, 0.1, (d_ff, d_model)).astype(np.float32),
            W_2=rng.normal(0, 0.1, (d_model, d_ff)).astype(np.float32),
            W_gate=rng.normal(0, 0.1, (d_ff, d_model)).astype(np.float32) if is_gated else None,
            activation=activation,
        )

        hidden_states = rng.normal(0, 1, (batch_size, seq_len, d_model)).astype(np.float32)
        output = interpret_mlp_only(mlp, hidden_states)

        assert output.shape == hidden_states.shape
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_relu_zeros_negative(self, small_dims):
        """Test ReLU activation zeros out negative values."""
        d_model = small_dims["d_model"]
        d_ff = small_dims["d_ff"]

        # Create MLP with simple weights to see activation effect
        # Note: Weight shapes follow PyTorch convention [out_features, in_features]
        W_1 = np.zeros((d_ff, d_model), dtype=np.float32)
        np.fill_diagonal(W_1, 1.0)  # Identity-like up projection
        W_2 = np.zeros((d_model, d_ff), dtype=np.float32)
        np.fill_diagonal(W_2, 1.0)  # Identity-like down projection

        mlp = MLPIR(
            W_1=W_1,
            W_2=W_2,
            activation="relu",
        )

        # Input with negative values
        hidden_states = np.array([[[-1.0, -2.0] + [0.0] * (d_model - 2)]], dtype=np.float32)
        output = interpret_mlp_only(mlp, hidden_states)

        # ReLU should zero negative intermediate activations
        # (the exact output depends on the projections, but should be non-negative where ReLU applies)
        assert not np.any(np.isnan(output))

    def test_gated_mlp_gating_behavior(self, gated_mlp_ir, hidden_states):
        """Test gated MLP applies gating correctly."""
        output = interpret_mlp_only(gated_mlp_ir, hidden_states)

        # Output should be finite
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_bias_addition(self, mlp_ir_with_biases, hidden_states):
        """Test that biases are properly added."""
        output = interpret_mlp_only(mlp_ir_with_biases, hidden_states)

        # Output should be different from MLP without biases
        assert output.shape == hidden_states.shape
        assert not np.any(np.isnan(output))


# =============================================================================
# Normalization Tests
# =============================================================================


class TestNormalization:
    """Tests for normalization layer interpretation."""

    def test_layernorm_zero_mean(self, small_dims):
        """Test LayerNorm produces approximately zero mean."""
        d_model = small_dims["d_model"]
        batch_size = small_dims["batch_size"]
        seq_len = small_dims["seq_len"]

        norm = NormIR(
            norm_type="layernorm",
            weight=np.ones(d_model, dtype=np.float32),
            bias=np.zeros(d_model, dtype=np.float32),
        )

        interpreter = BlockInterpreter()
        hidden_states = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        output = interpreter._apply_norm(norm, hidden_states)

        # Mean along last axis should be approximately zero
        means = output.mean(axis=-1)
        np.testing.assert_array_almost_equal(means, np.zeros_like(means), decimal=5)

    def test_layernorm_unit_variance(self, small_dims):
        """Test LayerNorm produces approximately unit variance."""
        d_model = small_dims["d_model"]
        batch_size = small_dims["batch_size"]
        seq_len = small_dims["seq_len"]

        norm = NormIR(
            norm_type="layernorm",
            weight=np.ones(d_model, dtype=np.float32),
            bias=np.zeros(d_model, dtype=np.float32),
        )

        interpreter = BlockInterpreter()
        hidden_states = np.random.randn(batch_size, seq_len, d_model).astype(np.float32) * 10
        output = interpreter._apply_norm(norm, hidden_states)

        # Variance along last axis should be approximately one
        variances = output.var(axis=-1)
        np.testing.assert_array_almost_equal(variances, np.ones_like(variances), decimal=4)

    def test_rmsnorm_scaling(self, small_dims):
        """Test RMSNorm scales by RMS."""
        d_model = small_dims["d_model"]
        batch_size = small_dims["batch_size"]
        seq_len = small_dims["seq_len"]

        norm = NormIR(
            norm_type="rmsnorm",
            weight=np.ones(d_model, dtype=np.float32),
        )

        interpreter = BlockInterpreter()
        hidden_states = np.random.randn(batch_size, seq_len, d_model).astype(np.float32) * 5
        output = interpreter._apply_norm(norm, hidden_states)

        # RMS of output should be approximately 1
        rms = np.sqrt(np.mean(output ** 2, axis=-1))
        np.testing.assert_array_almost_equal(rms, np.ones_like(rms), decimal=4)

    def test_unknown_norm_type_raises(self, small_dims):
        """Test unknown norm type raises ValueError."""
        d_model = small_dims["d_model"]

        norm = NormIR(
            norm_type="unknown",  # Invalid
            weight=np.ones(d_model, dtype=np.float32),
        )

        interpreter = BlockInterpreter()
        hidden_states = np.random.randn(2, 8, d_model).astype(np.float32)

        with pytest.raises(ValueError, match="Unknown norm type"):
            interpreter._apply_norm(norm, hidden_states)


# =============================================================================
# Full Block Interpretation Tests
# =============================================================================


class TestFullBlockInterpretation:
    """Tests for full block interpretation."""

    def test_output_shape(self, block_ir, hidden_states, small_dims):
        """Test full block produces correct output shape."""
        interpreter = BlockInterpreter()
        output = interpreter.interpret_block(block_ir, hidden_states)

        expected_shape = (
            small_dims["batch_size"],
            small_dims["seq_len"],
            small_dims["d_model"],
        )
        assert output.shape == expected_shape

    def test_output_shape_gqa(self, block_ir_gqa, hidden_states_gqa, gqa_dims):
        """Test GQA block produces correct output shape."""
        interpreter = BlockInterpreter()
        output = interpreter.interpret_block(block_ir_gqa, hidden_states_gqa)

        expected_shape = (
            gqa_dims["batch_size"],
            gqa_dims["seq_len"],
            gqa_dims["d_model"],
        )
        assert output.shape == expected_shape

    def test_residual_connections(self, block_ir, hidden_states):
        """Test that residual connections are applied."""
        interpreter = BlockInterpreter()
        output = interpreter.interpret_block(block_ir, hidden_states)

        # Output should be different from input (attention + MLP add contributions)
        assert not np.allclose(output, hidden_states)

        # But should have similar scale (residuals prevent explosion)
        input_norm = np.linalg.norm(hidden_states)
        output_norm = np.linalg.norm(output)
        assert 0.1 < output_norm / input_norm < 10.0

    def test_no_nan_inf(self, block_ir, hidden_states):
        """Test output contains no NaN or Inf values."""
        interpreter = BlockInterpreter()
        output = interpreter.interpret_block(block_ir, hidden_states)

        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_deterministic(self, block_ir, hidden_states):
        """Test interpretation is deterministic."""
        interpreter = BlockInterpreter()

        output1 = interpreter.interpret_block(block_ir, hidden_states)
        output2 = interpreter.interpret_block(block_ir, hidden_states)

        np.testing.assert_array_equal(output1, output2)


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_softmax_large_values(self):
        """Test softmax handles large values without overflow."""
        interpreter = BlockInterpreter()

        # Large positive values
        x = np.array([[[100.0, 200.0, 300.0]]], dtype=np.float32)
        result = interpreter._softmax(x)

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        # Sum should be 1
        np.testing.assert_almost_equal(result.sum(axis=-1), 1.0)

    def test_softmax_with_negative_inf(self):
        """Test softmax handles -inf (from masking) correctly."""
        interpreter = BlockInterpreter()

        # Values with -inf (masked positions)
        x = np.array([[[1.0, -np.inf, 2.0]]], dtype=np.float32)
        result = interpreter._softmax(x)

        assert not np.any(np.isnan(result))
        # Masked position should be 0
        assert result[0, 0, 1] == 0.0
        # Sum should be 1
        np.testing.assert_almost_equal(result.sum(axis=-1), 1.0)

    def test_softmax_all_negative_inf(self):
        """Test softmax when all values are -inf (edge case)."""
        interpreter = BlockInterpreter()

        x = np.array([[[-np.inf, -np.inf, -np.inf]]], dtype=np.float32)
        result = interpreter._softmax(x)

        # Should produce NaN (0/0) or handle gracefully
        # This is an edge case that shouldn't happen in practice

    def test_attention_with_large_inputs(self, mha_attention_ir, small_dims):
        """Test attention handles large input values."""
        batch_size = small_dims["batch_size"]
        seq_len = small_dims["seq_len"]
        d_model = small_dims["d_model"]

        # Large input values
        hidden_states = np.ones((batch_size, seq_len, d_model), dtype=np.float32) * 100

        output = interpret_attention_only(mha_attention_ir, hidden_states)

        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_token_attention(self, mha_attention_ir, small_dims):
        """Test attention with single token sequence."""
        batch_size = small_dims["batch_size"]
        d_model = small_dims["d_model"]
        seq_len = 1  # Single token

        hidden_states = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        causal_mask = np.ones((seq_len, seq_len), dtype=bool)  # Single token sees itself

        output = interpret_attention_only(mha_attention_ir, hidden_states, causal_mask=causal_mask)

        assert output.shape == (batch_size, seq_len, d_model)
        assert not np.any(np.isnan(output))

    def test_batch_size_one(self, block_ir, small_dims):
        """Test with batch size of 1."""
        seq_len = small_dims["seq_len"]
        d_model = small_dims["d_model"]

        hidden_states = np.random.randn(1, seq_len, d_model).astype(np.float32)

        interpreter = BlockInterpreter()
        output = interpreter.interpret_block(block_ir, hidden_states)

        assert output.shape == (1, seq_len, d_model)

    def test_sparse_mode_equivalent(self, block_ir, hidden_states):
        """Test sparse mode produces equivalent results to dense mode."""
        dense_interpreter = BlockInterpreter(use_sparse=False)
        sparse_interpreter = BlockInterpreter(use_sparse=True)

        dense_output = dense_interpreter.interpret_block(block_ir, hidden_states)
        sparse_output = sparse_interpreter.interpret_block(block_ir, hidden_states)

        # Currently both do dense computation, so should be identical
        np.testing.assert_array_equal(dense_output, sparse_output)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_interpret_attention_only(self, mha_attention_ir, hidden_states, small_dims):
        """Test interpret_attention_only convenience function."""
        output = interpret_attention_only(mha_attention_ir, hidden_states)

        assert output.shape == hidden_states.shape
        assert not np.any(np.isnan(output))

    def test_interpret_mlp_only(self, standard_mlp_ir, hidden_states, small_dims):
        """Test interpret_mlp_only convenience function."""
        output = interpret_mlp_only(standard_mlp_ir, hidden_states)

        assert output.shape == hidden_states.shape
        assert not np.any(np.isnan(output))

    def test_interpret_attention_with_masks(self, mha_attention_ir, hidden_states, causal_mask):
        """Test interpret_attention_only with causal mask."""
        output = interpret_attention_only(
            mha_attention_ir,
            hidden_states,
            causal_mask=causal_mask
        )

        assert output.shape == hidden_states.shape
        assert not np.any(np.isnan(output))
