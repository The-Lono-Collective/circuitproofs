"""
Tests for BlockCert Intermediate Representation (IR) classes.

Tests the dataclasses: NormIR, AttentionIR, MLPIR, BlockIR, TraceRecord, TraceDataset.
"""

import numpy as np
import pytest
from pathlib import Path

from extraction.blockcert.ir import (
    NormIR,
    AttentionIR,
    MLPIR,
    BlockIR,
    TraceRecord,
    TraceDataset,
)


# =============================================================================
# NormIR Tests
# =============================================================================


class TestNormIR:
    """Tests for NormIR dataclass."""

    def test_layernorm_creation(self, small_dims):
        """Test LayerNorm IR creation with all fields."""
        d_model = small_dims["d_model"]
        norm = NormIR(
            norm_type="layernorm",
            weight=np.ones(d_model, dtype=np.float32),
            bias=np.zeros(d_model, dtype=np.float32),
            eps=1e-5,
        )

        assert norm.norm_type == "layernorm"
        assert norm.weight.shape == (d_model,)
        assert norm.bias.shape == (d_model,)
        assert norm.eps == 1e-5

    def test_rmsnorm_creation(self, small_dims):
        """Test RMSNorm IR creation (no bias)."""
        d_model = small_dims["d_model"]
        norm = NormIR(
            norm_type="rmsnorm",
            weight=np.ones(d_model, dtype=np.float32),
            bias=None,
            eps=1e-6,
        )

        assert norm.norm_type == "rmsnorm"
        assert norm.bias is None
        assert norm.eps == 1e-6

    def test_save_load_layernorm_roundtrip(self, layernorm_ir, temp_output_dir):
        """Test LayerNorm save/load roundtrip preserves data."""
        path = temp_output_dir / "norm.npz"
        original_hash = layernorm_ir.save(path)

        loaded = NormIR.load(path)

        assert loaded.norm_type == layernorm_ir.norm_type
        np.testing.assert_array_equal(loaded.weight, layernorm_ir.weight)
        np.testing.assert_array_equal(loaded.bias, layernorm_ir.bias)
        assert loaded.eps == layernorm_ir.eps

        # Verify hash is deterministic
        reload_hash = NormIR._compute_hash(path)
        assert original_hash == reload_hash

    def test_save_load_rmsnorm_roundtrip(self, rmsnorm_ir, temp_output_dir):
        """Test RMSNorm save/load roundtrip (no bias)."""
        path = temp_output_dir / "rmsnorm.npz"
        rmsnorm_ir.save(path)

        loaded = NormIR.load(path)

        assert loaded.norm_type == "rmsnorm"
        assert loaded.bias is None
        np.testing.assert_array_equal(loaded.weight, rmsnorm_ir.weight)

    def test_hash_differs_for_different_weights(self, small_dims, temp_output_dir):
        """Test that different weights produce different hashes."""
        d_model = small_dims["d_model"]

        norm1 = NormIR(
            norm_type="layernorm",
            weight=np.ones(d_model, dtype=np.float32),
            bias=np.zeros(d_model, dtype=np.float32),
        )
        norm2 = NormIR(
            norm_type="layernorm",
            weight=np.ones(d_model, dtype=np.float32) * 2,  # Different
            bias=np.zeros(d_model, dtype=np.float32),
        )

        path1 = temp_output_dir / "norm1.npz"
        path2 = temp_output_dir / "norm2.npz"

        hash1 = norm1.save(path1)
        hash2 = norm2.save(path2)

        assert hash1 != hash2


# =============================================================================
# AttentionIR Tests
# =============================================================================


class TestAttentionIR:
    """Tests for AttentionIR dataclass."""

    def test_mha_creation(self, mha_attention_ir, small_dims):
        """Test MHA creation with expected shapes."""
        d_model = small_dims["d_model"]
        num_heads = small_dims["num_heads"]
        head_dim = small_dims["head_dim"]

        assert mha_attention_ir.d_model == d_model
        assert mha_attention_ir.num_heads == num_heads
        assert mha_attention_ir.head_dim == head_dim
        assert mha_attention_ir.num_kv_heads == num_heads
        assert not mha_attention_ir.is_gqa
        assert mha_attention_ir.kv_head_repeat == 1

    def test_gqa_creation(self, gqa_attention_ir, gqa_dims):
        """Test GQA creation with fewer KV heads."""
        assert gqa_attention_ir.is_gqa
        assert gqa_attention_ir.num_kv_heads == gqa_dims["num_kv_heads"]
        assert gqa_attention_ir.kv_head_repeat == gqa_dims["num_heads"] // gqa_dims["num_kv_heads"]

    def test_default_masks_initialized(self, mha_attention_ir):
        """Test that masks are initialized to all-ones by default."""
        assert mha_attention_ir.mask_Q is not None
        assert mha_attention_ir.mask_K is not None
        assert mha_attention_ir.mask_V is not None
        assert mha_attention_ir.mask_O is not None
        assert mha_attention_ir.head_mask is not None

        # All ones (no pruning)
        np.testing.assert_array_equal(mha_attention_ir.mask_Q, np.ones_like(mha_attention_ir.W_Q))
        np.testing.assert_array_equal(mha_attention_ir.head_mask, np.ones(mha_attention_ir.num_heads))

    def test_sparsity_full_mask(self, mha_attention_ir):
        """Test sparsity computation with full mask (no pruning)."""
        sparsity = mha_attention_ir.compute_sparsity()
        assert sparsity == 1.0  # All active

    def test_sparsity_partial_mask(self, mha_attention_ir):
        """Test sparsity with 50% pruned weights."""
        # Set half of mask_Q to zero
        mha_attention_ir.mask_Q = np.ones_like(mha_attention_ir.W_Q)
        mha_attention_ir.mask_Q[:, :mha_attention_ir.mask_Q.shape[1] // 2] = 0

        sparsity = mha_attention_ir.compute_sparsity()
        assert 0.4 < sparsity < 0.9  # Partial sparsity

    def test_sparsity_all_zeros(self, mha_attention_ir):
        """Test sparsity with all-zero masks."""
        mha_attention_ir.mask_Q = np.zeros_like(mha_attention_ir.W_Q)
        mha_attention_ir.mask_K = np.zeros_like(mha_attention_ir.W_K)
        mha_attention_ir.mask_V = np.zeros_like(mha_attention_ir.W_V)
        mha_attention_ir.mask_O = np.zeros_like(mha_attention_ir.W_O)

        sparsity = mha_attention_ir.compute_sparsity()
        assert sparsity == 0.0

    def test_get_masked_weights(self, mha_attention_ir):
        """Test masked weight retrieval."""
        # Zero out part of mask_Q
        mha_attention_ir.mask_Q[:, 0] = 0

        weights = mha_attention_ir.get_masked_weights()

        # Column 0 should be zeroed
        np.testing.assert_array_equal(weights["W_Q"][:, 0], 0)
        # Other columns should be original
        np.testing.assert_array_equal(
            weights["W_Q"][:, 1:],
            mha_attention_ir.W_Q[:, 1:] * mha_attention_ir.mask_Q[:, 1:]
        )

    def test_save_load_roundtrip(self, mha_attention_ir, temp_output_dir):
        """Test AttentionIR save/load roundtrip."""
        path = temp_output_dir / "attention.npz"
        original_hash = mha_attention_ir.save(path)

        loaded = AttentionIR.load(path)

        np.testing.assert_array_almost_equal(loaded.W_Q, mha_attention_ir.W_Q)
        np.testing.assert_array_almost_equal(loaded.W_K, mha_attention_ir.W_K)
        np.testing.assert_array_almost_equal(loaded.W_V, mha_attention_ir.W_V)
        np.testing.assert_array_almost_equal(loaded.W_O, mha_attention_ir.W_O)
        assert loaded.num_heads == mha_attention_ir.num_heads
        assert loaded.head_dim == mha_attention_ir.head_dim

    def test_save_load_with_biases(self, attention_ir_with_biases, temp_output_dir):
        """Test save/load with biases."""
        path = temp_output_dir / "attention_biases.npz"
        attention_ir_with_biases.save(path)

        loaded = AttentionIR.load(path)

        np.testing.assert_array_almost_equal(loaded.b_Q, attention_ir_with_biases.b_Q)
        np.testing.assert_array_almost_equal(loaded.b_K, attention_ir_with_biases.b_K)
        np.testing.assert_array_almost_equal(loaded.b_V, attention_ir_with_biases.b_V)
        np.testing.assert_array_almost_equal(loaded.b_O, attention_ir_with_biases.b_O)

    def test_gqa_save_load_roundtrip(self, gqa_attention_ir, temp_output_dir):
        """Test GQA AttentionIR save/load preserves num_kv_heads."""
        path = temp_output_dir / "gqa_attention.npz"
        gqa_attention_ir.save(path)

        loaded = AttentionIR.load(path)

        assert loaded.num_kv_heads == gqa_attention_ir.num_kv_heads
        assert loaded.is_gqa
        assert loaded.kv_head_repeat == gqa_attention_ir.kv_head_repeat

    def test_num_kv_heads_defaults_to_num_heads(self, small_dims):
        """Test that num_kv_heads defaults to num_heads in __post_init__."""
        d_model = small_dims["d_model"]
        num_heads = small_dims["num_heads"]
        head_dim = small_dims["head_dim"]

        attn = AttentionIR(
            W_Q=np.zeros((d_model, num_heads * head_dim)),
            W_K=np.zeros((d_model, num_heads * head_dim)),
            W_V=np.zeros((d_model, num_heads * head_dim)),
            W_O=np.zeros((d_model, num_heads * head_dim)),
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=None,  # Explicitly None
        )

        assert attn.num_kv_heads == num_heads  # Should default


# =============================================================================
# MLPIR Tests
# =============================================================================


class TestMLPIR:
    """Tests for MLPIR dataclass."""

    def test_standard_mlp_creation(self, standard_mlp_ir, small_dims):
        """Test standard MLP IR creation.

        Note: The d_model and d_ff properties derive from W_1 shape, which follows
        PyTorch convention [out, in]. So d_model = W_1.shape[0] actually returns
        what we'd call d_ff in standard notation. The tests verify the actual
        weight dimensions directly.
        """
        # Verify weight shapes (PyTorch convention)
        d_model = small_dims["d_model"]
        d_ff = small_dims["d_ff"]
        assert standard_mlp_ir.W_1.shape == (d_ff, d_model)
        assert standard_mlp_ir.W_2.shape == (d_model, d_ff)
        assert not standard_mlp_ir.is_gated
        assert standard_mlp_ir.W_gate is None
        assert standard_mlp_ir.activation == "gelu"

    def test_gated_mlp_creation(self, gated_mlp_ir, small_dims):
        """Test gated MLP IR (SwiGLU) creation."""
        d_model = small_dims["d_model"]
        d_ff = small_dims["d_ff"]
        assert gated_mlp_ir.is_gated
        assert gated_mlp_ir.W_gate is not None
        assert gated_mlp_ir.W_gate.shape == (d_ff, d_model)
        assert gated_mlp_ir.W_gate.shape == gated_mlp_ir.W_1.shape
        assert gated_mlp_ir.activation == "swiglu"

    def test_default_masks_initialized(self, standard_mlp_ir):
        """Test that masks are initialized by default."""
        assert standard_mlp_ir.mask_1 is not None
        assert standard_mlp_ir.mask_2 is not None
        np.testing.assert_array_equal(standard_mlp_ir.mask_1, np.ones_like(standard_mlp_ir.W_1))

    def test_gated_mask_initialized(self, gated_mlp_ir):
        """Test that gate mask is initialized for gated MLP."""
        assert gated_mlp_ir.mask_gate is not None
        np.testing.assert_array_equal(gated_mlp_ir.mask_gate, np.ones_like(gated_mlp_ir.W_gate))

    def test_sparsity_full_mask(self, standard_mlp_ir):
        """Test sparsity with full mask."""
        sparsity = standard_mlp_ir.compute_sparsity()
        assert sparsity == 1.0

    def test_sparsity_partial_mask(self, standard_mlp_ir):
        """Test sparsity with partial mask."""
        standard_mlp_ir.mask_1 = np.zeros_like(standard_mlp_ir.W_1)  # All pruned

        sparsity = standard_mlp_ir.compute_sparsity()
        assert 0 < sparsity < 1.0  # Some pruned

    def test_sparsity_gated(self, gated_mlp_ir):
        """Test sparsity includes gate weight."""
        # Zero out gate mask
        gated_mlp_ir.mask_gate = np.zeros_like(gated_mlp_ir.W_gate)

        sparsity = gated_mlp_ir.compute_sparsity()
        assert sparsity < 1.0

    def test_get_masked_weights_standard(self, standard_mlp_ir):
        """Test masked weights for standard MLP."""
        weights = standard_mlp_ir.get_masked_weights()

        assert "W_1" in weights
        assert "W_2" in weights
        assert "W_gate" not in weights

    def test_get_masked_weights_gated(self, gated_mlp_ir):
        """Test masked weights include gate for gated MLP."""
        weights = gated_mlp_ir.get_masked_weights()

        assert "W_1" in weights
        assert "W_2" in weights
        assert "W_gate" in weights

    def test_save_load_roundtrip(self, standard_mlp_ir, temp_output_dir):
        """Test MLP IR save/load roundtrip."""
        path = temp_output_dir / "mlp.npz"
        standard_mlp_ir.save(path)

        loaded = MLPIR.load(path)

        np.testing.assert_array_almost_equal(loaded.W_1, standard_mlp_ir.W_1)
        np.testing.assert_array_almost_equal(loaded.W_2, standard_mlp_ir.W_2)
        assert loaded.activation == standard_mlp_ir.activation
        assert not loaded.is_gated

    def test_save_load_gated_roundtrip(self, gated_mlp_ir, temp_output_dir):
        """Test gated MLP IR save/load roundtrip."""
        path = temp_output_dir / "mlp_gated.npz"
        gated_mlp_ir.save(path)

        loaded = MLPIR.load(path)

        assert loaded.is_gated
        np.testing.assert_array_almost_equal(loaded.W_gate, gated_mlp_ir.W_gate)

    def test_save_load_with_biases(self, mlp_ir_with_biases, temp_output_dir):
        """Test MLP IR with biases save/load."""
        path = temp_output_dir / "mlp_biases.npz"
        mlp_ir_with_biases.save(path)

        loaded = MLPIR.load(path)

        np.testing.assert_array_almost_equal(loaded.b_1, mlp_ir_with_biases.b_1)
        np.testing.assert_array_almost_equal(loaded.b_2, mlp_ir_with_biases.b_2)


# =============================================================================
# BlockIR Tests
# =============================================================================


class TestBlockIR:
    """Tests for BlockIR dataclass."""

    def test_block_creation(self, block_ir, small_dims):
        """Test complete BlockIR creation."""
        assert block_ir.block_idx == 0
        assert block_ir.attention is not None
        assert block_ir.mlp is not None
        assert block_ir.causal_mask is not None
        assert block_ir.causal_mask.shape == (small_dims["seq_len"], small_dims["seq_len"])

    def test_compute_sparsity_dict(self, block_ir):
        """Test sparsity returns dict with attention, mlp, total."""
        sparsity = block_ir.compute_sparsity()

        assert "attention" in sparsity
        assert "mlp" in sparsity
        assert "total" in sparsity
        assert sparsity["total"] == (sparsity["attention"] + sparsity["mlp"]) / 2

    def test_save_creates_files(self, block_ir, temp_output_dir):
        """Test BlockIR.save() creates expected files."""
        hashes = block_ir.save(temp_output_dir)

        # Check files exist
        assert (temp_output_dir / "block_0_attention.npz").exists()
        assert (temp_output_dir / "block_0_mlp.npz").exists()
        assert (temp_output_dir / "block_0_causal_mask.npz").exists()
        assert (temp_output_dir / "block_0_metadata.npz").exists()

        # Check hashes returned
        assert "attention" in hashes
        assert "mlp" in hashes
        assert "causal_mask" in hashes

    def test_save_load_roundtrip(self, block_ir, temp_output_dir):
        """Test BlockIR save/load roundtrip."""
        block_ir.save(temp_output_dir)

        loaded = BlockIR.load(temp_output_dir, block_idx=0)

        assert loaded.block_idx == block_ir.block_idx
        np.testing.assert_array_almost_equal(loaded.attention.W_Q, block_ir.attention.W_Q)
        np.testing.assert_array_almost_equal(loaded.mlp.W_1, block_ir.mlp.W_1)
        np.testing.assert_array_equal(loaded.causal_mask, block_ir.causal_mask)

    def test_save_with_pre_norms(self, block_ir, temp_output_dir):
        """Test save includes pre-attention and pre-MLP norms."""
        hashes = block_ir.save(temp_output_dir)

        # Pre-norms should be saved
        assert (temp_output_dir / "block_0_pre_attn_norm.npz").exists()
        assert (temp_output_dir / "block_0_pre_mlp_norm.npz").exists()
        assert "pre_attn_norm" in hashes
        assert "pre_mlp_norm" in hashes

    def test_save_with_post_norms(self, block_ir, layernorm_ir, temp_output_dir):
        """Test save includes post-attention and post-MLP norms."""
        block_ir.post_attn_norm = layernorm_ir
        block_ir.post_mlp_norm = layernorm_ir

        hashes = block_ir.save(temp_output_dir)

        assert (temp_output_dir / "block_0_post_attn_norm.npz").exists()
        assert (temp_output_dir / "block_0_post_mlp_norm.npz").exists()
        assert "post_attn_norm" in hashes
        assert "post_mlp_norm" in hashes

    def test_load_with_norms(self, block_ir, layernorm_ir, temp_output_dir):
        """Test load restores all norms."""
        block_ir.post_attn_norm = layernorm_ir
        block_ir.post_mlp_norm = layernorm_ir
        block_ir.save(temp_output_dir)

        loaded = BlockIR.load(temp_output_dir, block_idx=0)

        assert loaded.attention.pre_norm is not None
        assert loaded.mlp.pre_norm is not None
        assert loaded.post_attn_norm is not None
        assert loaded.post_mlp_norm is not None

    def test_metadata_roundtrip(self, block_ir, temp_output_dir):
        """Test metadata is preserved through save/load."""
        block_ir.metadata = {"source_model": "test_model", "version": 1.0}
        block_ir.save(temp_output_dir)

        loaded = BlockIR.load(temp_output_dir, block_idx=0)

        assert loaded.metadata.get("source_model") == "test_model"
        assert loaded.metadata.get("version") == 1.0


# =============================================================================
# TraceRecord Tests
# =============================================================================


class TestTraceRecord:
    """Tests for TraceRecord dataclass."""

    def test_creation(self, trace_record, small_dims):
        """Test TraceRecord creation with all fields."""
        assert trace_record.block_idx == 0
        assert trace_record.input_activations.shape == (small_dims["seq_len"], small_dims["d_model"])
        assert trace_record.output_activations.shape == (small_dims["seq_len"], small_dims["d_model"])
        assert trace_record.prompt_id == "test_prompt_0"

    def test_optional_fields(self, small_dims):
        """Test TraceRecord with optional fields as None."""
        record = TraceRecord(
            block_idx=0,
            input_activations=np.zeros((8, 64), dtype=np.float32),
            output_activations=np.zeros((8, 64), dtype=np.float32),
            attention_mask=None,
            attention_weights=None,
            prompt_id=None,
        )

        assert record.attention_mask is None
        assert record.attention_weights is None
        assert record.prompt_id is None


# =============================================================================
# TraceDataset Tests
# =============================================================================


class TestTraceDataset:
    """Tests for TraceDataset dataclass."""

    def test_creation(self, trace_dataset):
        """Test TraceDataset creation."""
        assert trace_dataset.block_idx == 0
        assert len(trace_dataset) == 5

    def test_add_record(self, empty_trace_dataset, small_dims):
        """Test adding records to dataset."""
        record = TraceRecord(
            block_idx=0,
            input_activations=np.zeros((small_dims["seq_len"], small_dims["d_model"]), dtype=np.float32),
            output_activations=np.zeros((small_dims["seq_len"], small_dims["d_model"]), dtype=np.float32),
        )

        empty_trace_dataset.add_record(record)

        assert len(empty_trace_dataset) == 1
        assert empty_trace_dataset[0] is record

    def test_add_record_wrong_block_idx(self, empty_trace_dataset):
        """Test that adding record with wrong block_idx raises assertion."""
        record = TraceRecord(
            block_idx=1,  # Wrong block
            input_activations=np.zeros((8, 64), dtype=np.float32),
            output_activations=np.zeros((8, 64), dtype=np.float32),
        )

        with pytest.raises(AssertionError):
            empty_trace_dataset.add_record(record)

    def test_get_inputs(self, trace_dataset, small_dims):
        """Test get_inputs stacks all input activations."""
        inputs = trace_dataset.get_inputs()

        assert inputs.shape == (5, small_dims["seq_len"], small_dims["d_model"])

    def test_get_outputs(self, trace_dataset, small_dims):
        """Test get_outputs stacks all output activations."""
        outputs = trace_dataset.get_outputs()

        assert outputs.shape == (5, small_dims["seq_len"], small_dims["d_model"])

    def test_get_flat_tokens(self, trace_dataset, small_dims):
        """Test get_flat_tokens flattens all records."""
        inputs, outputs = trace_dataset.get_flat_tokens()

        expected_tokens = 5 * small_dims["seq_len"]  # 5 records * seq_len
        assert inputs.shape == (expected_tokens, small_dims["d_model"])
        assert outputs.shape == (expected_tokens, small_dims["d_model"])

    def test_save_load_roundtrip(self, trace_dataset, temp_output_dir):
        """Test TraceDataset save/load roundtrip."""
        path = temp_output_dir / "trace_dataset.npz"
        trace_dataset.save(path)

        loaded = TraceDataset.load(path)

        assert loaded.block_idx == trace_dataset.block_idx
        assert len(loaded) == len(trace_dataset)

        for i in range(len(trace_dataset)):
            np.testing.assert_array_almost_equal(
                loaded[i].input_activations,
                trace_dataset[i].input_activations
            )
            np.testing.assert_array_almost_equal(
                loaded[i].output_activations,
                trace_dataset[i].output_activations
            )
            assert loaded[i].prompt_id == trace_dataset[i].prompt_id

    def test_empty_dataset_operations(self, empty_trace_dataset):
        """Test operations on empty dataset."""
        assert len(empty_trace_dataset) == 0

        # get_inputs/get_outputs should fail on empty
        with pytest.raises((ValueError, IndexError)):
            empty_trace_dataset.get_inputs()

    def test_getitem(self, trace_dataset):
        """Test __getitem__ access."""
        record = trace_dataset[0]
        assert isinstance(record, TraceRecord)
        assert record.block_idx == 0

    def test_getitem_out_of_range(self, trace_dataset):
        """Test __getitem__ with out-of-range index."""
        with pytest.raises(IndexError):
            _ = trace_dataset[100]
