"""
Shared pytest fixtures for BlockCert tests.

Provides reusable fixtures for model dimensions, IR components,
input data, and mocking utilities.
"""

import numpy as np
import pytest
from pathlib import Path
from typing import Dict, Tuple
from unittest.mock import patch, MagicMock

from extraction.blockcert.ir import (
    NormIR,
    AttentionIR,
    MLPIR,
    BlockIR,
    TraceRecord,
    TraceDataset,
)


# =============================================================================
# Model Dimension Fixtures
# =============================================================================


@pytest.fixture
def small_dims() -> Dict[str, int]:
    """Small model dimensions for fast testing."""
    return {
        "d_model": 64,
        "d_ff": 256,
        "num_heads": 4,
        "head_dim": 16,
        "num_kv_heads": 4,  # MHA (same as num_heads)
        "seq_len": 8,
        "batch_size": 2,
    }


@pytest.fixture
def gqa_dims() -> Dict[str, int]:
    """Dimensions for Grouped Query Attention testing."""
    return {
        "d_model": 64,
        "d_ff": 256,
        "num_heads": 8,
        "head_dim": 8,
        "num_kv_heads": 2,  # GQA: 4 query heads per KV head
        "seq_len": 8,
        "batch_size": 2,
    }


@pytest.fixture(params=[
    {"d_model": 32, "d_ff": 128, "num_heads": 2, "head_dim": 16, "num_kv_heads": 2},
    {"d_model": 64, "d_ff": 256, "num_heads": 4, "head_dim": 16, "num_kv_heads": 4},
    {"d_model": 64, "d_ff": 256, "num_heads": 8, "head_dim": 8, "num_kv_heads": 2},
])
def model_dims(request) -> Dict[str, int]:
    """Parameterized model dimensions for comprehensive testing."""
    dims = request.param.copy()
    dims["seq_len"] = 8
    dims["batch_size"] = 2
    return dims


# =============================================================================
# NormIR Fixtures
# =============================================================================


@pytest.fixture
def layernorm_ir(small_dims) -> NormIR:
    """LayerNorm IR with weight and bias."""
    d_model = small_dims["d_model"]
    return NormIR(
        norm_type="layernorm",
        weight=np.ones(d_model, dtype=np.float32),
        bias=np.zeros(d_model, dtype=np.float32),
        eps=1e-5,
    )


@pytest.fixture
def rmsnorm_ir(small_dims) -> NormIR:
    """RMSNorm IR (no bias)."""
    d_model = small_dims["d_model"]
    return NormIR(
        norm_type="rmsnorm",
        weight=np.ones(d_model, dtype=np.float32),
        bias=None,
        eps=1e-6,
    )


# =============================================================================
# AttentionIR Fixtures
# =============================================================================


@pytest.fixture
def mha_attention_ir(small_dims) -> AttentionIR:
    """Multi-Head Attention IR (standard MHA).

    Note: Weight shapes follow PyTorch convention [out_features, in_features].
    The interpreter uses W.T for matmul, so we store [out, in] to get [in, out] after transpose.
    """
    d_model = small_dims["d_model"]
    num_heads = small_dims["num_heads"]
    head_dim = small_dims["head_dim"]

    # Random weights with Xavier initialization
    rng = np.random.default_rng(42)
    scale = np.sqrt(2.0 / (d_model + num_heads * head_dim))

    # Weights: [out_features, in_features] following PyTorch convention
    return AttentionIR(
        W_Q=rng.normal(0, scale, (num_heads * head_dim, d_model)).astype(np.float32),
        W_K=rng.normal(0, scale, (num_heads * head_dim, d_model)).astype(np.float32),
        W_V=rng.normal(0, scale, (num_heads * head_dim, d_model)).astype(np.float32),
        W_O=rng.normal(0, scale, (d_model, num_heads * head_dim)).astype(np.float32),
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_heads,  # MHA
    )


@pytest.fixture
def gqa_attention_ir(gqa_dims) -> AttentionIR:
    """Grouped Query Attention IR.

    Note: Weight shapes follow PyTorch convention [out_features, in_features].
    """
    d_model = gqa_dims["d_model"]
    num_heads = gqa_dims["num_heads"]
    head_dim = gqa_dims["head_dim"]
    num_kv_heads = gqa_dims["num_kv_heads"]

    rng = np.random.default_rng(42)
    scale = np.sqrt(2.0 / (d_model + num_heads * head_dim))

    # W_Q: [num_heads * head_dim, d_model]
    # W_K, W_V: [num_kv_heads * head_dim, d_model] - smaller for GQA
    # W_O: [d_model, num_heads * head_dim]
    return AttentionIR(
        W_Q=rng.normal(0, scale, (num_heads * head_dim, d_model)).astype(np.float32),
        W_K=rng.normal(0, scale, (num_kv_heads * head_dim, d_model)).astype(np.float32),
        W_V=rng.normal(0, scale, (num_kv_heads * head_dim, d_model)).astype(np.float32),
        W_O=rng.normal(0, scale, (d_model, num_heads * head_dim)).astype(np.float32),
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,  # GQA
    )


@pytest.fixture
def attention_ir_with_biases(small_dims) -> AttentionIR:
    """Attention IR with biases and head mask.

    Note: Weight shapes follow PyTorch convention [out_features, in_features].
    """
    d_model = small_dims["d_model"]
    num_heads = small_dims["num_heads"]
    head_dim = small_dims["head_dim"]

    rng = np.random.default_rng(42)
    scale = np.sqrt(2.0 / (d_model + num_heads * head_dim))

    # Prune 2 out of 4 heads
    head_mask = np.array([1, 0, 1, 0], dtype=np.float32)

    return AttentionIR(
        W_Q=rng.normal(0, scale, (num_heads * head_dim, d_model)).astype(np.float32),
        W_K=rng.normal(0, scale, (num_heads * head_dim, d_model)).astype(np.float32),
        W_V=rng.normal(0, scale, (num_heads * head_dim, d_model)).astype(np.float32),
        W_O=rng.normal(0, scale, (d_model, num_heads * head_dim)).astype(np.float32),
        b_Q=rng.normal(0, 0.01, num_heads * head_dim).astype(np.float32),
        b_K=rng.normal(0, 0.01, num_heads * head_dim).astype(np.float32),
        b_V=rng.normal(0, 0.01, num_heads * head_dim).astype(np.float32),
        b_O=rng.normal(0, 0.01, d_model).astype(np.float32),
        head_mask=head_mask,
        num_heads=num_heads,
        head_dim=head_dim,
    )


# =============================================================================
# MLPIR Fixtures
# =============================================================================


@pytest.fixture
def standard_mlp_ir(small_dims) -> MLPIR:
    """Standard MLP IR (W_1 -> activation -> W_2).

    Note: Weight shapes follow PyTorch convention [out_features, in_features].
    W_1 is up projection: [d_ff, d_model]
    W_2 is down projection: [d_model, d_ff]
    """
    d_model = small_dims["d_model"]
    d_ff = small_dims["d_ff"]

    rng = np.random.default_rng(42)
    scale_1 = np.sqrt(2.0 / (d_model + d_ff))
    scale_2 = np.sqrt(2.0 / (d_ff + d_model))

    return MLPIR(
        W_1=rng.normal(0, scale_1, (d_ff, d_model)).astype(np.float32),
        W_2=rng.normal(0, scale_2, (d_model, d_ff)).astype(np.float32),
        activation="gelu",
    )


@pytest.fixture
def gated_mlp_ir(small_dims) -> MLPIR:
    """Gated MLP IR (SwiGLU style).

    Note: Weight shapes follow PyTorch convention [out_features, in_features].
    """
    d_model = small_dims["d_model"]
    d_ff = small_dims["d_ff"]

    rng = np.random.default_rng(42)
    scale = np.sqrt(2.0 / (d_model + d_ff))

    return MLPIR(
        W_1=rng.normal(0, scale, (d_ff, d_model)).astype(np.float32),
        W_2=rng.normal(0, scale, (d_model, d_ff)).astype(np.float32),
        W_gate=rng.normal(0, scale, (d_ff, d_model)).astype(np.float32),
        activation="swiglu",
    )


@pytest.fixture
def mlp_ir_with_biases(small_dims) -> MLPIR:
    """MLP IR with biases.

    Note: Weight shapes follow PyTorch convention [out_features, in_features].
    """
    d_model = small_dims["d_model"]
    d_ff = small_dims["d_ff"]

    rng = np.random.default_rng(42)
    scale = np.sqrt(2.0 / (d_model + d_ff))

    return MLPIR(
        W_1=rng.normal(0, scale, (d_ff, d_model)).astype(np.float32),
        W_2=rng.normal(0, scale, (d_model, d_ff)).astype(np.float32),
        b_1=rng.normal(0, 0.01, d_ff).astype(np.float32),
        b_2=rng.normal(0, 0.01, d_model).astype(np.float32),
        activation="relu",
    )


@pytest.fixture(params=["relu", "gelu", "silu", "swiglu"])
def mlp_ir_activations(request, small_dims) -> MLPIR:
    """MLP IR parameterized by activation function.

    Note: Weight shapes follow PyTorch convention [out_features, in_features].
    """
    d_model = small_dims["d_model"]
    d_ff = small_dims["d_ff"]

    rng = np.random.default_rng(42)
    scale = np.sqrt(2.0 / (d_model + d_ff))

    activation = request.param
    is_gated = activation == "swiglu"

    return MLPIR(
        W_1=rng.normal(0, scale, (d_ff, d_model)).astype(np.float32),
        W_2=rng.normal(0, scale, (d_model, d_ff)).astype(np.float32),
        W_gate=rng.normal(0, scale, (d_ff, d_model)).astype(np.float32) if is_gated else None,
        activation=activation,
    )


# =============================================================================
# BlockIR Fixtures
# =============================================================================


@pytest.fixture
def block_ir(mha_attention_ir, standard_mlp_ir, layernorm_ir, small_dims) -> BlockIR:
    """Complete BlockIR with pre-layer norms."""
    seq_len = small_dims["seq_len"]

    # Create causal mask
    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))

    # Add pre-norms to attention and MLP
    mha_attention_ir.pre_norm = layernorm_ir
    standard_mlp_ir.pre_norm = layernorm_ir

    return BlockIR(
        block_idx=0,
        attention=mha_attention_ir,
        mlp=standard_mlp_ir,
        causal_mask=causal_mask,
    )


@pytest.fixture
def block_ir_gqa(gqa_attention_ir, gated_mlp_ir, rmsnorm_ir, gqa_dims) -> BlockIR:
    """BlockIR with GQA attention and gated MLP."""
    seq_len = gqa_dims["seq_len"]
    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))

    gqa_attention_ir.pre_norm = rmsnorm_ir
    gated_mlp_ir.pre_norm = rmsnorm_ir

    return BlockIR(
        block_idx=0,
        attention=gqa_attention_ir,
        mlp=gated_mlp_ir,
        causal_mask=causal_mask,
    )


# =============================================================================
# Input Data Fixtures
# =============================================================================


@pytest.fixture
def hidden_states(small_dims) -> np.ndarray:
    """Random hidden states [batch, seq_len, d_model]."""
    rng = np.random.default_rng(42)
    batch_size = small_dims["batch_size"]
    seq_len = small_dims["seq_len"]
    d_model = small_dims["d_model"]
    return rng.normal(0, 1, (batch_size, seq_len, d_model)).astype(np.float32)


@pytest.fixture
def hidden_states_gqa(gqa_dims) -> np.ndarray:
    """Random hidden states for GQA tests."""
    rng = np.random.default_rng(42)
    batch_size = gqa_dims["batch_size"]
    seq_len = gqa_dims["seq_len"]
    d_model = gqa_dims["d_model"]
    return rng.normal(0, 1, (batch_size, seq_len, d_model)).astype(np.float32)


@pytest.fixture
def causal_mask(small_dims) -> np.ndarray:
    """Causal attention mask [seq_len, seq_len]."""
    seq_len = small_dims["seq_len"]
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


# =============================================================================
# TraceDataset Fixtures
# =============================================================================


@pytest.fixture
def trace_record(small_dims) -> TraceRecord:
    """Single trace record."""
    rng = np.random.default_rng(42)
    seq_len = small_dims["seq_len"]
    d_model = small_dims["d_model"]

    return TraceRecord(
        block_idx=0,
        input_activations=rng.normal(0, 1, (seq_len, d_model)).astype(np.float32),
        output_activations=rng.normal(0, 1, (seq_len, d_model)).astype(np.float32),
        attention_mask=np.tril(np.ones((seq_len, seq_len), dtype=np.float32)),
        prompt_id="test_prompt_0",
    )


@pytest.fixture
def trace_dataset(small_dims) -> TraceDataset:
    """TraceDataset with 5 records."""
    rng = np.random.default_rng(42)
    seq_len = small_dims["seq_len"]
    d_model = small_dims["d_model"]

    dataset = TraceDataset(block_idx=0)

    for i in range(5):
        record = TraceRecord(
            block_idx=0,
            input_activations=rng.normal(0, 1, (seq_len, d_model)).astype(np.float32),
            output_activations=rng.normal(0, 1, (seq_len, d_model)).astype(np.float32),
            attention_mask=np.tril(np.ones((seq_len, seq_len), dtype=np.float32)),
            prompt_id=f"prompt_{i}",
        )
        dataset.add_record(record)

    return dataset


@pytest.fixture
def empty_trace_dataset() -> TraceDataset:
    """Empty TraceDataset."""
    return TraceDataset(block_idx=0)


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_auto_lirpa_unavailable():
    """Mock auto-LiRPA as unavailable."""
    with patch.dict("sys.modules", {"auto_LiRPA": None}):
        yield


@pytest.fixture
def mock_psutil_low_memory():
    """Mock psutil to report low available memory (8 GB)."""
    mock_mem = MagicMock()
    mock_mem.available = 8 * 1024**3  # 8 GB
    with patch("extraction.blockcert.certifier.get_available_memory_gb", return_value=8.0):
        yield


@pytest.fixture
def mock_psutil_high_memory():
    """Mock psutil to report high available memory (64 GB)."""
    with patch("extraction.blockcert.certifier.get_available_memory_gb", return_value=64.0):
        yield


@pytest.fixture
def mock_psutil_unavailable():
    """Mock psutil as unavailable."""
    with patch("extraction.blockcert.certifier.get_available_memory_gb", return_value=-1.0):
        yield


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Temporary directory for output files."""
    output_dir = tmp_path / "blockcert_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =============================================================================
# Helper Functions (available as fixtures)
# =============================================================================


@pytest.fixture
def create_sparse_mask():
    """Factory fixture to create sparse masks."""
    def _create_mask(shape: Tuple[int, ...], sparsity: float, seed: int = 42) -> np.ndarray:
        """
        Create a sparse mask with given sparsity.

        Args:
            shape: Shape of the mask
            sparsity: Fraction of zeros (0.0 = dense, 1.0 = all zeros)
            seed: Random seed

        Returns:
            Binary mask array
        """
        rng = np.random.default_rng(seed)
        mask = rng.random(shape) > sparsity
        return mask.astype(np.float32)

    return _create_mask


@pytest.fixture
def assert_shapes_preserved():
    """Factory fixture to verify output shapes match input shapes."""
    def _assert_shapes(input_arr: np.ndarray, output_arr: np.ndarray):
        assert output_arr.shape == input_arr.shape, (
            f"Shape mismatch: input {input_arr.shape} vs output {output_arr.shape}"
        )
    return _assert_shapes
