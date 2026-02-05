"""
Integration tests for the complete BlockCert pipeline.

Tests end-to-end workflows: MHA + MLP, GQA + SwiGLU, multi-block certification,
and certificate save/load/verify.
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
from extraction.blockcert.interpreter import BlockInterpreter
from extraction.blockcert.certifier import BlockCertifier
from extraction.blockcert.certificate import Certificate, generate_certificate


# =============================================================================
# Fixtures for Integration Tests
# =============================================================================


@pytest.fixture
def create_block_ir():
    """Factory fixture to create BlockIR with configurable parameters.

    Note: Weight shapes follow PyTorch convention [out_features, in_features].
    """
    def _create(
        d_model: int = 64,
        d_ff: int = 256,
        num_heads: int = 4,
        head_dim: int = 16,
        num_kv_heads: int = 4,
        seq_len: int = 8,
        is_gated: bool = False,
        norm_type: str = "layernorm",
        activation: str = "gelu",
        block_idx: int = 0,
        seed: int = 42,
    ) -> BlockIR:
        rng = np.random.default_rng(seed)

        # Create normalization
        norm = NormIR(
            norm_type=norm_type,
            weight=np.ones(d_model, dtype=np.float32),
            bias=np.zeros(d_model, dtype=np.float32) if norm_type == "layernorm" else None,
        )

        # Create attention (PyTorch convention: [out, in])
        scale = np.sqrt(2.0 / (d_model + num_heads * head_dim))
        attention = AttentionIR(
            W_Q=rng.normal(0, scale, (num_heads * head_dim, d_model)).astype(np.float32),
            W_K=rng.normal(0, scale, (num_kv_heads * head_dim, d_model)).astype(np.float32),
            W_V=rng.normal(0, scale, (num_kv_heads * head_dim, d_model)).astype(np.float32),
            W_O=rng.normal(0, scale, (d_model, num_heads * head_dim)).astype(np.float32),
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            pre_norm=norm,
        )

        # Create MLP (PyTorch convention: [out, in])
        mlp_scale = np.sqrt(2.0 / (d_model + d_ff))
        mlp = MLPIR(
            W_1=rng.normal(0, mlp_scale, (d_ff, d_model)).astype(np.float32),
            W_2=rng.normal(0, mlp_scale, (d_model, d_ff)).astype(np.float32),
            W_gate=rng.normal(0, mlp_scale, (d_ff, d_model)).astype(np.float32) if is_gated else None,
            activation=activation,
            pre_norm=norm,
        )

        # Create causal mask
        causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))

        return BlockIR(
            block_idx=block_idx,
            attention=attention,
            mlp=mlp,
            causal_mask=causal_mask,
        )

    return _create


@pytest.fixture
def create_trace_dataset():
    """Factory fixture to create TraceDataset with generated traces."""
    def _create(
        block_ir: BlockIR,
        n_records: int = 10,
        batch_size: int = 1,
        seq_len: int = 8,
        noise_scale: float = 0.01,
        seed: int = 42,
    ) -> TraceDataset:
        rng = np.random.default_rng(seed)
        interpreter = BlockInterpreter()
        d_model = block_ir.attention.d_model

        dataset = TraceDataset(block_idx=block_ir.block_idx)

        for i in range(n_records):
            # Generate random input
            input_act = rng.normal(0, 1, (batch_size, seq_len, d_model)).astype(np.float32)

            # Run through interpreter to get "ground truth" output
            output_act = interpreter.interpret_block(block_ir, input_act)

            # Add small noise to output to simulate approximation error
            output_act = output_act + rng.normal(0, noise_scale, output_act.shape).astype(np.float32)

            # Create record (squeeze batch dim if batch_size=1)
            record = TraceRecord(
                block_idx=block_ir.block_idx,
                input_activations=input_act.squeeze(0),
                output_activations=output_act.squeeze(0),
                prompt_id=f"prompt_{i}",
            )
            dataset.add_record(record)

        return dataset

    return _create


# =============================================================================
# MHA + Standard MLP Pipeline Tests
# =============================================================================


class TestMHAStandardMLPPipeline:
    """Integration tests for MHA attention + standard MLP pipeline."""

    def test_full_pipeline(self, create_block_ir, create_trace_dataset, temp_output_dir):
        """Test complete pipeline: create IR, interpret, certify, generate certificate."""
        # Create block IR
        block_ir = create_block_ir(
            d_model=64,
            d_ff=256,
            num_heads=4,
            head_dim=16,
            is_gated=False,
            activation="gelu",
        )

        # Create trace dataset
        trace_dataset = create_trace_dataset(block_ir, n_records=5, noise_scale=0.005)

        # Save block IR and get hashes
        hashes = block_ir.save(temp_output_dir)

        # Certify block
        certifier = BlockCertifier(use_auto_lirpa=False)
        metrics = certifier.certify_block(block_ir, trace_dataset)

        # Generate certificate
        cert = generate_certificate(
            model_name="test_mha_mlp",
            block_metrics=[metrics],
            block_hashes=[hashes],
            output_path=temp_output_dir / "certificate.json",
        )

        # Verify certificate
        assert cert.total_blocks == 1
        assert cert.global_epsilon > 0
        assert (temp_output_dir / "certificate.json").exists()

        # Verify hashes match
        hash_results = cert.verify_hashes(temp_output_dir)
        for path, matches in hash_results.items():
            assert matches, f"Hash mismatch for {path}"

    def test_interpreter_output_matches_traces(self, create_block_ir, small_dims):
        """Test interpreter output has same shape as trace dataset."""
        block_ir = create_block_ir(
            d_model=small_dims["d_model"],
            seq_len=small_dims["seq_len"],
        )

        interpreter = BlockInterpreter()
        batch_size = small_dims["batch_size"]
        seq_len = small_dims["seq_len"]
        d_model = small_dims["d_model"]

        hidden_states = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        output = interpreter.interpret_block(block_ir, hidden_states)

        assert output.shape == hidden_states.shape


# =============================================================================
# GQA + SwiGLU Pipeline Tests
# =============================================================================


class TestGQASwiGLUPipeline:
    """Integration tests for GQA attention + SwiGLU MLP pipeline."""

    def test_full_pipeline(self, create_block_ir, create_trace_dataset, temp_output_dir):
        """Test GQA + SwiGLU pipeline end-to-end."""
        # Create block IR with GQA and SwiGLU
        block_ir = create_block_ir(
            d_model=64,
            d_ff=256,
            num_heads=8,
            head_dim=8,
            num_kv_heads=2,  # GQA: 4 query heads per KV head
            is_gated=True,
            activation="swiglu",
            norm_type="rmsnorm",
        )

        # Verify GQA configuration
        assert block_ir.attention.is_gqa
        assert block_ir.attention.kv_head_repeat == 4
        assert block_ir.mlp.is_gated

        # Create trace dataset
        trace_dataset = create_trace_dataset(block_ir, n_records=5, noise_scale=0.005)

        # Save and get hashes
        hashes = block_ir.save(temp_output_dir)

        # Certify
        certifier = BlockCertifier(use_auto_lirpa=False)
        metrics = certifier.certify_block(block_ir, trace_dataset)

        # Generate certificate
        cert = generate_certificate(
            model_name="test_gqa_swiglu",
            block_metrics=[metrics],
            block_hashes=[hashes],
            output_path=temp_output_dir / "certificate.json",
        )

        # Verify
        assert cert.total_blocks == 1
        assert cert.global_epsilon > 0

    def test_gqa_kv_repetition_consistent(self, create_block_ir):
        """Test GQA KV head repetition produces consistent results."""
        block_ir = create_block_ir(
            d_model=64,
            num_heads=8,
            head_dim=8,
            num_kv_heads=2,
        )

        interpreter = BlockInterpreter()
        hidden_states = np.random.randn(2, 8, 64).astype(np.float32)

        # Run multiple times
        output1 = interpreter.interpret_block(block_ir, hidden_states)
        output2 = interpreter.interpret_block(block_ir, hidden_states)

        np.testing.assert_array_equal(output1, output2)


# =============================================================================
# Multi-Block Pipeline Tests
# =============================================================================


class TestMultiBlockPipeline:
    """Integration tests for multi-block certification pipeline."""

    def test_three_block_pipeline(self, create_block_ir, create_trace_dataset, temp_output_dir):
        """Test certification of 3 sequential blocks."""
        n_blocks = 3
        blocks = []
        datasets = []
        hashes_list = []

        # Create blocks with different seeds for variety
        for i in range(n_blocks):
            block_ir = create_block_ir(
                block_idx=i,
                seed=42 + i,
            )
            blocks.append(block_ir)

            # Create traces for this block
            dataset = create_trace_dataset(
                block_ir, n_records=5, noise_scale=0.005 * (i + 1), seed=100 + i
            )
            datasets.append(dataset)

            # Save block and collect hashes
            block_dir = temp_output_dir / f"block_{i}"
            block_dir.mkdir()
            hashes = block_ir.save(block_dir)
            hashes_list.append(hashes)

        # Certify all blocks
        certifier = BlockCertifier(use_auto_lirpa=False)
        metrics_list = []

        for block_ir, dataset in zip(blocks, datasets):
            metrics = certifier.certify_block(block_ir, dataset)
            metrics_list.append(metrics)

        # Generate combined certificate
        cert = generate_certificate(
            model_name="test_3_blocks",
            block_metrics=metrics_list,
            block_hashes=hashes_list,
            output_path=temp_output_dir / "certificate.json",
        )

        # Verify
        assert cert.total_blocks == 3
        assert len(cert.blocks) == 3
        assert cert.global_epsilon > 0

        # Global bound should be larger than any single epsilon
        single_max = max(m.epsilon for m in metrics_list)
        assert cert.global_epsilon >= single_max

    def test_global_bound_composition(self, create_block_ir, create_trace_dataset):
        """Test global error bound composition formula."""
        # Create blocks with known error characteristics
        blocks = []
        metrics_list = []

        certifier = BlockCertifier(use_auto_lirpa=False)

        for i in range(3):
            block_ir = create_block_ir(block_idx=i, seed=42 + i)
            blocks.append(block_ir)

            dataset = create_trace_dataset(block_ir, n_records=5, noise_scale=0.01)
            metrics = certifier.certify_block(block_ir, dataset)
            metrics_list.append(metrics)

        # Generate certificate (computes global bound)
        cert = generate_certificate(
            model_name="test",
            block_metrics=metrics_list,
            block_hashes=[{} for _ in range(3)],
        )

        # Manually verify composition formula
        # global = sum_i (epsilon_i * prod_{j>i} L_j)
        expected = 0.0
        for i, metrics in enumerate(metrics_list):
            lipschitz_prod = 1.0
            for j in range(i + 1, len(metrics_list)):
                if metrics_list[j].lipschitz:
                    lipschitz_prod *= metrics_list[j].lipschitz.L_block
            expected += metrics.epsilon * lipschitz_prod

        np.testing.assert_almost_equal(cert.global_epsilon, expected)


# =============================================================================
# Certificate Verification After Save/Load Tests
# =============================================================================


class TestCertificateVerification:
    """Tests for certificate verification after save/load cycle."""

    def test_save_load_verify_hashes(self, create_block_ir, create_trace_dataset, temp_output_dir):
        """Test full cycle: save weights, generate cert, reload, verify hashes."""
        # Create and save block
        block_ir = create_block_ir()
        weight_dir = temp_output_dir / "weights"
        weight_dir.mkdir()
        hashes = block_ir.save(weight_dir)

        # Create traces and certify
        dataset = create_trace_dataset(block_ir, n_records=5)
        certifier = BlockCertifier(use_auto_lirpa=False)
        metrics = certifier.certify_block(block_ir, dataset)

        # Generate and save certificate
        cert_path = temp_output_dir / "certificate.json"
        cert = generate_certificate(
            model_name="test_verify",
            block_metrics=[metrics],
            block_hashes=[hashes],
            output_path=cert_path,
        )

        # Load certificate
        loaded_cert = Certificate.load(cert_path)

        # Verify hashes against weight files
        hash_results = loaded_cert.verify_hashes(weight_dir)

        # All hashes should match
        assert len(hash_results) > 0
        for path, matches in hash_results.items():
            assert matches, f"Hash verification failed for {path}"

    def test_detect_tampered_weights(self, create_block_ir, create_trace_dataset, temp_output_dir):
        """Test certificate detects tampered weight files."""
        # Create and save block
        block_ir = create_block_ir()
        weight_dir = temp_output_dir / "weights"
        weight_dir.mkdir()
        hashes = block_ir.save(weight_dir)

        # Create and save certificate
        dataset = create_trace_dataset(block_ir, n_records=5)
        certifier = BlockCertifier(use_auto_lirpa=False)
        metrics = certifier.certify_block(block_ir, dataset)

        cert_path = temp_output_dir / "certificate.json"
        generate_certificate(
            model_name="test_tamper",
            block_metrics=[metrics],
            block_hashes=[hashes],
            output_path=cert_path,
        )

        # Tamper with weight file
        attn_path = weight_dir / "block_0_attention.npz"
        np.savez(attn_path, tampered=np.array([999, 999, 999]))

        # Load certificate and verify
        loaded_cert = Certificate.load(cert_path)
        hash_results = loaded_cert.verify_hashes(weight_dir)

        # Should detect tampering
        assert not hash_results[str(attn_path)]


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestPipelineErrorHandling:
    """Tests for error handling in the pipeline."""

    def test_empty_trace_dataset_handling(self, create_block_ir):
        """Test certifier handles edge case of minimal trace data."""
        block_ir = create_block_ir()

        # Create minimal trace dataset (1 record)
        rng = np.random.default_rng(42)
        dataset = TraceDataset(block_idx=0)
        record = TraceRecord(
            block_idx=0,
            input_activations=rng.normal(0, 1, (8, 64)).astype(np.float32),
            output_activations=rng.normal(0, 1, (8, 64)).astype(np.float32),
        )
        dataset.add_record(record)

        # Should still work
        certifier = BlockCertifier(use_auto_lirpa=False)
        metrics = certifier.certify_block(block_ir, dataset)

        assert metrics.epsilon >= 0
        assert len(metrics.per_token_errors) == 8  # seq_len tokens

    def test_large_error_detection(self, create_block_ir):
        """Test certifier correctly identifies large approximation errors."""
        block_ir = create_block_ir()

        # Create trace dataset with large errors (random outputs, not from interpreter)
        rng = np.random.default_rng(42)
        dataset = TraceDataset(block_idx=0)

        for i in range(5):
            record = TraceRecord(
                block_idx=0,
                input_activations=rng.normal(0, 1, (8, 64)).astype(np.float32),
                output_activations=rng.normal(0, 10, (8, 64)).astype(np.float32),  # Large random
            )
            dataset.add_record(record)

        certifier = BlockCertifier(use_auto_lirpa=False)
        metrics = certifier.certify_block(block_ir, dataset)

        # Large errors should result in low coverage
        assert metrics.activation_coverage < 0.5  # Most tokens should fail threshold
        assert not metrics.is_certified  # Should not be certified
