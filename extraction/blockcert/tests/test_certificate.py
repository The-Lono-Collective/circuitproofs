"""
Tests for BlockCert Certificate generation.

Tests the Certificate, BlockCertificate, and NumpyEncoder classes.
"""

import json
import numpy as np
import pytest
from pathlib import Path

from extraction.blockcert.certificate import (
    Certificate,
    BlockCertificate,
    NumpyEncoder,
    generate_certificate,
)
from extraction.blockcert.certifier import (
    CertificationMetrics,
    LipschitzBounds,
)


# =============================================================================
# BlockCertificate Tests
# =============================================================================


class TestBlockCertificate:
    """Tests for BlockCertificate dataclass."""

    def test_creation(self):
        """Test BlockCertificate creation with all fields."""
        cert = BlockCertificate(
            block_idx=0,
            epsilon=0.01,
            mae=0.005,
            activation_coverage=0.95,
            loss_coverage=0.92,
            K_attn=2.5,
            K_mlp=3.0,
            L_block=10.5,
            K_mlp_method="auto_lirpa",
            K_mlp_certified=True,
            weight_hashes={"attention": "abc123", "mlp": "def456"},
            is_certified=True,
            tau_act=1e-2,
            tau_loss=1e-3,
        )

        assert cert.block_idx == 0
        assert cert.epsilon == 0.01
        assert cert.mae == 0.005
        assert cert.activation_coverage == 0.95
        assert cert.is_certified
        assert len(cert.weight_hashes) == 2

    def test_optional_loss_coverage(self):
        """Test BlockCertificate with None loss coverage."""
        cert = BlockCertificate(
            block_idx=0,
            epsilon=0.01,
            mae=0.005,
            activation_coverage=0.95,
            loss_coverage=None,
            K_attn=2.5,
            K_mlp=3.0,
            L_block=10.5,
            K_mlp_method="analytic_estimate",
            K_mlp_certified=False,
            weight_hashes={},
            is_certified=True,
            tau_act=1e-2,
            tau_loss=1e-3,
        )

        assert cert.loss_coverage is None


# =============================================================================
# Certificate Tests
# =============================================================================


class TestCertificate:
    """Tests for Certificate dataclass."""

    def test_creation_minimal(self):
        """Test Certificate creation with minimal fields."""
        cert = Certificate(model_name="test_model")

        assert cert.model_name == "test_model"
        assert cert.total_blocks == 0
        assert cert.certified_blocks == 0
        assert cert.blocks == []
        assert cert.global_epsilon == 0.0

    def test_creation_with_all_fields(self):
        """Test Certificate creation with all fields."""
        cert = Certificate(
            model_name="test_model",
            model_hash="abc123",
            global_epsilon=0.05,
            extraction_method="blockcert",
            calibration_prompts=100,
            calibration_tokens=50000,
        )

        assert cert.model_name == "test_model"
        assert cert.model_hash == "abc123"
        assert cert.global_epsilon == 0.05
        assert cert.calibration_prompts == 100

    def test_add_block_updates_counts(self):
        """Test add_block updates total_blocks and certified_blocks."""
        cert = Certificate(model_name="test_model")

        # Create certification metrics
        metrics1 = CertificationMetrics(
            block_idx=0,
            epsilon=0.01,
            mae=0.005,
            per_token_errors=np.array([0.01]),
            activation_coverage=0.95,
            lipschitz=LipschitzBounds(K_attn=1.0, K_mlp=2.0, L_block=4.0),
        )
        metrics1.is_certified = True

        metrics2 = CertificationMetrics(
            block_idx=1,
            epsilon=0.02,
            mae=0.01,
            per_token_errors=np.array([0.02]),
            activation_coverage=0.90,  # Below threshold
            lipschitz=LipschitzBounds(K_attn=1.0, K_mlp=2.0, L_block=4.0),
        )
        metrics2.is_certified = False

        cert.add_block(metrics1, {"attention": "hash1", "mlp": "hash2"})
        cert.add_block(metrics2, {"attention": "hash3", "mlp": "hash4"})

        assert cert.total_blocks == 2
        assert cert.certified_blocks == 1
        assert len(cert.blocks) == 2

    def test_add_block_handles_no_lipschitz(self):
        """Test add_block handles metrics without Lipschitz bounds."""
        cert = Certificate(model_name="test_model")

        metrics = CertificationMetrics(
            block_idx=0,
            epsilon=0.01,
            mae=0.005,
            per_token_errors=np.array([0.01]),
            activation_coverage=0.95,
            lipschitz=None,  # No Lipschitz
        )
        metrics.is_certified = True

        cert.add_block(metrics, {})

        assert cert.total_blocks == 1
        assert cert.blocks[0].K_attn == 0.0
        assert cert.blocks[0].L_block == 1.0  # Default

    def test_compute_global_bound(self):
        """Test compute_global_bound from block metrics."""
        cert = Certificate(model_name="test_model")

        metrics = [
            CertificationMetrics(
                block_idx=i,
                epsilon=0.01,
                mae=0.005,
                per_token_errors=np.array([0.01]),
                activation_coverage=0.95,
                lipschitz=LipschitzBounds(K_attn=1.0, K_mlp=1.0, L_block=2.0),
            )
            for i in range(3)
        ]

        cert.compute_global_bound(metrics)

        # Expected: 0.01 * 4 + 0.01 * 2 + 0.01 * 1 = 0.07
        expected = 0.01 * 4 + 0.01 * 2 + 0.01
        np.testing.assert_almost_equal(cert.global_epsilon, expected)

    def test_to_dict(self):
        """Test to_dict includes all fields."""
        cert = Certificate(
            model_name="test_model",
            model_hash="abc123",
            global_epsilon=0.05,
        )

        d = cert.to_dict()

        assert d["model_name"] == "test_model"
        assert d["model_hash"] == "abc123"
        assert d["global_epsilon"] == 0.05
        assert "blocks" in d
        assert "thresholds" in d
        assert "blockcert_version" in d

    def test_to_dict_with_blocks(self):
        """Test to_dict serializes blocks correctly."""
        cert = Certificate(model_name="test_model")

        metrics = CertificationMetrics(
            block_idx=0,
            epsilon=0.01,
            mae=0.005,
            per_token_errors=np.array([0.01]),
            activation_coverage=0.95,
            lipschitz=LipschitzBounds(K_attn=1.0, K_mlp=2.0, L_block=4.0),
        )
        metrics.is_certified = True
        cert.add_block(metrics, {"attention": "hash1"})

        d = cert.to_dict()

        assert len(d["blocks"]) == 1
        assert d["blocks"][0]["block_idx"] == 0
        assert d["blocks"][0]["epsilon"] == 0.01


# =============================================================================
# Certificate Save/Load Tests
# =============================================================================


class TestCertificateSaveLoad:
    """Tests for Certificate save and load functionality."""

    def test_save_creates_file(self, temp_output_dir):
        """Test save creates valid JSON file."""
        cert = Certificate(model_name="test_model")
        path = temp_output_dir / "certificate.json"

        hash_value = cert.save(path)

        assert path.exists()
        assert hash_value is not None
        assert len(hash_value) == 64  # SHA-256 hex

    def test_save_returns_hash(self, temp_output_dir):
        """Test save returns certificate hash."""
        cert = Certificate(model_name="test_model")
        path = temp_output_dir / "certificate.json"

        hash_value = cert.save(path)

        assert cert.certificate_hash == hash_value

    def test_load_reconstructs_certificate(self, temp_output_dir):
        """Test load reconstructs certificate from JSON."""
        cert = Certificate(
            model_name="test_model",
            model_hash="abc123",
            global_epsilon=0.05,
            calibration_prompts=100,
        )
        path = temp_output_dir / "certificate.json"
        cert.save(path)

        loaded = Certificate.load(path)

        assert loaded.model_name == cert.model_name
        assert loaded.model_hash == cert.model_hash
        assert loaded.global_epsilon == cert.global_epsilon
        assert loaded.calibration_prompts == cert.calibration_prompts

    def test_save_load_roundtrip_with_blocks(self, temp_output_dir):
        """Test save/load roundtrip preserves blocks."""
        cert = Certificate(model_name="test_model")

        metrics = CertificationMetrics(
            block_idx=0,
            epsilon=0.01,
            mae=0.005,
            per_token_errors=np.array([0.01]),
            activation_coverage=0.95,
            lipschitz=LipschitzBounds(K_attn=1.0, K_mlp=2.0, L_block=4.0),
        )
        metrics.is_certified = True
        cert.add_block(metrics, {"attention": "hash1", "mlp": "hash2"})

        path = temp_output_dir / "certificate.json"
        cert.save(path)

        loaded = Certificate.load(path)

        assert len(loaded.blocks) == 1
        assert loaded.blocks[0].block_idx == 0
        assert loaded.blocks[0].epsilon == 0.01
        assert loaded.blocks[0].weight_hashes == {"attention": "hash1", "mlp": "hash2"}

    def test_load_preserves_thresholds(self, temp_output_dir):
        """Test load preserves threshold values."""
        cert = Certificate(
            model_name="test_model",
            tau_act=0.05,
            tau_loss=0.005,
            alpha_act=0.90,
            alpha_loss=0.85,
        )
        path = temp_output_dir / "certificate.json"
        cert.save(path)

        loaded = Certificate.load(path)

        assert loaded.tau_act == 0.05
        assert loaded.tau_loss == 0.005
        assert loaded.alpha_act == 0.90
        assert loaded.alpha_loss == 0.85


# =============================================================================
# Hash Verification Tests
# =============================================================================


class TestHashVerification:
    """Tests for weight hash verification."""

    def test_verify_hashes_all_match(self, temp_output_dir):
        """Test verify_hashes returns True for matching hashes."""
        # Create actual weight files
        weight_dir = temp_output_dir / "weights"
        weight_dir.mkdir()

        attn_path = weight_dir / "block_0_attention.npz"
        mlp_path = weight_dir / "block_0_mlp.npz"

        # Save some data
        np.savez(attn_path, data=np.array([1, 2, 3]))
        np.savez(mlp_path, data=np.array([4, 5, 6]))

        # Compute actual hashes
        import hashlib
        with open(attn_path, "rb") as f:
            attn_hash = hashlib.sha256(f.read()).hexdigest()
        with open(mlp_path, "rb") as f:
            mlp_hash = hashlib.sha256(f.read()).hexdigest()

        # Create certificate with correct hashes
        cert = Certificate(model_name="test_model")
        block_cert = BlockCertificate(
            block_idx=0,
            epsilon=0.01,
            mae=0.005,
            activation_coverage=0.95,
            loss_coverage=None,
            K_attn=1.0,
            K_mlp=2.0,
            L_block=4.0,
            K_mlp_method="analytic_estimate",
            K_mlp_certified=False,
            weight_hashes={"attention": attn_hash, "mlp": mlp_hash},
            is_certified=True,
            tau_act=1e-2,
            tau_loss=1e-3,
        )
        cert.blocks.append(block_cert)

        results = cert.verify_hashes(weight_dir)

        assert all(results.values())

    def test_verify_hashes_mismatch(self, temp_output_dir):
        """Test verify_hashes detects mismatched hashes."""
        weight_dir = temp_output_dir / "weights"
        weight_dir.mkdir()

        attn_path = weight_dir / "block_0_attention.npz"
        np.savez(attn_path, data=np.array([1, 2, 3]))

        # Create certificate with wrong hash
        cert = Certificate(model_name="test_model")
        block_cert = BlockCertificate(
            block_idx=0,
            epsilon=0.01,
            mae=0.005,
            activation_coverage=0.95,
            loss_coverage=None,
            K_attn=1.0,
            K_mlp=2.0,
            L_block=4.0,
            K_mlp_method="analytic_estimate",
            K_mlp_certified=False,
            weight_hashes={"attention": "wrong_hash_value"},
            is_certified=True,
            tau_act=1e-2,
            tau_loss=1e-3,
        )
        cert.blocks.append(block_cert)

        results = cert.verify_hashes(weight_dir)

        assert not results[str(attn_path)]

    def test_verify_hashes_missing_file(self, temp_output_dir):
        """Test verify_hashes returns False for missing files."""
        weight_dir = temp_output_dir / "weights"
        weight_dir.mkdir()

        cert = Certificate(model_name="test_model")
        block_cert = BlockCertificate(
            block_idx=0,
            epsilon=0.01,
            mae=0.005,
            activation_coverage=0.95,
            loss_coverage=None,
            K_attn=1.0,
            K_mlp=2.0,
            L_block=4.0,
            K_mlp_method="analytic_estimate",
            K_mlp_certified=False,
            weight_hashes={"attention": "some_hash"},
            is_certified=True,
            tau_act=1e-2,
            tau_loss=1e-3,
        )
        cert.blocks.append(block_cert)

        results = cert.verify_hashes(weight_dir)

        expected_path = str(weight_dir / "block_0_attention.npz")
        assert not results[expected_path]


# =============================================================================
# Summary Tests
# =============================================================================


class TestCertificateSummary:
    """Tests for Certificate summary generation."""

    def test_summary_includes_model_name(self):
        """Test summary includes model name."""
        cert = Certificate(model_name="my_test_model")
        summary = cert.summary()

        assert "my_test_model" in summary

    def test_summary_includes_global_epsilon(self):
        """Test summary includes global error bound."""
        cert = Certificate(model_name="test_model", global_epsilon=0.0123)
        summary = cert.summary()

        assert "1.23" in summary or "0.0123" in summary  # Scientific or decimal notation

    def test_summary_includes_block_count(self):
        """Test summary includes block certification count."""
        cert = Certificate(model_name="test_model")
        cert.total_blocks = 5
        cert.certified_blocks = 3

        summary = cert.summary()

        assert "3/5" in summary or "3" in summary

    def test_summary_notes_analytic_fallback(self):
        """Test summary notes blocks with analytic K_mlp."""
        cert = Certificate(model_name="test_model")

        block_cert = BlockCertificate(
            block_idx=0,
            epsilon=0.01,
            mae=0.005,
            activation_coverage=0.95,
            loss_coverage=None,
            K_attn=1.0,
            K_mlp=2.0,
            L_block=4.0,
            K_mlp_method="analytic_estimate",
            K_mlp_certified=False,  # Analytic fallback
            weight_hashes={},
            is_certified=True,
            tau_act=1e-2,
            tau_loss=1e-3,
        )
        cert.blocks.append(block_cert)
        cert.total_blocks = 1
        cert.certified_blocks = 1

        summary = cert.summary()

        assert "analytic" in summary.lower()


# =============================================================================
# generate_certificate Function Tests
# =============================================================================


class TestGenerateCertificate:
    """Tests for generate_certificate function."""

    def test_generates_certificate(self):
        """Test generate_certificate creates Certificate."""
        metrics = [
            CertificationMetrics(
                block_idx=0,
                epsilon=0.01,
                mae=0.005,
                per_token_errors=np.array([0.01]),
                activation_coverage=0.95,
                lipschitz=LipschitzBounds(K_attn=1.0, K_mlp=2.0, L_block=4.0),
            )
        ]
        metrics[0].is_certified = True

        hashes = [{"attention": "hash1", "mlp": "hash2"}]

        cert = generate_certificate(
            model_name="test_model",
            block_metrics=metrics,
            block_hashes=hashes,
        )

        assert isinstance(cert, Certificate)
        assert cert.model_name == "test_model"
        assert len(cert.blocks) == 1

    def test_generates_with_calibration_info(self):
        """Test generate_certificate includes calibration info."""
        metrics = [
            CertificationMetrics(
                block_idx=0,
                epsilon=0.01,
                mae=0.005,
                per_token_errors=np.array([0.01]),
                activation_coverage=0.95,
            )
        ]
        metrics[0].is_certified = True

        cert = generate_certificate(
            model_name="test_model",
            block_metrics=metrics,
            block_hashes=[{}],
            calibration_prompts=100,
            calibration_tokens=50000,
        )

        assert cert.calibration_prompts == 100
        assert cert.calibration_tokens == 50000

    def test_generates_and_saves(self, temp_output_dir):
        """Test generate_certificate saves when output_path provided."""
        metrics = [
            CertificationMetrics(
                block_idx=0,
                epsilon=0.01,
                mae=0.005,
                per_token_errors=np.array([0.01]),
                activation_coverage=0.95,
            )
        ]
        metrics[0].is_certified = True

        path = temp_output_dir / "generated_cert.json"

        cert = generate_certificate(
            model_name="test_model",
            block_metrics=metrics,
            block_hashes=[{}],
            output_path=path,
        )

        assert path.exists()
        assert cert.certificate_hash is not None

    def test_computes_global_bound(self):
        """Test generate_certificate computes global error bound."""
        metrics = [
            CertificationMetrics(
                block_idx=i,
                epsilon=0.01,
                mae=0.005,
                per_token_errors=np.array([0.01]),
                activation_coverage=0.95,
                lipschitz=LipschitzBounds(K_attn=1.0, K_mlp=1.0, L_block=2.0),
            )
            for i in range(2)
        ]
        for m in metrics:
            m.is_certified = True

        cert = generate_certificate(
            model_name="test_model",
            block_metrics=metrics,
            block_hashes=[{}, {}],
        )

        # Expected: 0.01 * 2 + 0.01 * 1 = 0.03
        assert cert.global_epsilon > 0


# =============================================================================
# NumpyEncoder Tests
# =============================================================================


class TestNumpyEncoder:
    """Tests for NumpyEncoder JSON encoder."""

    def test_handles_int64(self):
        """Test NumpyEncoder handles numpy int64."""
        data = {"value": np.int64(42)}
        result = json.dumps(data, cls=NumpyEncoder)

        assert '"value": 42' in result

    def test_handles_float32(self):
        """Test NumpyEncoder handles numpy float32."""
        data = {"value": np.float32(3.14)}
        result = json.dumps(data, cls=NumpyEncoder)

        assert "3.14" in result

    def test_handles_float64(self):
        """Test NumpyEncoder handles numpy float64."""
        data = {"value": np.float64(2.718)}
        result = json.dumps(data, cls=NumpyEncoder)

        assert "2.718" in result

    def test_handles_array(self):
        """Test NumpyEncoder handles numpy arrays."""
        data = {"values": np.array([1, 2, 3])}
        result = json.dumps(data, cls=NumpyEncoder)

        assert "[1, 2, 3]" in result

    def test_handles_2d_array(self):
        """Test NumpyEncoder handles 2D numpy arrays."""
        data = {"matrix": np.array([[1, 2], [3, 4]])}
        result = json.dumps(data, cls=NumpyEncoder)

        parsed = json.loads(result)
        assert parsed["matrix"] == [[1, 2], [3, 4]]

    def test_falls_back_for_other_types(self):
        """Test NumpyEncoder falls back for non-numpy types."""
        data = {"value": "string"}
        result = json.dumps(data, cls=NumpyEncoder)

        assert '"value": "string"' in result
