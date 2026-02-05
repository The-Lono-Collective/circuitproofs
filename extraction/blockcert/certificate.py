"""
Certificate Artifact Generation for BlockCert

Generates machine-verifiable JSON certificates containing:
- Model and block metadata
- Certification metrics (epsilon, coverage)
- SHA-256 hashes of weight files
- Global error bounds
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

from extraction.blockcert.certifier import CertificationMetrics, LipschitzBounds, compute_global_error_bound


def _validate_path_component(value: Any, field_name: str) -> None:
    """
    Validate a value used in path construction contains no traversal characters.

    Args:
        value: The value to validate (will be converted to string)
        field_name: Name of the field for error messages

    Raises:
        ValueError: If the value contains path traversal characters
    """
    str_val = str(value)
    if ".." in str_val or "/" in str_val or "\\" in str_val:
        raise ValueError(
            f"Invalid {field_name}: contains path traversal characters: {str_val!r}"
        )


@dataclass
class BlockCertificate:
    """Certificate for a single transformer block."""

    block_idx: int

    # Certification metrics
    epsilon: float  # Local error bound
    mae: float  # Mean absolute error
    activation_coverage: float
    loss_coverage: Optional[float]

    # Lipschitz bounds
    K_attn: float
    K_mlp: float
    L_block: float
    K_mlp_method: str
    K_mlp_certified: bool

    # Weight file hashes
    weight_hashes: Dict[str, str]

    # Certification status
    is_certified: bool

    # Thresholds used
    tau_act: float
    tau_loss: float

    def __post_init__(self) -> None:
        """Validate fields to prevent path traversal attacks."""
        if not isinstance(self.block_idx, int) or self.block_idx < 0:
            raise ValueError(
                f"block_idx must be a non-negative integer, got: {self.block_idx!r}"
            )
        for component in self.weight_hashes.keys():
            _validate_path_component(component, "weight_hashes key")


@dataclass
class Certificate:
    """
    Complete certificate for a model's extracted circuits.

    This is the artifact that proves the surrogate matches the original.
    """

    # Model metadata
    model_name: str
    model_hash: Optional[str] = None  # Hash of original model weights

    # Block certificates
    blocks: List[BlockCertificate] = field(default_factory=list)

    # Global metrics
    global_epsilon: float = 0.0  # Global error bound from composition theorem
    total_blocks: int = 0
    certified_blocks: int = 0

    # Extraction metadata
    extraction_method: str = "blockcert"
    extraction_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    calibration_prompts: int = 0
    calibration_tokens: int = 0

    # Thresholds
    tau_act: float = 1e-2
    tau_loss: float = 1e-3
    alpha_act: float = 0.94
    alpha_loss: float = 0.90

    # Version info
    blockcert_version: str = "0.1.0"

    # Master certificate hash (computed on save)
    certificate_hash: Optional[str] = None

    def add_block(
        self,
        metrics: CertificationMetrics,
        weight_hashes: Dict[str, str],
    ) -> None:
        """Add a block certificate from metrics."""
        block_cert = BlockCertificate(
            block_idx=metrics.block_idx,
            epsilon=metrics.epsilon,
            mae=metrics.mae,
            activation_coverage=metrics.activation_coverage,
            loss_coverage=metrics.loss_coverage,
            K_attn=metrics.lipschitz.K_attn if metrics.lipschitz else 0.0,
            K_mlp=metrics.lipschitz.K_mlp if metrics.lipschitz else 0.0,
            L_block=metrics.lipschitz.L_block if metrics.lipschitz else 1.0,
            K_mlp_method=metrics.lipschitz.K_mlp_method if metrics.lipschitz else "unknown",
            K_mlp_certified=metrics.lipschitz.K_mlp_certified if metrics.lipschitz else False,
            weight_hashes=weight_hashes,
            is_certified=metrics.is_certified,
            tau_act=metrics.tau_act,
            tau_loss=metrics.tau_loss,
        )
        self.blocks.append(block_cert)
        self.total_blocks = len(self.blocks)
        self.certified_blocks = sum(1 for b in self.blocks if b.is_certified)

    def compute_global_bound(self, block_metrics: List[CertificationMetrics]) -> None:
        """Compute global error bound from block metrics."""
        self.global_epsilon = compute_global_error_bound(block_metrics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "model_hash": self.model_hash,
            "blocks": [asdict(b) for b in self.blocks],
            "global_epsilon": self.global_epsilon,
            "total_blocks": self.total_blocks,
            "certified_blocks": self.certified_blocks,
            "extraction_method": self.extraction_method,
            "extraction_timestamp": self.extraction_timestamp,
            "calibration_prompts": self.calibration_prompts,
            "calibration_tokens": self.calibration_tokens,
            "thresholds": {
                "tau_act": self.tau_act,
                "tau_loss": self.tau_loss,
                "alpha_act": self.alpha_act,
                "alpha_loss": self.alpha_loss,
            },
            "blockcert_version": self.blockcert_version,
            "certificate_hash": self.certificate_hash,
        }

    def save(self, path: Path) -> str:
        """
        Save certificate to JSON file.

        Returns the certificate hash.
        """
        path = Path(path)

        # Convert to dict (without hash)
        cert_dict = self.to_dict()
        cert_dict["certificate_hash"] = None  # Compute after serialization

        # Compute hash of certificate content
        cert_json = json.dumps(cert_dict, sort_keys=True, indent=2)
        self.certificate_hash = hashlib.sha256(cert_json.encode()).hexdigest()

        # Update and save
        cert_dict["certificate_hash"] = self.certificate_hash
        with open(path, "w") as f:
            json.dump(cert_dict, f, indent=2)

        return self.certificate_hash

    @classmethod
    def load(cls, path: Path) -> "Certificate":
        """Load certificate from JSON file."""
        with open(path) as f:
            data = json.load(f)

        # Reconstruct block certificates
        blocks = [
            BlockCertificate(**block_data)
            for block_data in data.get("blocks", [])
        ]

        thresholds = data.get("thresholds", {})

        return cls(
            model_name=data["model_name"],
            model_hash=data.get("model_hash"),
            blocks=blocks,
            global_epsilon=data.get("global_epsilon", 0.0),
            total_blocks=data.get("total_blocks", 0),
            certified_blocks=data.get("certified_blocks", 0),
            extraction_method=data.get("extraction_method", "blockcert"),
            extraction_timestamp=data.get("extraction_timestamp", ""),
            calibration_prompts=data.get("calibration_prompts", 0),
            calibration_tokens=data.get("calibration_tokens", 0),
            tau_act=thresholds.get("tau_act", 1e-2),
            tau_loss=thresholds.get("tau_loss", 1e-3),
            alpha_act=thresholds.get("alpha_act", 0.94),
            alpha_loss=thresholds.get("alpha_loss", 0.90),
            blockcert_version=data.get("blockcert_version", "0.1.0"),
            certificate_hash=data.get("certificate_hash"),
        )

    def verify_hashes(self, weight_dir: Path) -> Dict[str, bool]:
        """
        Verify that weight file hashes match the certificate.

        Args:
            weight_dir: Directory containing weight files

        Returns:
            Dict mapping file names to verification status (True if hash matches)

        Raises:
            ValueError: If path traversal is detected in file paths
        """
        results = {}
        weight_dir = Path(weight_dir).resolve()

        for block in self.blocks:
            for component, expected_hash in block.weight_hashes.items():
                # Reconstruct expected file path
                file_path = weight_dir / f"block_{block.block_idx}_{component}.npz"

                # Security: verify path stays within weight_dir (prevent path traversal)
                try:
                    file_path.resolve().relative_to(weight_dir)
                except ValueError:
                    raise ValueError(
                        f"Path traversal detected: {file_path} escapes {weight_dir}"
                    )

                if not file_path.exists():
                    results[str(file_path)] = False
                    continue

                # Compute actual hash
                with open(file_path, "rb") as f:
                    actual_hash = hashlib.sha256(f.read()).hexdigest()

                results[str(file_path)] = (actual_hash == expected_hash)

        return results

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"BlockCert Certificate for {self.model_name}",
            "=" * 50,
            f"Extraction timestamp: {self.extraction_timestamp}",
            f"Calibration: {self.calibration_prompts} prompts, {self.calibration_tokens} tokens",
            "",
            "Global Metrics:",
            f"  Global error bound (ε): {self.global_epsilon:.6e}",
            f"  Certified blocks: {self.certified_blocks}/{self.total_blocks}",
            "",
            "Per-Block Metrics:",
        ]

        for block in self.blocks:
            status = "✓" if block.is_certified else "✗"
            bound_type = "certified" if block.K_mlp_certified else "analytic"
            lines.append(
                f"  Block {block.block_idx}: ε={block.epsilon:.6e}, "
                f"cov_act={block.activation_coverage:.2%}, "
                f"K_mlp={block.K_mlp:.2e} ({bound_type}), "
                f"L={block.L_block:.2e} [{status}]"
            )

        # Check if any blocks used analytic fallback
        analytic_blocks = [b for b in self.blocks if not b.K_mlp_certified]
        if analytic_blocks:
            lines.extend([
                "",
                "⚠ Note: Some blocks used analytic Lipschitz estimates (less tight):",
                f"  Blocks with analytic K_mlp: {[b.block_idx for b in analytic_blocks]}",
                "  For certified bounds, run with >= 24GB RAM.",
            ])

        lines.extend([
            "",
            f"Certificate hash: {self.certificate_hash}",
        ])

        return "\n".join(lines)


def generate_certificate(
    model_name: str,
    block_metrics: List[CertificationMetrics],
    block_hashes: List[Dict[str, str]],
    calibration_prompts: int = 0,
    calibration_tokens: int = 0,
    model_hash: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Certificate:
    """
    Generate a complete certificate from block metrics.

    Args:
        model_name: Name of the model
        block_metrics: List of CertificationMetrics for each block
        block_hashes: List of weight hash dicts for each block
        calibration_prompts: Number of calibration prompts used
        calibration_tokens: Total calibration tokens
        model_hash: Optional hash of original model
        output_path: Optional path to save certificate

    Returns:
        Certificate object
    """
    cert = Certificate(
        model_name=model_name,
        model_hash=model_hash,
        calibration_prompts=calibration_prompts,
        calibration_tokens=calibration_tokens,
    )

    # Add block certificates
    for metrics, hashes in zip(block_metrics, block_hashes):
        cert.add_block(metrics, hashes)

    # Compute global bound
    cert.compute_global_bound(block_metrics)

    # Save if path provided
    if output_path is not None:
        cert.save(output_path)

    return cert


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
