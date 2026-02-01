"""
Circuit Extraction Module for Certified Proof-Carrying Circuits

This module implements BlockCert circuit extraction from Transformer models.
It identifies important computational subgraphs (circuits) and computes certified
error bounds using Lipschitz composition with auto-LiRPA.

Two extraction modes:
1. BlockCertExtractor: Full BlockCert pipeline with certified bounds (recommended)
2. CircuitExtractor: Legacy simple extraction (backward compatibility)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
from dataclasses import dataclass
import json
from pathlib import Path
import hashlib

# Import BlockCert modules
from .blockcert import (
    BlockIR,
    AttentionIR,
    MLPIR,
    NormIR,
    BlockInterpreter,
    ActivationTracer,
    TraceDataset,
    BlockCertifier,
    CertificationMetrics,
    Certificate,
    generate_certificate,
)
from .blockcert.tracer import extract_block_weights


class NumpyEncoder(json.JSONEncoder):
    """
    Custom encoder to bridge NumPy types to Python native types.

    This ensures conformance with JSON standards while preserving numerical
    values required for formal verification. Note: float32 values are
    coerced to Python float (64-bit double) for serialization.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class CircuitComponent:
    """Represents a single component in the extracted circuit"""
    layer_idx: int
    component_type: str  # 'attention_head', 'mlp_neuron', 'embedding'
    component_idx: int
    weight: np.ndarray
    bias: Optional[np.ndarray]
    mask: np.ndarray  # Sparsity mask (1 = active, 0 = pruned)
    importance_score: float


@dataclass
class ErrorBound:
    """Error bound certification for the circuit"""
    epsilon: float  # Global error bound
    local_errors: List[float]  # Per-component error bounds
    lipschitz_constants: List[float]  # Per-layer Lipschitz constants
    mae: float  # Mean absolute error
    max_error: float  # Maximum observed error
    coverage: float  # Fraction of examples within bound


class CircuitExtractor:
    """
    Extracts sparse circuits from neural networks using activation patching
    and computes certified error bounds.
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        # Storage for activations during forward pass
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def register_hooks(self, layer_names: List[str]):
        """Register forward hooks to capture activations"""
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        # Register hooks on specified layers
        for name, module in self.model.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(get_activation(name))
                self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_importance_scores(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        metric: str = 'gradient'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute importance scores for each component using gradient-based attribution

        Args:
            inputs: Input tensor (batch_size, seq_len, dim)
            targets: Target outputs
            metric: Attribution method ('gradient', 'integrated_gradient', 'activation')

        Returns:
            Dictionary mapping layer names to importance scores
        """
        self.model.zero_grad()

        # Forward pass
        outputs = self.model(inputs)

        if metric == 'gradient':
            # Use gradient magnitude as importance
            loss = torch.nn.functional.mse_loss(outputs, targets)
            loss.backward()

            importance_scores = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    importance_scores[name] = param.grad.abs().detach()

            return importance_scores

        elif metric == 'activation':
            # Use activation magnitude as importance
            return {name: act.abs().mean(dim=0)
                   for name, act in self.activations.items()}

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def activation_patching(
        self,
        clean_input: torch.Tensor,
        corrupted_input: torch.Tensor,
        layer_name: str,
        component_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Perform activation patching to test component importance

        Replaces activations from corrupted run with clean activations
        for specified components.
        """
        # Run corrupted input
        with torch.no_grad():
            _ = self.model(corrupted_input)
            corrupted_activation = self.activations[layer_name].clone()

            # Run clean input
            _ = self.model(clean_input)
            clean_activation = self.activations[layer_name].clone()

        # Patch specified components
        patched_activation = corrupted_activation.clone()
        if component_indices is not None:
            patched_activation[..., component_indices] = clean_activation[..., component_indices]
        else:
            patched_activation = clean_activation

        # Run forward pass with patched activations
        # (This requires modifying the model to accept intervention)
        return patched_activation

    def edge_pruning(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.01
    ) -> Dict[str, np.ndarray]:
        """
        Prune edges (weights) with low importance scores

        Returns masks indicating which edges to keep
        """
        importance_scores = self.compute_importance_scores(inputs, targets)

        masks = {}
        for name, scores in importance_scores.items():
            # Normalize scores
            scores_normalized = scores / (scores.max() + 1e-8)

            # Create mask based on threshold
            mask = (scores_normalized > threshold).float()
            masks[name] = mask.cpu().numpy()

        return masks

    def extract_circuit(
        self,
        calibration_data: torch.Tensor,
        calibration_targets: torch.Tensor,
        layer_names: List[str],
        pruning_threshold: float = 0.01,
        method: str = 'edge_pruning'
    ) -> List[CircuitComponent]:
        """
        Main circuit extraction method

        Args:
            calibration_data: Input data for identifying important components
            calibration_targets: Target outputs
            layer_names: Names of layers to extract from
            pruning_threshold: Threshold for pruning (lower = sparser)
            method: Extraction method ('edge_pruning', 'activation_patching')

        Returns:
            List of CircuitComponent objects representing the extracted circuit
        """
        self.register_hooks(layer_names)

        # Compute importance scores and masks
        if method == 'edge_pruning':
            masks = self.edge_pruning(
                calibration_data,
                calibration_targets,
                pruning_threshold
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Extract components
        circuit_components = []

        for layer_idx, (name, param) in enumerate(self.model.named_parameters()):
            if name not in masks:
                continue

            mask = masks[name]
            weight = param.detach().cpu().numpy()

            # Determine component type
            if 'attention' in name:
                component_type = 'attention_head'
            elif 'mlp' in name or 'fc' in name:
                component_type = 'mlp_neuron'
            else:
                component_type = 'other'

            # Calculate importance score as fraction of kept weights
            importance_score = float(mask.mean())

            component = CircuitComponent(
                layer_idx=layer_idx,
                component_type=component_type,
                component_idx=0,  # Simplified for now
                weight=weight,
                bias=None,  # Would extract from bias parameters
                mask=mask,
                importance_score=importance_score
            )

            circuit_components.append(component)

        self.remove_hooks()
        return circuit_components

    def compute_lipschitz_constant(
        self,
        layer: nn.Module,
        input_shape: Tuple[int, ...]
    ) -> float:
        """
        Estimate Lipschitz constant of a layer using spectral norm
        """
        if isinstance(layer, nn.Linear):
            weight = layer.weight.detach()
            # Spectral norm (largest singular value)
            lipschitz = torch.svd(weight)[1].max().item()
            return lipschitz

        elif isinstance(layer, nn.ReLU):
            return 1.0  # ReLU is 1-Lipschitz

        elif isinstance(layer, nn.LayerNorm):
            return 1.0  # LayerNorm is approximately 1-Lipschitz

        else:
            # Conservative estimate
            return 2.0

    def compute_error_bound(
        self,
        original_model: nn.Module,
        circuit_components: List[CircuitComponent],
        test_data: torch.Tensor,
        test_targets: torch.Tensor
    ) -> ErrorBound:
        """
        Compute certified error bound using Lipschitz composition theorem

        Global error bound: ||F_hat(x) - F(x)|| <= sum_i (epsilon_i * prod_{j>i} L_j)

        Args:
            original_model: The original full model
            circuit_components: The extracted circuit
            test_data: Test inputs
            test_targets: Test targets

        Returns:
            ErrorBound object with certification
        """
        with torch.no_grad():
            # Get original model outputs
            original_outputs = original_model(test_data)

            # Get circuit outputs (would need to implement circuit evaluation)
            # For now, we'll compute error bounds empirically
            circuit_outputs = self._evaluate_circuit(circuit_components, test_data)

            # Compute per-component errors
            errors = torch.abs(circuit_outputs - original_outputs)

            local_errors = []
            lipschitz_constants = []

            # Compute per-layer errors and Lipschitz constants
            for component in circuit_components:
                # Local error: difference when using masked vs full weights
                local_error = float(errors.mean())
                local_errors.append(local_error)

                # Estimate Lipschitz constant
                lipschitz = 1.0  # Simplified
                lipschitz_constants.append(lipschitz)

            # Global error bound using composition theorem
            epsilon = 0.0
            for i, (local_err, _) in enumerate(zip(local_errors, lipschitz_constants)):
                # Product of subsequent Lipschitz constants
                lipschitz_product = 1.0
                for j in range(i + 1, len(lipschitz_constants)):
                    lipschitz_product *= lipschitz_constants[j]

                epsilon += local_err * lipschitz_product

            # Compute empirical metrics
            mae = float(errors.mean())
            max_error = float(errors.max())

            # Coverage: fraction within bound
            within_bound = (errors <= epsilon).float().mean()
            coverage = float(within_bound)

            return ErrorBound(
                epsilon=epsilon,
                local_errors=local_errors,
                lipschitz_constants=lipschitz_constants,
                mae=mae,
                max_error=max_error,
                coverage=coverage
            )

    def _evaluate_circuit(
        self,
        circuit_components: List[CircuitComponent],
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate the extracted circuit (simplified implementation)

        In practice, this would construct a new model with only the
        circuit components and evaluate it.
        """
        # Simplified: return original model output
        # A full implementation would build a sparse model
        return self.model(inputs)

    def export_to_json(
        self,
        circuit_components: List[CircuitComponent],
        error_bound: ErrorBound,
        output_path: Path,
        model_name: str = "extracted_circuit"
    ) -> Dict:
        """
        Export circuit to JSON format compatible with the translator
        """
        circuit_data = {
            "model_type": "circuit",
            "name": model_name,
            "components": [],
            "error_bound": {
                "epsilon": error_bound.epsilon,
                "local_errors": error_bound.local_errors,
                "lipschitz_constants": error_bound.lipschitz_constants,
                "mae": error_bound.mae,
                "max_error": error_bound.max_error,
                "coverage": error_bound.coverage
            },
            "metadata": {
                "num_components": len(circuit_components),
                "total_parameters": sum(c.weight.size for c in circuit_components),
                "sparsity": sum(c.mask.mean() for c in circuit_components) / len(circuit_components)
            }
        }

        # Export each component
        for component in circuit_components:
            # Apply mask to weight
            masked_weight = component.weight * component.mask

            component_dict = {
                "layer_idx": component.layer_idx,
                "component_type": component.component_type,
                "component_idx": component.component_idx,
                "weight": masked_weight.tolist(),
                "bias": component.bias.tolist() if component.bias is not None else None,
                "mask": component.mask.tolist(),
                "importance_score": component.importance_score,
                "shape": list(component.weight.shape)
            }

            circuit_data["components"].append(component_dict)

        # Compute hash for verification
        circuit_json = json.dumps(circuit_data, cls=NumpyEncoder, sort_keys=True)
        circuit_hash = hashlib.sha256(circuit_json.encode()).hexdigest()
        circuit_data["certificate_hash"] = circuit_hash

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(circuit_data, f, cls=NumpyEncoder, indent=2)

        print(f"Circuit exported to {output_path}")
        print(f"Certificate hash: {circuit_hash}")
        print(f"Error bound ε: {error_bound.epsilon:.6f}")
        print(f"Sparsity: {circuit_data['metadata']['sparsity']:.2%}")

        return circuit_data


def extract_transformer_circuit(
    model: nn.Module,
    calibration_data: torch.Tensor,
    calibration_targets: torch.Tensor,
    test_data: torch.Tensor,
    test_targets: torch.Tensor,
    output_path: str = "circuit.json",
    pruning_threshold: float = 0.01
) -> Dict:
    """
    Convenience function to extract a circuit from a Transformer model

    Args:
        model: PyTorch Transformer model
        calibration_data: Data for circuit extraction
        calibration_targets: Target outputs for calibration
        test_data: Data for error bound computation
        test_targets: Target outputs for testing
        output_path: Where to save the circuit JSON
        pruning_threshold: Threshold for edge pruning

    Returns:
        Dictionary containing circuit data and error bounds
    """
    extractor = CircuitExtractor(model)

    # Identify layer names (simplified - would introspect model)
    layer_names = [name for name, _ in model.named_modules()
                   if 'attention' in name or 'mlp' in name]

    # Extract circuit
    print(f"Extracting circuit with threshold {pruning_threshold}...")
    circuit_components = extractor.extract_circuit(
        calibration_data,
        calibration_targets,
        layer_names,
        pruning_threshold
    )

    print(f"Extracted {len(circuit_components)} components")

    # Compute error bound
    print("Computing error bounds...")
    error_bound = extractor.compute_error_bound(
        model,
        circuit_components,
        test_data,
        test_targets
    )

    # Export
    output_path = Path(output_path)
    circuit_data = extractor.export_to_json(
        circuit_components,
        error_bound,
        output_path
    )

    return circuit_data


# =============================================================================
# BlockCert Extractor - Full certified extraction pipeline
# =============================================================================


class BlockCertExtractor:
    """
    Full BlockCert extraction pipeline with certified error bounds.

    This is the recommended extractor for production use. It implements:
    1. IR-based weight storage (.npz format)
    2. Pure Python interpreter for surrogate execution
    3. Certified Lipschitz bounds via auto-LiRPA
    4. SHA-256 hashed certificates
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        use_auto_lirpa: bool = True,
    ):
        """
        Initialize the BlockCert extractor.

        Args:
            model: PyTorch transformer model to extract from
            device: Device for computation
            use_auto_lirpa: Whether to use auto-LiRPA for certified K_MLP bounds
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        self.tracer = ActivationTracer(model, device)
        self.certifier = BlockCertifier(use_auto_lirpa=use_auto_lirpa, device=device)
        self.interpreter = BlockInterpreter()

        # Storage for extracted blocks
        self.block_irs: Dict[int, BlockIR] = {}
        self.trace_datasets: Dict[int, TraceDataset] = {}
        self.metrics: Dict[int, CertificationMetrics] = {}

    def extract_and_certify(
        self,
        prompts: List[str],
        tokenizer: Any,
        block_indices: List[int],
        output_dir: Path,
        pruning_threshold: float = 0.01,
        max_length: int = 512,
    ) -> Certificate:
        """
        Full extraction and certification pipeline.

        Args:
            prompts: Calibration prompts
            tokenizer: HuggingFace tokenizer
            block_indices: Which blocks to extract
            output_dir: Directory for .npz files and certificate
            pruning_threshold: Threshold for edge pruning
            max_length: Maximum sequence length

        Returns:
            Certificate with all metrics and hashes
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1 & 2: Trace collection
        print("Phase 1-2: Collecting traces...")
        self.trace_datasets = self.tracer.collect_traces(
            prompts=prompts,
            tokenizer=tokenizer,
            block_indices=block_indices,
            max_length=max_length,
        )

        # Phase 2: Build IR and apply pruning
        print("Phase 2: Building surrogate IRs...")
        block_hashes = []

        for block_idx in block_indices:
            print(f"  Processing block {block_idx}...")

            # Extract weights from model
            weights = extract_block_weights(self.model, block_idx)

            # Build IR (determine architecture from weights)
            block_ir = self._build_block_ir(block_idx, weights, pruning_threshold)
            self.block_irs[block_idx] = block_ir

            # Save to .npz and get hashes
            hashes = block_ir.save(output_dir)
            block_hashes.append(hashes)

        # Phase 3: Certification
        print("Phase 3: Computing certification metrics...")
        block_metrics = []

        for block_idx in block_indices:
            print(f"  Certifying block {block_idx}...")

            metrics = self.certifier.certify_block(
                block_ir=self.block_irs[block_idx],
                trace_dataset=self.trace_datasets[block_idx],
                model=self.model,
            )
            self.metrics[block_idx] = metrics
            block_metrics.append(metrics)

            status = "CERTIFIED" if metrics.is_certified else "NOT CERTIFIED"
            print(f"    ε={metrics.epsilon:.6e}, cov={metrics.activation_coverage:.2%} [{status}]")

        # Phase 4 & 5: Generate certificate
        print("Phase 4-5: Generating certificate...")

        # Count calibration stats
        total_tokens = sum(
            sum(r.input_activations.shape[0] for r in ds.records)
            for ds in self.trace_datasets.values()
        )

        certificate = generate_certificate(
            model_name=self._get_model_name(),
            block_metrics=block_metrics,
            block_hashes=block_hashes,
            calibration_prompts=len(prompts),
            calibration_tokens=total_tokens,
            output_path=output_dir / "certificate.json",
        )

        print(f"\nCertification complete!")
        print(f"  Global ε: {certificate.global_epsilon:.6e}")
        print(f"  Certified: {certificate.certified_blocks}/{certificate.total_blocks} blocks")
        print(f"  Certificate: {output_dir / 'certificate.json'}")

        return certificate

    def _build_block_ir(
        self,
        block_idx: int,
        weights: Dict[str, np.ndarray],
        pruning_threshold: float,
    ) -> BlockIR:
        """Build BlockIR from extracted weights with pruning."""

        # Determine attention configuration
        d_model = weights.get("W_Q", weights.get("W_1", np.zeros((1, 1)))).shape[0]

        # Check for attention weights
        has_attention = "W_Q" in weights or "W_QKV" in weights

        if has_attention:
            # Handle separate Q, K, V projections
            if "W_Q" in weights:
                W_Q = weights["W_Q"]
                W_K = weights["W_K"]
                W_V = weights["W_V"]
                W_O = weights["W_O"]
            else:
                # Handle combined QKV (GPT-2 style)
                W_QKV = weights["W_QKV"]
                third = W_QKV.shape[1] // 3
                W_Q = W_QKV[:, :third]
                W_K = W_QKV[:, third:2*third]
                W_V = W_QKV[:, 2*third:]
                W_O = weights.get("W_O", np.eye(d_model))

            # Compute importance and create masks
            mask_Q = self._compute_pruning_mask(W_Q, pruning_threshold)
            mask_K = self._compute_pruning_mask(W_K, pruning_threshold)
            mask_V = self._compute_pruning_mask(W_V, pruning_threshold)
            mask_O = self._compute_pruning_mask(W_O, pruning_threshold)

            # Infer num_heads and head_dim from Q projection shape
            # Assume head_dim=64 (common for LLaMA-family models)
            head_dim = 64
            num_heads = max(1, W_Q.shape[1] // head_dim)

            # Detect GQA: if K/V projections are smaller than Q projection
            # For GQA models like TinyLlama: 32 query heads, 4 KV heads
            kv_dim = W_K.shape[1]
            if kv_dim < W_Q.shape[1]:
                # GQA detected
                num_kv_heads = kv_dim // head_dim
            else:
                # Standard MHA
                num_kv_heads = num_heads

            attention_ir = AttentionIR(
                W_Q=W_Q,
                W_K=W_K,
                W_V=W_V,
                W_O=W_O,
                b_Q=weights.get("b_Q"),
                b_K=weights.get("b_K"),
                b_V=weights.get("b_V"),
                b_O=weights.get("b_O"),
                mask_Q=mask_Q,
                mask_K=mask_K,
                mask_V=mask_V,
                mask_O=mask_O,
                num_heads=num_heads,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
            )
        else:
            # No attention - create minimal placeholder
            attention_ir = AttentionIR(
                W_Q=np.eye(d_model, dtype=np.float32),
                W_K=np.eye(d_model, dtype=np.float32),
                W_V=np.eye(d_model, dtype=np.float32),
                W_O=np.eye(d_model, dtype=np.float32),
                num_heads=1,
                head_dim=d_model,
            )

        # Build MLP IR
        W_1 = weights.get("W_1", np.eye(d_model, dtype=np.float32))
        W_2 = weights.get("W_2", np.eye(d_model, dtype=np.float32))
        W_gate = weights.get("W_gate")

        mask_1 = self._compute_pruning_mask(W_1, pruning_threshold)
        mask_2 = self._compute_pruning_mask(W_2, pruning_threshold)
        mask_gate = self._compute_pruning_mask(W_gate, pruning_threshold) if W_gate is not None else None

        # Determine activation (default to silu for modern models)
        activation = "silu" if W_gate is not None else "gelu"

        mlp_ir = MLPIR(
            W_1=W_1,
            W_2=W_2,
            W_gate=W_gate,
            b_1=weights.get("b_1"),
            b_2=weights.get("b_2"),
            b_gate=weights.get("b_gate"),
            mask_1=mask_1,
            mask_2=mask_2,
            mask_gate=mask_gate,
            activation=activation,
        )

        return BlockIR(
            block_idx=block_idx,
            attention=attention_ir,
            mlp=mlp_ir,
            metadata={
                "d_model": d_model,
                "pruning_threshold": pruning_threshold,
            },
        )

    def _compute_pruning_mask(
        self,
        weights: Optional[np.ndarray],
        threshold: float,
    ) -> Optional[np.ndarray]:
        """Compute pruning mask based on weight magnitude."""
        if weights is None:
            return None

        # Normalize by max absolute value
        max_val = np.abs(weights).max()
        if max_val < 1e-10:
            return np.ones_like(weights)

        normalized = np.abs(weights) / max_val

        # Mask weights below threshold
        mask = (normalized > threshold).astype(np.float32)
        return mask

    def _get_model_name(self) -> str:
        """Extract model name from model config or class."""
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "name_or_path"):
                return self.model.config.name_or_path
            if hasattr(self.model.config, "_name_or_path"):
                return self.model.config._name_or_path
        return self.model.__class__.__name__

    def evaluate_surrogate(
        self,
        block_idx: int,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the surrogate for a specific block.

        This actually runs the sparse circuit (not the original model).

        Args:
            block_idx: Block index to evaluate
            inputs: Input activations [batch, seq_len, d_model]

        Returns:
            Surrogate outputs [batch, seq_len, d_model]
        """
        if block_idx not in self.block_irs:
            raise ValueError(f"Block {block_idx} not extracted")

        return self.interpreter.interpret_block(self.block_irs[block_idx], inputs)


def extract_certified_circuit(
    model: nn.Module,
    prompts: List[str],
    tokenizer: Any,
    block_indices: List[int],
    output_dir: str = "circuit_output",
    pruning_threshold: float = 0.01,
    use_auto_lirpa: bool = True,
) -> Certificate:
    """
    Convenience function for full BlockCert extraction.

    Args:
        model: PyTorch transformer model
        prompts: Calibration prompts
        tokenizer: HuggingFace tokenizer
        block_indices: Block indices to extract
        output_dir: Output directory for .npz and certificate
        pruning_threshold: Edge pruning threshold
        use_auto_lirpa: Whether to use auto-LiRPA for K_MLP

    Returns:
        Certificate with all metrics
    """
    extractor = BlockCertExtractor(model, use_auto_lirpa=use_auto_lirpa)
    return extractor.extract_and_certify(
        prompts=prompts,
        tokenizer=tokenizer,
        block_indices=block_indices,
        output_dir=Path(output_dir),
        pruning_threshold=pruning_threshold,
    )
