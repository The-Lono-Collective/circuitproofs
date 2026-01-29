"""
Activation Tracer for BlockCert

Hook-based trace collection from transformer models.
Records input activations, output activations, and attention masks
for each target block.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
import numpy as np
from pathlib import Path


@dataclass
class TraceRecord:
    """Single trace record for a block."""

    block_idx: int
    input_activations: np.ndarray  # [seq_len, d_model]
    output_activations: np.ndarray  # [seq_len, d_model]
    attention_mask: Optional[np.ndarray] = None  # [seq_len, seq_len]
    attention_weights: Optional[np.ndarray] = None  # [num_heads, seq_len, seq_len]
    prompt_id: Optional[str] = None


@dataclass
class TraceDataset:
    """
    Dataset of trace records for certification.

    D_l = {(x^(l), x^(l+1), m_l)} for all prompts
    """

    block_idx: int
    records: List[TraceRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> TraceRecord:
        return self.records[idx]

    def add_record(self, record: TraceRecord) -> None:
        """Add a trace record."""
        assert record.block_idx == self.block_idx
        self.records.append(record)

    def get_inputs(self) -> np.ndarray:
        """Get all input activations stacked. Shape: [num_records, seq_len, d_model]"""
        return np.stack([r.input_activations for r in self.records])

    def get_outputs(self) -> np.ndarray:
        """Get all output activations stacked. Shape: [num_records, seq_len, d_model]"""
        return np.stack([r.output_activations for r in self.records])

    def get_flat_tokens(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all tokens flattened for per-token analysis.

        Returns:
            inputs: [total_tokens, d_model]
            outputs: [total_tokens, d_model]
        """
        inputs = []
        outputs = []
        for record in self.records:
            # Flatten sequence dimension
            inputs.append(record.input_activations.reshape(-1, record.input_activations.shape[-1]))
            outputs.append(record.output_activations.reshape(-1, record.output_activations.shape[-1]))

        return np.concatenate(inputs, axis=0), np.concatenate(outputs, axis=0)

    def save(self, path: Path) -> None:
        """Save dataset to .npz file."""
        data = {
            "block_idx": self.block_idx,
            "num_records": len(self.records),
        }

        for i, record in enumerate(self.records):
            data[f"input_{i}"] = record.input_activations
            data[f"output_{i}"] = record.output_activations
            if record.attention_mask is not None:
                data[f"attn_mask_{i}"] = record.attention_mask
            if record.attention_weights is not None:
                data[f"attn_weights_{i}"] = record.attention_weights
            if record.prompt_id is not None:
                data[f"prompt_id_{i}"] = record.prompt_id

        np.savez(path, **data)

    @classmethod
    def load(cls, path: Path) -> "TraceDataset":
        """Load dataset from .npz file."""
        data = np.load(path, allow_pickle=True)
        block_idx = int(data["block_idx"])
        num_records = int(data["num_records"])

        dataset = cls(block_idx=block_idx)

        for i in range(num_records):
            record = TraceRecord(
                block_idx=block_idx,
                input_activations=data[f"input_{i}"],
                output_activations=data[f"output_{i}"],
                attention_mask=data.get(f"attn_mask_{i}"),
                attention_weights=data.get(f"attn_weights_{i}"),
                prompt_id=data.get(f"prompt_id_{i}"),
            )
            dataset.add_record(record)

        return dataset


class ActivationTracer:
    """
    Hook-based activation tracer for transformer models.

    Captures input/output activations at specified blocks during forward pass.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Initialize the tracer.

        Args:
            model: PyTorch transformer model
            device: Device to run tracing on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        # Storage for captured activations
        self._input_activations: Dict[int, torch.Tensor] = {}
        self._output_activations: Dict[int, torch.Tensor] = {}
        self._attention_weights: Dict[int, torch.Tensor] = {}
        self._hooks: List[Any] = []

    def _get_block_modules(self, block_indices: List[int]) -> Dict[int, nn.Module]:
        """
        Find the transformer block modules by index.

        Supports common model architectures (GPT, Llama, etc.)
        """
        blocks = {}

        # Try common naming conventions
        for name, module in self.model.named_modules():
            # HuggingFace transformers patterns
            patterns = [
                f"model.layers.{{}}.self_attn",  # Llama, Mistral
                f"transformer.h.{{}}.attn",  # GPT-2
                f"gpt_neox.layers.{{}}.attention",  # GPT-NeoX
                f"model.decoder.layers.{{}}.self_attn",  # Decoder models
            ]

            for idx in block_indices:
                for pattern in patterns:
                    if name == pattern.format(idx) or name.endswith(f".{idx}"):
                        # Found a block - get its parent (the full block)
                        parent_name = ".".join(name.split(".")[:-1])
                        for pname, pmodule in self.model.named_modules():
                            if pname == parent_name:
                                blocks[idx] = pmodule
                                break

        # Fallback: try to find numbered modules
        if not blocks:
            for name, module in self.model.named_modules():
                for idx in block_indices:
                    if f".{idx}." in name or name.endswith(f".{idx}"):
                        # Check if this is a decoder/transformer layer
                        if hasattr(module, "self_attn") or hasattr(module, "attention"):
                            if idx not in blocks:
                                blocks[idx] = module

        return blocks

    def register_hooks(
        self,
        block_indices: List[int],
        capture_attention_weights: bool = False,
    ) -> None:
        """
        Register forward hooks on specified blocks.

        Args:
            block_indices: List of block indices to trace
            capture_attention_weights: Whether to capture attention weight matrices
        """
        self.remove_hooks()  # Clear any existing hooks

        blocks = self._get_block_modules(block_indices)

        if not blocks:
            raise ValueError(
                f"Could not find blocks at indices {block_indices}. "
                "Model architecture may not be supported."
            )

        for block_idx, block_module in blocks.items():
            # Input hook
            def make_input_hook(idx: int):
                def hook(module, args, kwargs):
                    # Handle both positional and keyword arguments
                    if args:
                        hidden_states = args[0]
                    elif "hidden_states" in kwargs:
                        hidden_states = kwargs["hidden_states"]
                    else:
                        return
                    self._input_activations[idx] = hidden_states.detach().clone()
                return hook

            # Output hook
            def make_output_hook(idx: int):
                def hook(module, args, kwargs, output):
                    # Output can be a tuple (hidden_states, ...) or just hidden_states
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                    self._output_activations[idx] = hidden_states.detach().clone()
                return hook

            # Register hooks
            handle_in = block_module.register_forward_pre_hook(
                make_input_hook(block_idx), with_kwargs=True
            )
            handle_out = block_module.register_forward_hook(
                make_output_hook(block_idx), with_kwargs=True
            )
            self._hooks.extend([handle_in, handle_out])

            # Optionally capture attention weights
            if capture_attention_weights:
                if hasattr(block_module, "self_attn"):
                    attn_module = block_module.self_attn
                elif hasattr(block_module, "attention"):
                    attn_module = block_module.attention
                else:
                    continue

                def make_attn_hook(idx: int):
                    def hook(module, args, kwargs, output):
                        if isinstance(output, tuple) and len(output) > 1:
                            # Second element is often attention weights
                            attn_weights = output[1]
                            if attn_weights is not None:
                                self._attention_weights[idx] = attn_weights.detach().clone()
                    return hook

                handle_attn = attn_module.register_forward_hook(
                    make_attn_hook(block_idx), with_kwargs=True
                )
                self._hooks.append(handle_attn)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._input_activations.clear()
        self._output_activations.clear()
        self._attention_weights.clear()

    def trace(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[int, TraceRecord]:
        """
        Run a forward pass and capture activations.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            **kwargs: Additional arguments for the model

        Returns:
            Dict mapping block_idx to TraceRecord
        """
        # Clear previous activations
        self._input_activations.clear()
        self._output_activations.clear()
        self._attention_weights.clear()

        # Move inputs to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Forward pass
        with torch.no_grad():
            _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )

        # Build trace records
        records = {}
        for block_idx in self._input_activations.keys():
            input_act = self._input_activations[block_idx].cpu().numpy()
            output_act = self._output_activations.get(block_idx)

            if output_act is None:
                continue

            output_act = output_act.cpu().numpy()

            # Squeeze batch dimension if batch_size=1
            if input_act.shape[0] == 1:
                input_act = input_act[0]
                output_act = output_act[0]

            attn_weights = self._attention_weights.get(block_idx)
            if attn_weights is not None:
                attn_weights = attn_weights.cpu().numpy()
                if attn_weights.shape[0] == 1:
                    attn_weights = attn_weights[0]

            records[block_idx] = TraceRecord(
                block_idx=block_idx,
                input_activations=input_act,
                output_activations=output_act,
                attention_weights=attn_weights,
            )

        return records

    def collect_traces(
        self,
        prompts: List[str],
        tokenizer: Any,
        block_indices: List[int],
        max_length: int = 512,
        capture_attention_weights: bool = False,
    ) -> Dict[int, TraceDataset]:
        """
        Collect traces for a set of prompts.

        Args:
            prompts: List of prompt strings
            tokenizer: HuggingFace tokenizer
            block_indices: Block indices to trace
            max_length: Maximum sequence length
            capture_attention_weights: Whether to capture attention weights

        Returns:
            Dict mapping block_idx to TraceDataset
        """
        # Register hooks
        self.register_hooks(block_indices, capture_attention_weights)

        # Initialize datasets
        datasets = {idx: TraceDataset(block_idx=idx) for idx in block_indices}

        try:
            for prompt_idx, prompt in enumerate(prompts):
                # Tokenize
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )

                # Trace
                records = self.trace(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )

                # Add to datasets
                for block_idx, record in records.items():
                    record.prompt_id = f"prompt_{prompt_idx}"
                    datasets[block_idx].add_record(record)

        finally:
            self.remove_hooks()

        return datasets


def extract_block_weights(
    model: nn.Module,
    block_idx: int,
) -> Dict[str, np.ndarray]:
    """
    Extract weight tensors from a transformer block.

    Returns dict with keys like 'W_Q', 'W_K', 'W_V', 'W_O', 'W_1', 'W_2', etc.
    """
    weights = {}

    # Find the block
    block = None
    for name, module in model.named_modules():
        if f".{block_idx}." in name or name.endswith(f".{block_idx}"):
            if hasattr(module, "self_attn") or hasattr(module, "attention"):
                block = module
                break

    if block is None:
        raise ValueError(f"Could not find block at index {block_idx}")

    # Extract attention weights
    attn = getattr(block, "self_attn", None) or getattr(block, "attention", None)
    if attn is not None:
        # Try common weight names
        weight_mappings = [
            # Llama-style
            ("q_proj.weight", "W_Q"),
            ("k_proj.weight", "W_K"),
            ("v_proj.weight", "W_V"),
            ("o_proj.weight", "W_O"),
            # GPT-2 style
            ("c_attn.weight", "W_QKV"),  # Combined
            ("c_proj.weight", "W_O"),
            # Bias
            ("q_proj.bias", "b_Q"),
            ("k_proj.bias", "b_K"),
            ("v_proj.bias", "b_V"),
            ("o_proj.bias", "b_O"),
        ]

        for param_name, weight_name in weight_mappings:
            try:
                param = attn
                for part in param_name.split("."):
                    param = getattr(param, part, None)
                    if param is None:
                        break
                if param is not None:
                    weights[weight_name] = param.detach().cpu().numpy()
            except AttributeError:
                continue

    # Extract MLP weights
    mlp = getattr(block, "mlp", None) or getattr(block, "feed_forward", None)
    if mlp is not None:
        mlp_mappings = [
            # Llama-style (gated)
            ("gate_proj.weight", "W_gate"),
            ("up_proj.weight", "W_1"),
            ("down_proj.weight", "W_2"),
            # GPT-2 style
            ("c_fc.weight", "W_1"),
            ("c_proj.weight", "W_2"),
            # Bias
            ("gate_proj.bias", "b_gate"),
            ("up_proj.bias", "b_1"),
            ("down_proj.bias", "b_2"),
        ]

        for param_name, weight_name in mlp_mappings:
            try:
                param = mlp
                for part in param_name.split("."):
                    param = getattr(param, part, None)
                    if param is None:
                        break
                if param is not None:
                    weights[weight_name] = param.detach().cpu().numpy()
            except AttributeError:
                continue

    # Extract normalization weights
    for norm_name in ["input_layernorm", "post_attention_layernorm", "ln_1", "ln_2"]:
        norm = getattr(block, norm_name, None)
        if norm is not None:
            if hasattr(norm, "weight"):
                weights[f"{norm_name}_weight"] = norm.weight.detach().cpu().numpy()
            if hasattr(norm, "bias") and norm.bias is not None:
                weights[f"{norm_name}_bias"] = norm.bias.detach().cpu().numpy()

    return weights
