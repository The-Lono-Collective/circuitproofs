"""
Intermediate Representation (IR) for BlockCert

Defines the .npz storage format for transformer block weights.
All weights are stored as explicit tensors with masks for sparsity.

Layout:
- Attention: W_Q, W_K, W_V, W_O (concatenated [d_model, d_model])
- MLP: W_1, W_2 (and W_3 for gated units like SwiGLU)
- Normalization: LayerNorm/RMSNorm weights
- Masks: Causal attention mask, head-level gating masks, weight sparsity masks
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Literal, Optional, Tuple
import numpy as np
import hashlib


def _encode_string(s: str) -> np.ndarray:
    """Encode string as uint8 array to avoid pickle for security."""
    return np.array(list(s.encode("utf-8")), dtype=np.uint8)


def _decode_string(arr: np.ndarray) -> str:
    """Decode uint8 array back to string."""
    return bytes(arr.tolist()).decode("utf-8")


# Maximum block index to prevent resource exhaustion and path issues
MAX_BLOCK_IDX = 10000


def _validate_block_idx(block_idx: int) -> None:
    """
    Validate block_idx is a non-negative integer within bounds.

    Raises:
        TypeError: If block_idx is not an integer.
        ValueError: If block_idx is negative or exceeds MAX_BLOCK_IDX.
    """
    if not isinstance(block_idx, int) or isinstance(block_idx, bool):
        raise TypeError(f"block_idx must be an integer, got {type(block_idx).__name__}")
    if block_idx < 0:
        raise ValueError(f"block_idx must be non-negative, got {block_idx}")
    if block_idx > MAX_BLOCK_IDX:
        raise ValueError(f"block_idx must not exceed {MAX_BLOCK_IDX}, got {block_idx}")


@dataclass
class NormIR:
    """Intermediate representation for normalization layers."""

    norm_type: Literal["layernorm", "rmsnorm"]
    weight: np.ndarray  # Shape: [d_model]
    bias: Optional[np.ndarray] = None  # Shape: [d_model], None for RMSNorm
    eps: float = 1e-5

    def save(self, path: Path) -> str:
        """Save to .npz and return SHA-256 hash."""
        data = {
            "norm_type": _encode_string(self.norm_type),
            "weight": self.weight,
            "eps": np.array([self.eps]),
        }
        if self.bias is not None:
            data["bias"] = self.bias

        np.savez(path, **data)
        return self._compute_hash(path)

    @classmethod
    def load(cls, path: Path) -> "NormIR":
        """Load from .npz file."""
        data = np.load(path, allow_pickle=False)
        return cls(
            norm_type=_decode_string(data["norm_type"]),
            weight=data["weight"],
            bias=data.get("bias"),
            eps=float(data["eps"][0]),
        )

    @staticmethod
    def _compute_hash(path: Path) -> str:
        """Compute SHA-256 hash of the .npz file."""
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()


@dataclass
class AttentionIR:
    """
    Intermediate representation for multi-head attention.

    Supports both Multi-Head Attention (MHA) and Grouped Query Attention (GQA).

    For MHA: num_kv_heads == num_heads
    For GQA: num_kv_heads < num_heads (e.g., TinyLlama uses 32 query heads, 4 KV heads)

    Weights are stored in concatenated format:
    - W_Q, W_O: [d_model, num_heads * head_dim]
    - W_K, W_V: [d_model, num_kv_heads * head_dim]
    """

    # Core weights - concatenated across heads
    W_Q: np.ndarray  # Shape: [d_model, num_heads * head_dim]
    W_K: np.ndarray  # Shape: [d_model, num_kv_heads * head_dim]
    W_V: np.ndarray  # Shape: [d_model, num_kv_heads * head_dim]
    W_O: np.ndarray  # Shape: [d_model, num_heads * head_dim]

    # Biases (optional, many models don't use them)
    b_Q: Optional[np.ndarray] = None  # Shape: [d_model]
    b_K: Optional[np.ndarray] = None  # Shape: [d_model]
    b_V: Optional[np.ndarray] = None  # Shape: [d_model]
    b_O: Optional[np.ndarray] = None  # Shape: [d_model]

    # Masks for sparsity
    mask_Q: Optional[np.ndarray] = None  # Shape: [d_model, d_model], 1=active, 0=pruned
    mask_K: Optional[np.ndarray] = None
    mask_V: Optional[np.ndarray] = None
    mask_O: Optional[np.ndarray] = None

    # Head-level gating mask (which heads are active)
    head_mask: Optional[np.ndarray] = None  # Shape: [num_heads], 1=active, 0=pruned

    # Attention configuration
    num_heads: int = 1
    head_dim: int = 64
    num_kv_heads: Optional[int] = None  # For GQA; defaults to num_heads if not set

    # Pre/post normalization
    pre_norm: Optional[NormIR] = None

    def __post_init__(self):
        """Validate shapes and initialize default masks."""
        d_model = self.W_Q.shape[0]

        # Default num_kv_heads to num_heads for standard MHA
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

        # Initialize masks to all-ones if not provided (no pruning)
        if self.mask_Q is None:
            self.mask_Q = np.ones_like(self.W_Q)
        if self.mask_K is None:
            self.mask_K = np.ones_like(self.W_K)
        if self.mask_V is None:
            self.mask_V = np.ones_like(self.W_V)
        if self.mask_O is None:
            self.mask_O = np.ones_like(self.W_O)
        if self.head_mask is None:
            self.head_mask = np.ones(self.num_heads)

    @property
    def d_model(self) -> int:
        return self.W_Q.shape[0]

    @property
    def is_gqa(self) -> bool:
        """Returns True if using Grouped Query Attention (fewer KV heads than Q heads)."""
        return self.num_kv_heads < self.num_heads

    @property
    def kv_head_repeat(self) -> int:
        """Number of times each KV head is repeated to match query heads."""
        return self.num_heads // self.num_kv_heads

    def get_masked_weights(self) -> Dict[str, np.ndarray]:
        """Return weights with masks applied (pruned weights zeroed)."""
        return {
            "W_Q": self.W_Q * self.mask_Q,
            "W_K": self.W_K * self.mask_K,
            "W_V": self.W_V * self.mask_V,
            "W_O": self.W_O * self.mask_O,
        }

    def compute_sparsity(self) -> float:
        """Compute fraction of non-zero weights after masking."""
        total = sum(m.size for m in [self.mask_Q, self.mask_K, self.mask_V, self.mask_O])
        active = sum(m.sum() for m in [self.mask_Q, self.mask_K, self.mask_V, self.mask_O])
        return float(active / total) if total > 0 else 1.0

    def save(self, path: Path) -> str:
        """Save to .npz and return SHA-256 hash."""
        data = {
            "W_Q": self.W_Q,
            "W_K": self.W_K,
            "W_V": self.W_V,
            "W_O": self.W_O,
            "mask_Q": self.mask_Q,
            "mask_K": self.mask_K,
            "mask_V": self.mask_V,
            "mask_O": self.mask_O,
            "head_mask": self.head_mask,
            "num_heads": np.array([self.num_heads]),
            "head_dim": np.array([self.head_dim]),
            "num_kv_heads": np.array([self.num_kv_heads]),
        }

        # Add optional biases
        for name, bias in [("b_Q", self.b_Q), ("b_K", self.b_K),
                           ("b_V", self.b_V), ("b_O", self.b_O)]:
            if bias is not None:
                data[name] = bias

        np.savez(path, **data)
        return self._compute_hash(path)

    @classmethod
    def load(cls, path: Path) -> "AttentionIR":
        """Load from .npz file."""
        data = np.load(path, allow_pickle=False)
        return cls(
            W_Q=data["W_Q"],
            W_K=data["W_K"],
            W_V=data["W_V"],
            W_O=data["W_O"],
            b_Q=data.get("b_Q"),
            b_K=data.get("b_K"),
            b_V=data.get("b_V"),
            b_O=data.get("b_O"),
            mask_Q=data["mask_Q"],
            mask_K=data["mask_K"],
            mask_V=data["mask_V"],
            mask_O=data["mask_O"],
            head_mask=data["head_mask"],
            num_heads=int(data["num_heads"][0]),
            head_dim=int(data["head_dim"][0]),
            num_kv_heads=int(data["num_kv_heads"][0]) if "num_kv_heads" in data else None,
        )

    @staticmethod
    def _compute_hash(path: Path) -> str:
        """Compute SHA-256 hash of the .npz file."""
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()


@dataclass
class MLPIR:
    """
    Intermediate representation for MLP/FFN blocks.

    Supports both standard (W_1, W_2) and gated (W_1, W_gate, W_2) architectures.
    """

    # Core weights
    W_1: np.ndarray  # Up projection: [d_model, d_ff]
    W_2: np.ndarray  # Down projection: [d_ff, d_model]

    # Gated architectures (SwiGLU, etc.)
    W_gate: Optional[np.ndarray] = None  # Gate projection: [d_model, d_ff]

    # Biases (optional)
    b_1: Optional[np.ndarray] = None  # Shape: [d_ff]
    b_2: Optional[np.ndarray] = None  # Shape: [d_model]
    b_gate: Optional[np.ndarray] = None  # Shape: [d_ff]

    # Sparsity masks
    mask_1: Optional[np.ndarray] = None  # Shape: [d_model, d_ff]
    mask_2: Optional[np.ndarray] = None  # Shape: [d_ff, d_model]
    mask_gate: Optional[np.ndarray] = None  # Shape: [d_model, d_ff]

    # Activation function
    activation: Literal["relu", "gelu", "silu", "swiglu"] = "gelu"

    # Pre/post normalization
    pre_norm: Optional[NormIR] = None

    def __post_init__(self):
        """Initialize default masks."""
        if self.mask_1 is None:
            self.mask_1 = np.ones_like(self.W_1)
        if self.mask_2 is None:
            self.mask_2 = np.ones_like(self.W_2)
        if self.W_gate is not None and self.mask_gate is None:
            self.mask_gate = np.ones_like(self.W_gate)

    @property
    def d_model(self) -> int:
        return self.W_1.shape[0]

    @property
    def d_ff(self) -> int:
        return self.W_1.shape[1]

    @property
    def is_gated(self) -> bool:
        return self.W_gate is not None

    def get_masked_weights(self) -> Dict[str, np.ndarray]:
        """Return weights with masks applied."""
        result = {
            "W_1": self.W_1 * self.mask_1,
            "W_2": self.W_2 * self.mask_2,
        }
        if self.W_gate is not None:
            result["W_gate"] = self.W_gate * self.mask_gate
        return result

    def compute_sparsity(self) -> float:
        """Compute fraction of non-zero weights after masking."""
        masks = [self.mask_1, self.mask_2]
        if self.mask_gate is not None:
            masks.append(self.mask_gate)

        total = sum(m.size for m in masks)
        active = sum(m.sum() for m in masks)
        return float(active / total) if total > 0 else 1.0

    def save(self, path: Path) -> str:
        """Save to .npz and return SHA-256 hash."""
        data = {
            "W_1": self.W_1,
            "W_2": self.W_2,
            "mask_1": self.mask_1,
            "mask_2": self.mask_2,
            "activation": _encode_string(self.activation),
        }

        # Add optional components
        if self.W_gate is not None:
            data["W_gate"] = self.W_gate
            data["mask_gate"] = self.mask_gate

        for name, bias in [("b_1", self.b_1), ("b_2", self.b_2), ("b_gate", self.b_gate)]:
            if bias is not None:
                data[name] = bias

        np.savez(path, **data)
        return self._compute_hash(path)

    @classmethod
    def load(cls, path: Path) -> "MLPIR":
        """Load from .npz file."""
        data = np.load(path, allow_pickle=False)
        return cls(
            W_1=data["W_1"],
            W_2=data["W_2"],
            W_gate=data.get("W_gate"),
            b_1=data.get("b_1"),
            b_2=data.get("b_2"),
            b_gate=data.get("b_gate"),
            mask_1=data["mask_1"],
            mask_2=data["mask_2"],
            mask_gate=data.get("mask_gate"),
            activation=_decode_string(data["activation"]),
        )

    @staticmethod
    def _compute_hash(path: Path) -> str:
        """Compute SHA-256 hash of the .npz file."""
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()


@dataclass
class BlockIR:
    """
    Complete intermediate representation for a transformer block.

    Combines attention, MLP, and normalization components.
    """

    block_idx: int
    attention: AttentionIR
    mlp: MLPIR

    # Post-attention and post-MLP layer norms (for post-norm architecture)
    post_attn_norm: Optional[NormIR] = None
    post_mlp_norm: Optional[NormIR] = None

    # Causal attention mask (shared across the block)
    causal_mask: Optional[np.ndarray] = None  # Shape: [seq_len, seq_len]

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate block_idx to prevent directory traversal attacks."""
        _validate_block_idx(self.block_idx)

    def compute_sparsity(self) -> Dict[str, float]:
        """Compute sparsity for each component."""
        return {
            "attention": self.attention.compute_sparsity(),
            "mlp": self.mlp.compute_sparsity(),
            "total": (self.attention.compute_sparsity() + self.mlp.compute_sparsity()) / 2,
        }

    def save(self, output_dir: Path) -> Dict[str, str]:
        """
        Save all components to .npz files and return SHA-256 hashes.

        Returns dict mapping component names to their hashes.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        hashes = {}

        # Save attention
        attn_path = output_dir / f"block_{self.block_idx}_attention.npz"
        hashes["attention"] = self.attention.save(attn_path)

        # Save MLP
        mlp_path = output_dir / f"block_{self.block_idx}_mlp.npz"
        hashes["mlp"] = self.mlp.save(mlp_path)

        # Save norms if present
        if self.attention.pre_norm is not None:
            norm_path = output_dir / f"block_{self.block_idx}_pre_attn_norm.npz"
            hashes["pre_attn_norm"] = self.attention.pre_norm.save(norm_path)

        if self.mlp.pre_norm is not None:
            norm_path = output_dir / f"block_{self.block_idx}_pre_mlp_norm.npz"
            hashes["pre_mlp_norm"] = self.mlp.pre_norm.save(norm_path)

        if self.post_attn_norm is not None:
            norm_path = output_dir / f"block_{self.block_idx}_post_attn_norm.npz"
            hashes["post_attn_norm"] = self.post_attn_norm.save(norm_path)

        if self.post_mlp_norm is not None:
            norm_path = output_dir / f"block_{self.block_idx}_post_mlp_norm.npz"
            hashes["post_mlp_norm"] = self.post_mlp_norm.save(norm_path)

        # Save causal mask if present
        if self.causal_mask is not None:
            mask_path = output_dir / f"block_{self.block_idx}_causal_mask.npz"
            np.savez(mask_path, causal_mask=self.causal_mask)
            with open(mask_path, "rb") as f:
                hashes["causal_mask"] = hashlib.sha256(f.read()).hexdigest()

        # Save metadata
        meta_path = output_dir / f"block_{self.block_idx}_metadata.npz"
        np.savez(meta_path,
                 block_idx=np.array([self.block_idx]),
                 **{k: np.array([v]) if isinstance(v, (int, float)) else v
                    for k, v in self.metadata.items()})

        return hashes

    @classmethod
    def load(cls, output_dir: Path, block_idx: int) -> "BlockIR":
        """Load all components from .npz files."""
        _validate_block_idx(block_idx)
        output_dir = Path(output_dir)

        # Load attention
        attn_path = output_dir / f"block_{block_idx}_attention.npz"
        attention = AttentionIR.load(attn_path)

        # Load MLP
        mlp_path = output_dir / f"block_{block_idx}_mlp.npz"
        mlp = MLPIR.load(mlp_path)

        # Load optional norms
        pre_attn_norm_path = output_dir / f"block_{block_idx}_pre_attn_norm.npz"
        if pre_attn_norm_path.exists():
            attention.pre_norm = NormIR.load(pre_attn_norm_path)

        pre_mlp_norm_path = output_dir / f"block_{block_idx}_pre_mlp_norm.npz"
        if pre_mlp_norm_path.exists():
            mlp.pre_norm = NormIR.load(pre_mlp_norm_path)

        post_attn_norm = None
        post_attn_norm_path = output_dir / f"block_{block_idx}_post_attn_norm.npz"
        if post_attn_norm_path.exists():
            post_attn_norm = NormIR.load(post_attn_norm_path)

        post_mlp_norm = None
        post_mlp_norm_path = output_dir / f"block_{block_idx}_post_mlp_norm.npz"
        if post_mlp_norm_path.exists():
            post_mlp_norm = NormIR.load(post_mlp_norm_path)

        # Load causal mask
        causal_mask = None
        mask_path = output_dir / f"block_{block_idx}_causal_mask.npz"
        if mask_path.exists():
            causal_mask = np.load(mask_path, allow_pickle=False)["causal_mask"]

        # Load metadata (only numeric values supported for security)
        metadata = {}
        meta_path = output_dir / f"block_{block_idx}_metadata.npz"
        if meta_path.exists():
            meta_data = np.load(meta_path, allow_pickle=False)
            metadata = {k: v.item() if v.size == 1 else v
                       for k, v in meta_data.items() if k != "block_idx"}

        return cls(
            block_idx=block_idx,
            attention=attention,
            mlp=mlp,
            post_attn_norm=post_attn_norm,
            post_mlp_norm=post_mlp_norm,
            causal_mask=causal_mask,
            metadata=metadata,
        )


@dataclass
class TraceRecord:
    """Single trace record for a block."""

    block_idx: int
    input_activations: np.ndarray  # [seq_len, d_model]
    output_activations: np.ndarray  # [seq_len, d_model]
    attention_mask: Optional[np.ndarray] = None  # [seq_len, seq_len]
    attention_weights: Optional[np.ndarray] = None  # [num_heads, seq_len, seq_len]
    prompt_id: Optional[str] = None

    def __post_init__(self):
        """Validate block_idx."""
        _validate_block_idx(self.block_idx)


@dataclass
class TraceDataset:
    """
    Dataset of trace records for certification.

    D_l = {(x^(l), x^(l+1), m_l)} for all prompts
    """

    block_idx: int
    records: List[TraceRecord] = field(default_factory=list)

    def __post_init__(self):
        """Validate block_idx."""
        _validate_block_idx(self.block_idx)

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
                data[f"prompt_id_{i}"] = _encode_string(record.prompt_id)

        np.savez(path, **data)

    @classmethod
    def load(cls, path: Path) -> "TraceDataset":
        """Load dataset from .npz file."""
        data = np.load(path, allow_pickle=False)
        block_idx = int(data["block_idx"])
        num_records = int(data["num_records"])

        dataset = cls(block_idx=block_idx)

        for i in range(num_records):
            prompt_id_arr = data.get(f"prompt_id_{i}")
            prompt_id = _decode_string(prompt_id_arr) if prompt_id_arr is not None else None
            record = TraceRecord(
                block_idx=block_idx,
                input_activations=data[f"input_{i}"],
                output_activations=data[f"output_{i}"],
                attention_mask=data.get(f"attn_mask_{i}"),
                attention_weights=data.get(f"attn_weights_{i}"),
                prompt_id=prompt_id,
            )
            dataset.add_record(record)

        return dataset
