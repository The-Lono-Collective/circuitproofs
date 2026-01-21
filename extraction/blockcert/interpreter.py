"""
Pure Python Interpreter for BlockCert IR

Executes transformer block logic using only linear algebra operations.
No hidden control flow - behavior is determined entirely by stored weights and masks.

This interpreter is designed to:
1. Match PyTorch numerical behavior (float32)
2. Support sparse computation (skip zero weights)
3. Be verifiable in Lean (deterministic, explicit operations)
"""

import numpy as np
from typing import Optional, Tuple
from .ir import BlockIR, AttentionIR, MLPIR, NormIR


class BlockInterpreter:
    """
    Pure Python interpreter for transformer blocks.

    Executes the block using explicit linear algebra without framework dependencies.
    """

    def __init__(self, use_sparse: bool = False):
        """
        Initialize the interpreter.

        Args:
            use_sparse: If True, skip zero-weight computations for efficiency.
                       Note: Result is mathematically equivalent, just faster.
        """
        self.use_sparse = use_sparse

    def interpret_block(
        self,
        block_ir: BlockIR,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        position_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Execute a full transformer block.

        Args:
            block_ir: The block's intermediate representation
            hidden_states: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask [batch, 1, seq_len, seq_len]
            position_ids: Optional position IDs for rotary embeddings

        Returns:
            Output hidden states [batch, seq_len, d_model]
        """
        residual = hidden_states

        # Pre-attention normalization (if present)
        if block_ir.attention.pre_norm is not None:
            hidden_states = self._apply_norm(block_ir.attention.pre_norm, hidden_states)

        # Self-attention
        attn_output = self._interpret_attention(
            block_ir.attention,
            hidden_states,
            attention_mask,
            block_ir.causal_mask,
        )

        # Residual connection
        hidden_states = residual + attn_output

        # Post-attention normalization (if present)
        if block_ir.post_attn_norm is not None:
            hidden_states = self._apply_norm(block_ir.post_attn_norm, hidden_states)

        residual = hidden_states

        # Pre-MLP normalization (if present)
        if block_ir.mlp.pre_norm is not None:
            hidden_states = self._apply_norm(block_ir.mlp.pre_norm, hidden_states)

        # MLP
        mlp_output = self._interpret_mlp(block_ir.mlp, hidden_states)

        # Residual connection
        hidden_states = residual + mlp_output

        # Post-MLP normalization (if present)
        if block_ir.post_mlp_norm is not None:
            hidden_states = self._apply_norm(block_ir.post_mlp_norm, hidden_states)

        return hidden_states

    def _interpret_attention(
        self,
        attn_ir: AttentionIR,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray],
        causal_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Execute multi-head attention.

        Args:
            attn_ir: Attention IR with weights and masks
            hidden_states: Input [batch, seq_len, d_model]
            attention_mask: Padding mask [batch, 1, seq_len, seq_len]
            causal_mask: Causal mask [seq_len, seq_len]

        Returns:
            Attention output [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = hidden_states.shape
        num_heads = attn_ir.num_heads
        head_dim = attn_ir.head_dim

        # Get masked weights
        weights = attn_ir.get_masked_weights()
        W_Q, W_K, W_V, W_O = weights["W_Q"], weights["W_K"], weights["W_V"], weights["W_O"]

        # Project to Q, K, V
        # Shape: [batch, seq_len, d_model] @ [d_model, d_model] -> [batch, seq_len, d_model]
        Q = self._matmul(hidden_states, W_Q.T, attn_ir.b_Q)
        K = self._matmul(hidden_states, W_K.T, attn_ir.b_K)
        V = self._matmul(hidden_states, W_V.T, attn_ir.b_V)

        # Reshape for multi-head attention
        # [batch, seq_len, d_model] -> [batch, seq_len, num_heads, head_dim]
        # -> [batch, num_heads, seq_len, head_dim]
        Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        # Apply head mask (zero out pruned heads)
        head_mask = attn_ir.head_mask.reshape(1, num_heads, 1, 1)
        Q = Q * head_mask
        K = K * head_mask
        V = V * head_mask

        # Compute attention scores
        # [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
        # -> [batch, num_heads, seq_len, seq_len]
        scale = 1.0 / np.sqrt(head_dim)
        attn_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale

        # Apply causal mask
        if causal_mask is not None:
            # causal_mask: [seq_len, seq_len], True where allowed
            # Convert to additive mask: 0 where allowed, -inf where masked
            causal_additive = np.where(causal_mask, 0.0, -np.inf)
            attn_scores = attn_scores + causal_additive

        # Apply attention mask (padding)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax
        attn_probs = self._softmax(attn_scores, axis=-1)

        # Apply attention to values
        # [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]
        # -> [batch, num_heads, seq_len, head_dim]
        attn_output = np.matmul(attn_probs, V)

        # Reshape back
        # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        # -> [batch, seq_len, d_model]
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

        # Output projection
        output = self._matmul(attn_output, W_O.T, attn_ir.b_O)

        return output

    def _interpret_mlp(
        self,
        mlp_ir: MLPIR,
        hidden_states: np.ndarray,
    ) -> np.ndarray:
        """
        Execute MLP/FFN block.

        Supports both standard (W_1 -> act -> W_2) and gated (SwiGLU) architectures.

        Args:
            mlp_ir: MLP IR with weights and masks
            hidden_states: Input [batch, seq_len, d_model]

        Returns:
            MLP output [batch, seq_len, d_model]
        """
        # Get masked weights
        weights = mlp_ir.get_masked_weights()
        W_1, W_2 = weights["W_1"], weights["W_2"]

        if mlp_ir.is_gated:
            # Gated architecture (SwiGLU, etc.)
            W_gate = weights["W_gate"]

            # Up projection
            up = self._matmul(hidden_states, W_1.T, mlp_ir.b_1)

            # Gate projection
            gate = self._matmul(hidden_states, W_gate.T, mlp_ir.b_gate)

            # Apply activation to gate
            gate = self._activation(gate, mlp_ir.activation)

            # Element-wise gating
            hidden = up * gate
        else:
            # Standard architecture
            hidden = self._matmul(hidden_states, W_1.T, mlp_ir.b_1)
            hidden = self._activation(hidden, mlp_ir.activation)

        # Down projection
        output = self._matmul(hidden, W_2.T, mlp_ir.b_2)

        return output

    def _apply_norm(
        self,
        norm_ir: NormIR,
        hidden_states: np.ndarray,
    ) -> np.ndarray:
        """
        Apply layer normalization or RMS normalization.

        Args:
            norm_ir: Normalization IR
            hidden_states: Input [batch, seq_len, d_model]

        Returns:
            Normalized output [batch, seq_len, d_model]
        """
        if norm_ir.norm_type == "layernorm":
            # LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
            mean = hidden_states.mean(axis=-1, keepdims=True)
            var = hidden_states.var(axis=-1, keepdims=True)
            normalized = (hidden_states - mean) / np.sqrt(var + norm_ir.eps)
            output = normalized * norm_ir.weight
            if norm_ir.bias is not None:
                output = output + norm_ir.bias
            return output

        elif norm_ir.norm_type == "rmsnorm":
            # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
            rms = np.sqrt(np.mean(hidden_states ** 2, axis=-1, keepdims=True) + norm_ir.eps)
            normalized = hidden_states / rms
            return normalized * norm_ir.weight

        else:
            raise ValueError(f"Unknown norm type: {norm_ir.norm_type}")

    def _matmul(
        self,
        x: np.ndarray,
        W: np.ndarray,
        bias: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Matrix multiplication with optional bias.

        If use_sparse=True, uses sparse operations for efficiency.
        """
        if self.use_sparse:
            # For sparse computation, we could use scipy.sparse
            # For now, dense computation with masking is used
            # (zeros in W naturally result in zero contributions)
            pass

        result = np.matmul(x, W)
        if bias is not None:
            result = result + bias
        return result

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Numerically stable softmax.

        Explicit implementation matching PyTorch behavior.
        """
        # Subtract max for numerical stability
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)

        # Handle -inf values (from masking)
        exp_x = np.where(np.isinf(x) & (x < 0), 0.0, exp_x)

        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _activation(self, x: np.ndarray, activation: str) -> np.ndarray:
        """
        Apply activation function.

        Explicit implementations for each supported activation.
        """
        if activation == "relu":
            return np.maximum(0, x)

        elif activation == "gelu":
            # GELU approximation matching PyTorch
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

        elif activation == "silu" or activation == "swiglu":
            # SiLU (Swish): x * sigmoid(x)
            return x * (1 / (1 + np.exp(-x)))

        else:
            raise ValueError(f"Unknown activation: {activation}")


def interpret_attention_only(
    attn_ir: AttentionIR,
    hidden_states: np.ndarray,
    attention_mask: Optional[np.ndarray] = None,
    causal_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convenience function to interpret just an attention layer.

    Useful for testing and certification.
    """
    interpreter = BlockInterpreter()
    return interpreter._interpret_attention(attn_ir, hidden_states, attention_mask, causal_mask)


def interpret_mlp_only(
    mlp_ir: MLPIR,
    hidden_states: np.ndarray,
) -> np.ndarray:
    """
    Convenience function to interpret just an MLP layer.

    Useful for testing and certification.
    """
    interpreter = BlockInterpreter()
    return interpreter._interpret_mlp(mlp_ir, hidden_states)
