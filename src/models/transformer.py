"""
Educational implementation of transformer components using JAX and Flax.
This module serves as a clear example of how to implement transformer architecture
components with detailed explanations of each part.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism implementation.
    
    This is a simplified but educational implementation of the attention mechanism
    used in transformer models. Each step is documented for clarity.
    
    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        dropout_rate: Dropout probability (0.0 means no dropout)
    """
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, mask: Optional[jnp.ndarray] = None, 
                 deterministic: bool = True) -> jnp.ndarray:
        """Applies multi-head attention on the input data.
        
        Args:
            inputs: Input of shape `[batch_size, seq_len, hidden_dim]`
            mask: Optional mask of shape `[batch_size, seq_len, seq_len]`
            deterministic: Whether to apply dropout
            
        Returns:
            Output of shape `[batch_size, seq_len, hidden_dim]`
        """
        batch_size, seq_len, hidden_dim = inputs.shape
        
        # Create query, key, and value projections
        qkv_proj = nn.Dense(3 * self.num_heads * self.head_dim, 
                           name='qkv_projection')
        qkv = qkv_proj(inputs)
        
        # Reshape for multi-head attention
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv
        
        # Scaled dot-product attention
        scale = jnp.sqrt(self.head_dim).astype(k.dtype)
        attention = (q @ jnp.transpose(k, (0, 1, 3, 2))) / scale
        
        if mask is not None:
            attention = jnp.where(mask[:, None, :, :], attention, -1e9)
        
        attention = jax.nn.softmax(attention)
        
        if not deterministic:
            attention = nn.Dropout(rate=self.dropout_rate)(
                attention, deterministic=deterministic)
        
        # Combine heads and project output
        output = attention @ v
        output = jnp.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch_size, seq_len, hidden_dim)
        
        return output


class TransformerBlock(nn.Module):
    """A basic transformer block implementation.
    
    This block combines multi-head attention with feed-forward layers,
    layer normalization, and residual connections.
    
    Attributes:
        hidden_dim: Dimension of the hidden layer
        num_heads: Number of attention heads
        mlp_dim: Dimension of the feed-forward layer
        dropout_rate: Dropout probability
    """
    hidden_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, 
                 deterministic: bool = True) -> jnp.ndarray:
        """Applies the transformer block to the input.
        
        Args:
            inputs: Input of shape `[batch_size, seq_len, hidden_dim]`
            deterministic: Whether to apply dropout
            
        Returns:
            Output of shape `[batch_size, seq_len, hidden_dim]`
        """
        # Layer normalization and attention
        x = nn.LayerNorm()(inputs)
        x = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.hidden_dim // self.num_heads,
            dropout_rate=self.dropout_rate
        )(x, deterministic=deterministic)
        x = nn.Dropout(rate=self.dropout_rate)(
            x, deterministic=deterministic)
        x = x + inputs
        
        # Feed-forward network with residual connection
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate)(
            y, deterministic=deterministic)
        y = nn.Dense(self.hidden_dim)(y)
        y = nn.Dropout(rate=self.dropout_rate)(
            y, deterministic=deterministic)
        
        return x + y


class SimpleLanguageModel(nn.Module):
    """A simple language model using transformer blocks.
    
    This is a basic but educational implementation of a language model
    that can be used for text generation tasks.
    
    Attributes:
        vocab_size: Size of the vocabulary
        hidden_dim: Dimension of the hidden layer
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_dim: Dimension of the feed-forward layer
        dropout_rate: Dropout probability
    """
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray, 
                 deterministic: bool = True) -> jnp.ndarray:
        """Applies the language model to the input tokens.
        
        Args:
            input_ids: Input tokens of shape `[batch_size, seq_len]`
            deterministic: Whether to apply dropout
            
        Returns:
            Logits of shape `[batch_size, seq_len, vocab_size]`
        """
        x = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_dim
        )(input_ids)
        
        # Add positional embeddings
        seq_len = input_ids.shape[1]
        x = x + self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (1, seq_len, self.hidden_dim)
        )
        
        x = nn.Dropout(rate=self.dropout_rate)(
            x, deterministic=deterministic)
        
        # Apply transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate
            )(x, deterministic=deterministic)
        
        # Final layer normalization and output projection
        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)
        
        return logits
