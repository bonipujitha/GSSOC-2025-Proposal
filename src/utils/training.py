"""
Training utilities for JAX/Flax models with educational comments.
This module provides helper functions for training language models.
"""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from typing import Any, Callable, Dict, Tuple

class TrainState(train_state.TrainState):
    """Custom train state that includes loss scaling for mixed precision training."""
    loss_scale: jnp.ndarray

def create_train_state(
    rng: jnp.ndarray,
    model: Any,
    learning_rate: float,
    weight_decay: float
) -> TrainState:
    """Creates initial training state with optimizer.
    
    Args:
        rng: PRNG key
        model: Flax model to train
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay factor
        
    Returns:
        Initial training state
    """
    # Create a learning rate schedule
    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=1000,
        decay_steps=10000,
    )
    
    # Create optimizer with weight decay
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(
            learning_rate=schedule_fn,
            weight_decay=weight_decay
        )
    )
    
    # Initialize model
    params = model.init(rng, jnp.ones((1, 1), dtype=jnp.int32))
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        loss_scale=jnp.array(2.0 ** 15)
    )

def compute_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: jnp.ndarray
) -> jnp.ndarray:
    """Computes cross entropy loss with label smoothing.
    
    Args:
        logits: Model predictions
        labels: Ground truth labels
        mask: Mask for padding tokens
        
    Returns:
        Scalar loss value
    """
    num_classes = logits.shape[-1]
    label_smoothing = 0.1
    
    # Create smoothed labels
    smooth_positives = 1.0 - label_smoothing
    smooth_negatives = label_smoothing / num_classes
    one_hot = jax.nn.one_hot(labels, num_classes)
    smoothed_labels = one_hot * smooth_positives + smooth_negatives
    
    # Compute cross entropy
    losses = -jnp.sum(smoothed_labels * jax.nn.log_softmax(logits), axis=-1)
    
    # Apply mask and compute mean
    return jnp.sum(losses * mask) / jnp.sum(mask)

@jax.jit
def train_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    dropout_rng: jnp.ndarray
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Performs a single training step.
    
    Args:
        state: Current training state
        batch: Batch of training data
        dropout_rng: PRNG key for dropout
        
    Returns:
        New state and metrics dictionary
    """
    # Training function
    def loss_fn(params):
        logits = state.apply_fn(
            params,
            batch['input_ids'],
            deterministic=False,
            rngs={'dropout': dropout_rng}
        )
        loss = compute_loss(
            logits=logits,
            labels=batch['labels'],
            mask=batch['attention_mask']
        )
        return loss
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    # Update parameters
    new_state = state.apply_gradients(grads=grads)
    
    metrics = {
        'loss': loss,
        'learning_rate': state.opt_state.hyperparams['learning_rate']
    }
    
    return new_state, metrics

@jax.jit
def eval_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray]
) -> Dict[str, jnp.ndarray]:
    """Performs a single evaluation step.
    
    Args:
        state: Current training state
        batch: Batch of evaluation data
        
    Returns:
        Metrics dictionary
    """
    logits = state.apply_fn(
        state.params,
        batch['input_ids'],
        deterministic=True
    )
    
    loss = compute_loss(
        logits=logits,
        labels=batch['labels'],
        mask=batch['attention_mask']
    )
    
    return {'eval_loss': loss}
