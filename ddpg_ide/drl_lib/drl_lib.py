from __future__ import annotations

from collections import deque
from typing import Deque, Tuple
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization


LAYER1_SIZE = 400
LAYER2_SIZE = 300
FINAL_LAYER_INIT_RANGE = 0.003



def _create_uniform_initializer(min_val: float, max_val: float):
    """Create uniform initializer for final layers."""
    return tf.keras.initializers.RandomUniform(minval=min_val, maxval=max_val)


class ActorNetwork(tf.keras.Model):
    """
    Architecture: State -> FC(400) -> LN -> ReLU -> FC(300) -> LN -> ReLU -> FC(action_dim) -> Sigmoid -> Scale
    Output: Action scaled to [0, action_bound]
    """
    
    def __init__(self, state_dim: int, action_dim: int, action_bound: float, name: str = "actor"):
        super().__init__(name=name)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.action_bound = float(action_bound)

        # Layer 1: state -> 400 units
        self.fc1 = tf.keras.layers.Dense(LAYER1_SIZE, name="fc1")
        # CHANGED: BatchNormalization -> LayerNormalization
        self.ln1 = LayerNormalization(name="ln1") 
        self.relu1 = tf.keras.layers.ReLU(name="relu1")

        # Layer 2: 400 -> 300 units
        self.fc2 = tf.keras.layers.Dense(LAYER2_SIZE, name="fc2")
        # CHANGED: BatchNormalization -> LayerNormalization
        self.ln2 = LayerNormalization(name="ln2")
        self.relu2 = tf.keras.layers.ReLU(name="relu2")

        # Output layer: 300 -> action_dim
        initializer = _create_uniform_initializer(-FINAL_LAYER_INIT_RANGE, FINAL_LAYER_INIT_RANGE)
        self.output_layer = tf.keras.layers.Dense(
            self.action_dim,
            activation="sigmoid",
            kernel_initializer=initializer,
            name="output"
        )

    def call(self, states: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through actor network."""
        x = self.fc1(states)
        # CHANGED: Removed training=training (LayerNorm doesn't use it)
        x = self.ln1(x) 
        x = self.relu1(x)

        x = self.fc2(x)
        # CHANGED: Removed training=training
        x = self.ln2(x)
        x = self.relu2(x)

        # Sigmoid output scaled to [0, action_bound]
        action = self.output_layer(x)
        return action * self.action_bound


class CriticNetwork(tf.keras.Model):
    """
    Architecture: (State, Action) -> Q-value
    State path: State -> FC(400) -> LN -> ReLU
    Combined: State_features + Action_features -> FC(300) -> ReLU -> FC(1)
    """
    
    def __init__(self, state_dim: int, action_dim: int, name: str = "critic"):
        super().__init__(name=name)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        # State processing: state -> 400 units
        self.state_fc1 = tf.keras.layers.Dense(LAYER1_SIZE, name="state_fc1")
        # CHANGED: BatchNormalization -> LayerNormalization
        self.state_ln1 = LayerNormalization(name="state_ln1")
        self.state_relu1 = tf.keras.layers.ReLU(name="state_relu1")

        # Combined processing: (state_features + action) -> 300 units
        # Note: LayerNorm is usually not applied immediately after the merge in this specific architecture 
        # (based on the original DDPG paper), but you can add it if convergence is slow. 
        # Here we stick to the original logic which only had BN on the state path.
        self.state_fc2 = tf.keras.layers.Dense(LAYER2_SIZE, use_bias=False, name="state_fc2")
        self.action_fc2 = tf.keras.layers.Dense(LAYER2_SIZE, use_bias=True, name="action_fc2")
        self.combined_relu = tf.keras.layers.ReLU(name="combined_relu")

        # Q-value output
        initializer = _create_uniform_initializer(-FINAL_LAYER_INIT_RANGE, FINAL_LAYER_INIT_RANGE)
        self.q_output = tf.keras.layers.Dense(1, kernel_initializer=initializer, name="q_output")

    def call(self, states: tf.Tensor, actions: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through critic network."""
        # Process state
        state_features = self.state_fc1(states)
        # CHANGED: Removed training=training
        state_features = self.state_ln1(state_features)
        state_features = self.state_relu1(state_features)

        # Combine state and action features
        combined = self.state_fc2(state_features) + self.action_fc2(actions)
        combined = self.combined_relu(combined)

        # Q-value output
        return self.q_output(combined)
    


# NOISE GENERATORS
class OrnsteinUhlenbeckActionNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration noise.
    Useful for continuous control tasks.
    """
    
    def __init__(self, mu: np.ndarray, sigma: float = 0.12, theta: float = 0.15, 
                 dt: float = 1e-2, x0: np.ndarray = None):
        self.theta = float(theta)
        self.mu = np.array(mu, dtype=np.float32)
        self.sigma = float(sigma)
        self.dt = float(dt)
        self.x0 = None if x0 is None else np.array(x0, dtype=np.float32)
        self.reset()

    def __call__(self) -> np.ndarray:
        """Generate next noise sample."""
        dx = self.theta * (self.mu - self.x_prev) * self.dt
        dx += self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = self.x_prev + dx
        return self.x_prev.astype(np.float32)

    def reset(self):
        """Reset noise process to initial state."""
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu, dtype=np.float32)

    def __repr__(self) -> str:
        return f"OrnsteinUhlenbeckActionNoise(mu={self.mu}, sigma={self.sigma})"


class GaussianNoise:
    """
    Annealing Gaussian noise for exploration.
    Noise standard deviation decays over time.
    """
    
    def __init__(self, sigma0: float = 1.0, sigma1: float = 0.0, size: tuple = (1,)):
        self.sigma0 = float(sigma0)
        self.sigma1 = float(sigma1)
        self.size = tuple(size)
        self.decay_rate = 0.9995

    def __call__(self) -> np.ndarray:
        """Generate noise and decay sigma."""
        noise = np.random.normal(0.0, self.sigma0, self.size).astype(np.float32)
        self.sigma0 = max(self.sigma0 * self.decay_rate, self.sigma1)
        return noise



class ReplayBuffer:
    """
    Experience replay buffer for continuous action spaces.
    Stores (state, action, reward, done, next_state) transitions.
    """
    
    def __init__(self, buffer_size: int, random_seed: int = 123):
        self.buffer_size = int(buffer_size)
        self.count = 0
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, bool, np.ndarray]] = deque()
        random.seed(int(random_seed))

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            done: bool, next_state: np.ndarray):
        """Add a transition to the buffer."""
        transition = (
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            bool(done),
            np.asarray(next_state, dtype=np.float32)
        )
        
        if self.count < self.buffer_size:
            self.buffer.append(transition)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(transition)

    def size(self) -> int:
        """Return current buffer size."""
        return self.count

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a random batch of transitions."""
        batch_size = min(int(batch_size), self.count)
        batch = random.sample(self.buffer, batch_size)

        states = np.stack([t[0] for t in batch]).astype(np.float32)
        actions = np.stack([t[1] for t in batch]).astype(np.float32)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        dones = np.array([t[3] for t in batch], dtype=np.bool_)
        next_states = np.stack([t[4] for t in batch]).astype(np.float32)
        
        return states, actions, rewards, dones, next_states

    def clear(self):
        """Clear all transitions from buffer."""
        self.buffer.clear()
        self.count = 0



# TARGET NETWORK UPDATE UTILITIES
# @tf.function
def soft_update(target: tf.keras.Model, source: tf.keras.Model, tau: float):
    """
    Soft update: target = tau * source + (1 - tau) * target
    """
    tau = tf.convert_to_tensor(tau, dtype=tf.float32)
    one_minus_tau = 1.0 - tau
    
    for target_var, source_var in zip(target.variables, source.variables):
        target_var.assign(target_var * one_minus_tau + source_var * tau)


# @tf.function
def hard_update(target: tf.keras.Model, source: tf.keras.Model):
    """
    Hard update: target = source
    """
    for target_var, source_var in zip(target.variables, source.variables):
        target_var.assign(source_var)
        
class BaseAgent:
    """Base class for all RL agents with common checkpoint utilities."""
    
    def save_checkpoint(self, checkpoint_dir: str, max_to_keep: int = 3) -> str:
        """Save agent checkpoint."""
        raise NotImplementedError
    
    def load_checkpoint(self, checkpoint_dir: str) -> str:
        """Load agent checkpoint."""
        raise NotImplementedError