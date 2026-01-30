
from __future__ import annotations

from collections import deque
import random
from typing import Deque, Tuple, Optional

import numpy as np
import tensorflow as tf



LAYER1 = 400
LAYER2 = 300


def _uniform_init_3e3():
    
    return tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)


class ActorNetwork(tf.keras.Model):
    
    def __init__(self, state_dim: int, action_dim: int, action_bound: float, name: str = "actor"):
        super().__init__(name=name)
        self.s_dim = int(state_dim)
        self.a_dim = int(action_dim)
        self.action_bound = float(action_bound)

        self.fc1 = tf.keras.layers.Dense(LAYER1, name="fc1")
        self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")
        self.act1 = tf.keras.layers.ReLU(name="relu1")

        self.fc2 = tf.keras.layers.Dense(LAYER2, name="fc2")
        self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")
        self.act2 = tf.keras.layers.ReLU(name="relu2")

        self.out_layer = tf.keras.layers.Dense(
            self.a_dim,
            activation="sigmoid",
            kernel_initializer=_uniform_init_3e3(),
            name="out",
        )

    def call(self, states: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.fc1(states)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)

        a = self.out_layer(x)
        return a * self.action_bound


class CriticNetwork(tf.keras.Model):
    
    def __init__(self, state_dim: int, action_dim: int, name: str = "critic"):
        super().__init__(name=name)
        self.s_dim = int(state_dim)
        self.a_dim = int(action_dim)

        self.s_fc1 = tf.keras.layers.Dense(LAYER1, name="s_fc1")
        self.s_bn1 = tf.keras.layers.BatchNormalization(name="s_bn1")
        self.s_act1 = tf.keras.layers.ReLU(name="s_relu1")


        self.s_fc2_lin = tf.keras.layers.Dense(LAYER2, use_bias=False, name="s_fc2_lin")
        self.a_fc2_lin = tf.keras.layers.Dense(LAYER2, use_bias=True, name="a_fc2_lin")
        self.s_act2 = tf.keras.layers.ReLU(name="relu2")

        self.q_out = tf.keras.layers.Dense(1, kernel_initializer=_uniform_init_3e3(), name="q")

    def call(self, states: tf.Tensor, actions: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.s_fc1(states)
        x = self.s_bn1(x, training=training)
        x = self.s_act1(x)

        x = self.s_fc2_lin(x) + self.a_fc2_lin(actions)
        x = self.s_act2(x)

        return self.q_out(x)



class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.12, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = np.array(mu, dtype=np.float32)
        self.sigma = float(sigma)
        self.dt = float(dt)
        self.x0 = None if x0 is None else np.array(x0, dtype=np.float32)
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape).astype(np.float32)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu, dtype=np.float32)

    def __repr__(self):
        return f"OrnsteinUhlenbeckActionNoise(mu={self.mu}, sigma={self.sigma})"


class GaussianNoise:
    def __init__(self, sigma0=1.0, sigma1=0.0, size=(1,)):
        self.sigma0 = float(sigma0)
        self.sigma1 = float(sigma1)
        self.size = tuple(size)

    def __call__(self):
        self.sigma0 *= 0.9995
        self.sigma0 = float(np.fmax(self.sigma0, self.sigma1))
        return np.random.normal(0.0, self.sigma0, self.size).astype(np.float32)



class ReplayBuffer:
    def __init__(self, buffer_size: int, random_seed: int = 123):
        self.buffer_size = int(buffer_size)
        self.count = 0
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, bool, np.ndarray]] = deque()
        random.seed(int(random_seed))

    def add(self, s, a, r, t, s2):
        s = np.asarray(s, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        r = float(r)
        t = bool(t)
        s2 = np.asarray(s2, dtype=np.float32)

        exp = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(exp)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(exp)

    def size(self) -> int:
        return self.count

    def sample_batch(self, batch_size: int):
        batch_size = int(batch_size)
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.stack([b[0] for b in batch]).astype(np.float32)
        a_batch = np.stack([b[1] for b in batch]).astype(np.float32)
        r_batch = np.array([b[2] for b in batch], dtype=np.float32)
        t_batch = np.array([b[3] for b in batch], dtype=np.bool_)
        s2_batch = np.stack([b[4] for b in batch]).astype(np.float32)
        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0



@tf.function
def soft_update(target: tf.keras.Model, source: tf.keras.Model, tau: tf.Tensor):
    """ target = tau*source + (1-tau)*target """
    tau = tf.convert_to_tensor(tau, dtype=tf.float32)
    one_minus = 1.0 - tau
    for (t, s) in zip(target.variables, source.variables):
        t.assign(t * one_minus + s * tau)


@tf.function
def hard_update(target: tf.keras.Model, source: tf.keras.Model):
    for (t, s) in zip(target.variables, source.variables):
        t.assign(s)
