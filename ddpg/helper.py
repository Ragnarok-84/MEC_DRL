
from __future__ import annotations

import os
import time
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ddpg_lib import (
    ActorNetwork,
    CriticNetwork,
    OrnsteinUhlenbeckActionNoise,
    ReplayBuffer,
    soft_update,
    hard_update,
)


state_avg = np.array([0.031,0.153,0.399,0.772,1.274,1.911,2.694,3.630,4.730,6.021,7.902])
trans_p = np.array([[0.514,0.514,1.000,],
                    [0.513,0.696,1.000,],
                    [0.513,0.745,1.000,],
                    [0.515,0.776,1.000,],
                    [0.513,0.799,1.000,],
                    [0.514,0.821,1.000,],
                    [0.516,0.842,1.000,],
                    [0.511,0.858,1.000,],
                    [0.516,0.880,1.000,],
                    [0.512,0.897,1.000,],
                    [0.671,1.000,1.000,],])   
alpha = 3.0
ref_loss = 0.001


class MarkovModel:
    """Finite-state Markov channel model (matches TF1)."""
    def __init__(self, dis, seed: int = 123):
        self.dis = dis
        self.path_loss = ref_loss*np.power(1./dis, alpha)
        np.random.seed([seed])

        self.trans_p = trans_p
        self.state_avg = state_avg
        self.state = np.random.randint(0, 11)

    def getCh(self):
        return np.array([np.sqrt(self.path_loss*self.state_avg[self.state])], dtype=np.float32)

    def sampleCh(self):
        temp = np.random.random()
        if temp >= trans_p[self.state, 1]:
            self.state += 1
        elif temp >= trans_p[self.state, 0]:
            self.state -= 1

        self.state = int(np.fmax(np.fmin(self.state, 11), 0))
        return self.getCh()


def complexGaussian(row=1, col=1, amp=1.0):
    real = np.random.normal(size=[row, col])[0] * np.sqrt(0.5)
    img = np.random.normal(size=[row, col])[0] * np.sqrt(0.5)
    return amp * (real + 1j * img)


class ARModel:
    """Auto-regressive complex channel model (matches TF1)."""
    def __init__(self, dis, n_t=1, n_r=1, rho=0.95, seed: int = 123):
        self.dis = dis
        self.n_t = n_t
        self.n_r = n_r
        self.path_loss = ref_loss*np.power(1./dis, alpha)
        np.random.seed([seed])

        self.rho = rho
        self.H = complexGaussian(self.n_t, self.n_r)

    def getCh(self):
        return self.H * np.sqrt(self.path_loss)

    def sampleCh(self):
        self.H = self.rho*self.H + complexGaussian(self.n_t, self.n_r, np.sqrt(1-self.rho*self.rho))
        return self.getCh()





class DDPGAgentLD:
    """Load-only agent for inference from a TF2 checkpoint directory."""
    def __init__(self, user_config, ckpt_dir: str):
        self.user_id = user_config['id']
        self.state_dim = int(user_config['state_dim'])
        self.action_dim = int(user_config['action_dim'])
        self.action_bound = float(user_config['action_bound'])

        self.actor = ActorNetwork(self.state_dim, self.action_dim, self.action_bound, name=f"actor_{self.user_id}")
        # Build
        _ = self.actor(tf.zeros((1, self.state_dim), dtype=tf.float32), training=False)

        ckpt = tf.train.Checkpoint(actor=self.actor)
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
        if manager.latest_checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in: {ckpt_dir}")
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print(f"Loaded DDPG checkpoint from: {manager.latest_checkpoint}")

        @tf.function(input_signature=[tf.TensorSpec(shape=(1, self.state_dim), dtype=tf.float32)])
        def _predict_fn(s):
            return self.actor(s, training=False)
        self._predict_fn = _predict_fn

    def predict(self, s):
        s = tf.convert_to_tensor(np.reshape(s, (1, self.state_dim)).astype(np.float32))
        a = self._predict_fn(s).numpy()[0]
        return a





class DDPGAgent:
    def __init__(self, user_config, train_config):
        self.user_id = user_config['id']
        self.state_dim = int(user_config['state_dim'])
        self.action_dim = int(user_config['action_dim'])
        self.action_bound = float(user_config['action_bound'])
        self.init_path = user_config.get('init_path', '')

        self.minibatch_size = int(train_config['minibatch_size'])
        self.noise_sigma = float(train_config['noise_sigma'])

        self.tau = tf.constant(float(train_config['tau']), dtype=tf.float32)
        self.gamma = tf.constant(float(train_config['gamma']), dtype=tf.float32)
        self.is_training = bool(train_config.get("is_training", False))  

        actor_lr = float(train_config['actor_lr'])
        critic_lr = float(train_config['critic_lr'])

        self.actor = ActorNetwork(self.state_dim, self.action_dim, self.action_bound, name=f"actor_{self.user_id}")
        self.actor_target = ActorNetwork(self.state_dim, self.action_dim, self.action_bound, name=f"actor_t_{self.user_id}")
        self.critic = CriticNetwork(self.state_dim, self.action_dim, name=f"critic_{self.user_id}")
        self.critic_target = CriticNetwork(self.state_dim, self.action_dim, name=f"critic_t_{self.user_id}")

        # Build networks (create variables)
        s0 = tf.zeros((1, self.state_dim), dtype=tf.float32)
        a0 = tf.zeros((1, self.action_dim), dtype=tf.float32)
        _ = self.actor(s0, training=False)
        _ = self.actor_target(s0, training=False)
        _ = self.critic(s0, a0, training=False)
        _ = self.critic_target(s0, a0, training=False)

        # Initialize target weights (hard copy at start)
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        self.replay_buffer = ReplayBuffer(int(train_config['buffer_size']), int(train_config['random_seed']))
        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim, dtype=np.float32), sigma=self.noise_sigma)

       
        bs = self.minibatch_size
        s_sig = tf.TensorSpec(shape=(bs, self.state_dim), dtype=tf.float32)
        a_sig = tf.TensorSpec(shape=(bs, self.action_dim), dtype=tf.float32)
        r_sig = tf.TensorSpec(shape=(bs,), dtype=tf.float32)
        t_sig = tf.TensorSpec(shape=(bs,), dtype=tf.bool)
        s2_sig = tf.TensorSpec(shape=(bs, self.state_dim), dtype=tf.float32)

        @tf.function(input_signature=[s_sig, a_sig, r_sig, t_sig, s2_sig])
        def _train_critic(s, a, r, done, s2):
            # target y = r + gamma*(1-done)*Q'(s2, mu'(s2))
            a2 = self.actor_target(s2, training=False)
            q2 = self.critic_target(s2, a2, training=False)
            q2 = tf.squeeze(q2, axis=1)  # (bs,)

            not_done = tf.cast(tf.logical_not(done), tf.float32)
            y = r + self.gamma * not_done * q2
            y = tf.stop_gradient(y)

            with tf.GradientTape() as tape:
                q = self.critic(s, a, training=self.is_training)
                q = tf.squeeze(q, axis=1)
                loss = tf.reduce_mean(tf.square(y - q))

            grads = tape.gradient(loss, self.critic.trainable_variables)
            self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))

            soft_update(self.critic_target, self.critic, self.tau)
            return loss

        @tf.function(input_signature=[s_sig, a_sig, r_sig, t_sig, s2_sig])
        def _train_actor_and_critic(s, a, r, done, s2):
            # --- Critic update ---
            a2 = self.actor_target(s2, training=False)
            q2 = self.critic_target(s2, a2, training=False)
            q2 = tf.squeeze(q2, axis=1)

            not_done = tf.cast(tf.logical_not(done), tf.float32)
            y = tf.stop_gradient(r + self.gamma * not_done * q2)

            with tf.GradientTape() as tape_c:
                q = self.critic(s, a, training=self.is_training)
                q = tf.squeeze(q, axis=1)
                critic_loss = tf.reduce_mean(tf.square(y - q))

            critic_grads = tape_c.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

            
            with tf.GradientTape() as tape_a:
                a_pred = self.actor(s, training=self.is_training)
                q_pred = self.critic(s, a_pred, training=False)
                actor_loss = -tf.reduce_mean(q_pred)

            actor_grads = tape_a.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
            return actor_loss, critic_loss

        self._train_critic = _train_critic
        self._train_actor_and_critic = _train_actor_and_critic

        @tf.function(input_signature=[tf.TensorSpec(shape=(1, self.state_dim), dtype=tf.float32)])
        def _predict_action(s):
            return self.actor(s, training=False)
        self._predict_action = _predict_action

    def init_target_network(self):
        if isinstance(self.init_path, str) and len(self.init_path) > 0 and os.path.exists(self.init_path):
            res = np.load(self.init_path, allow_pickle=True)
            weights_load = res['arr_0']

            actor_w = self.actor.get_weights()
            n = len(actor_w)

            w_list = list(weights_load)
            if len(w_list) == n:
                self.actor.set_weights(w_list)
                self.actor_target.set_weights(w_list)
            elif len(w_list) == 2*n:
                self.actor.set_weights(w_list[:n])
                self.actor_target.set_weights(w_list[n:])
            else:
                raise ValueError(
                    f"Unexpected weight count in npz: {len(w_list)}. "
                    f"Expected {n} or {2*n} for actor/actor_target."
                )

            hard_update(self.critic_target, self.critic)
        else:
            hard_update(self.actor_target, self.actor)
            hard_update(self.critic_target, self.critic)

    def predict(self, s, isUpdateActor: bool):
        noise = self.actor_noise() if isUpdateActor else np.zeros(self.action_dim, dtype=np.float32)
        s_tf = tf.convert_to_tensor(np.reshape(s, (1, self.state_dim)).astype(np.float32))
        a = self._predict_action(s_tf).numpy()[0]
        return a + noise, noise

    def update(self, s, a, r, t, s2, isUpdateActor: bool):
        self.replay_buffer.add(
            np.reshape(s, (self.state_dim,)),
            np.reshape(a, (self.action_dim,)),
            r,
            t,
            np.reshape(s2, (self.state_dim,))
        )

        if self.replay_buffer.size() >= self.minibatch_size:
            s_b, a_b, r_b, t_b, s2_b = self.replay_buffer.sample_batch(self.minibatch_size)

            s_b = tf.convert_to_tensor(s_b, dtype=tf.float32)
            a_b = tf.convert_to_tensor(a_b, dtype=tf.float32)
            r_b = tf.convert_to_tensor(r_b, dtype=tf.float32)
            t_b = tf.convert_to_tensor(t_b, dtype=tf.bool)
            s2_b = tf.convert_to_tensor(s2_b, dtype=tf.float32)

            if isUpdateActor:
                a_loss, c_loss = self._train_actor_and_critic(s_b, a_b, r_b, t_b, s2_b)
                return float(a_loss.numpy()), float(c_loss.numpy())
            else:
                c_loss = self._train_critic(s_b, a_b, r_b, t_b, s2_b)
                return None, float(c_loss.numpy())

        return None, None

    
    def save(self, ckpt_dir: str, max_to_keep: int = 3):
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = tf.train.Checkpoint(
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            actor_opt=self.actor_opt,
            critic_opt=self.critic_opt,
        )
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=max_to_keep)
        return manager.save()

    def load(self, ckpt_dir: str):
        ckpt = tf.train.Checkpoint(
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            actor_opt=self.actor_opt,
            critic_opt=self.critic_opt,
        )
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
        if manager.latest_checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in: {ckpt_dir}")
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        return manager.latest_checkpoint


