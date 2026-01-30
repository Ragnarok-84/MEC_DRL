from __future__ import annotations

import numpy as np
import tensorflow as tf
import os

from typing import Tuple, Optional



from drl_lib.drl_lib import (
    ActorNetwork, CriticNetwork, 
    OrnsteinUhlenbeckActionNoise, ReplayBuffer, 
    BaseAgent,
    soft_update, hard_update
)



class DDPGAgent(BaseAgent):
        
    def __init__(self, user_config: dict, train_config: dict):
        # Environment parameters
        self.user_id = user_config['id']
        self.state_dim = int(user_config['state_dim'])
        self.action_dim = int(user_config['action_dim'])
        self.action_bound = float(user_config['action_bound'])
        
        # Training parameters
        self.batch_size = int(train_config['minibatch_size'])
        self.tau = tf.constant(float(train_config['tau']), dtype=tf.float32)
        self.gamma = tf.constant(float(train_config['gamma']), dtype=tf.float32)
        self.is_training = bool(train_config.get('is_training', False))
        
        # Build networks
        self._build_networks(train_config)
        
        # Initialize replay buffer and noise
        buffer_size = int(train_config['buffer_size'])
        random_seed = int(train_config['random_seed'])
        self.replay_buffer = ReplayBuffer(buffer_size, random_seed)
        
        noise_sigma = float(train_config['noise_sigma'])
        self.exploration_noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(self.action_dim, dtype=np.float32),
            sigma=noise_sigma
        )
        
        # Compile training functions
        self._compile_training_functions()
        
        # Optional: Load pretrained weights
        init_path = user_config.get('init_path', '')
        if init_path and os.path.exists(init_path):
            self._load_pretrained_weights(init_path)

    def _build_networks(self, train_config: dict):
        """Build actor, critic and their target networks."""
        # Actor networks
        self.actor = ActorNetwork(
            self.state_dim, self.action_dim, self.action_bound,
            name=f"actor_{self.user_id}"
        )
        self.actor_target = ActorNetwork(
            self.state_dim, self.action_dim, self.action_bound,
            name=f"actor_target_{self.user_id}"
        )
        
        # Critic networks
        self.critic = CriticNetwork(
            self.state_dim, self.action_dim,
            name=f"critic_{self.user_id}"
        )
        self.critic_target = CriticNetwork(
            self.state_dim, self.action_dim,
            name=f"critic_target_{self.user_id}"
        )
        
        # Build networks by calling them once
        dummy_state = tf.zeros((1, self.state_dim), dtype=tf.float32)
        dummy_action = tf.zeros((1, self.action_dim), dtype=tf.float32)
        
        _ = self.actor(dummy_state, training=False)
        _ = self.actor_target(dummy_state, training=False)
        _ = self.critic(dummy_state, dummy_action, training=False)
        _ = self.critic_target(dummy_state, dummy_action, training=False)
        
        # Initialize target networks
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        
        # Optimizers
        actor_lr = float(train_config['actor_lr'])
        critic_lr = float(train_config['critic_lr'])
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    def _compile_training_functions(self):
        
        bs = self.batch_size
        
        # Define input signatures
        state_sig = tf.TensorSpec(shape=(bs, self.state_dim), dtype=tf.float32)
        action_sig = tf.TensorSpec(shape=(bs, self.action_dim), dtype=tf.float32)
        reward_sig = tf.TensorSpec(shape=(bs,), dtype=tf.float32)
        done_sig = tf.TensorSpec(shape=(bs,), dtype=tf.bool)
        
        @tf.function(input_signature=[state_sig, action_sig, reward_sig, done_sig, state_sig])
        def train_critic_only(states, actions, rewards, dones, next_states):
            
            # Compute target Q-values
            next_actions = self.actor_target(next_states, training=False)
            target_q = self.critic_target(next_states, next_actions, training=False)
            target_q = tf.squeeze(target_q, axis=1)
            
            not_done = tf.cast(tf.logical_not(dones), tf.float32)
            y = tf.stop_gradient(rewards + self.gamma * not_done * target_q)
            
            # Update critic
            with tf.GradientTape() as tape:
                q_pred = self.critic(states, actions, training=self.is_training)
                q_pred = tf.squeeze(q_pred, axis=1)
                critic_loss = tf.reduce_mean(tf.square(y - q_pred))
            
            gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(gradients, self.critic.trainable_variables)
            )
            
            # Soft update target
            soft_update(self.critic_target, self.critic, self.tau)
            return critic_loss
        
        @tf.function(input_signature=[state_sig, action_sig, reward_sig, done_sig, state_sig])
        def train_actor_and_critic(states, actions, rewards, dones, next_states):
            
            # --- Update Critic ---
            next_actions = self.actor_target(next_states, training=False)
            target_q = self.critic_target(next_states, next_actions, training=False)
            target_q = tf.squeeze(target_q, axis=1)
            
            not_done = tf.cast(tf.logical_not(dones), tf.float32)
            y = tf.stop_gradient(rewards + self.gamma * not_done * target_q)
            
            with tf.GradientTape() as critic_tape:
                q_pred = self.critic(states, actions, training=self.is_training)
                q_pred = tf.squeeze(q_pred, axis=1)
                critic_loss = tf.reduce_mean(tf.square(y - q_pred))
            
            critic_gradients = critic_tape.gradient(
                critic_loss, self.critic.trainable_variables
            )
            self.critic_optimizer.apply_gradients(
                zip(critic_gradients, self.critic.trainable_variables)
            )
            
            # --- Update Actor ---
            with tf.GradientTape() as actor_tape:
                predicted_actions = self.actor(states, training=self.is_training)
                q_values = self.critic(states, predicted_actions, training=False)
                actor_loss = -tf.reduce_mean(q_values)
            
            actor_gradients = actor_tape.gradient(
                actor_loss, self.actor.trainable_variables
            )
            self.actor_optimizer.apply_gradients(
                zip(actor_gradients, self.actor.trainable_variables)
            )
            
            # Soft update targets
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
            
            return actor_loss, critic_loss
        
        # Prediction function
        @tf.function(input_signature=[tf.TensorSpec(shape=(1, self.state_dim), dtype=tf.float32)])
        def predict_action(state):
            return self.actor(state, training=False)
        
        self._train_critic_fn = train_critic_only
        self._train_both_fn = train_actor_and_critic
        self._predict_fn = predict_action

    def predict(self, state: np.ndarray, use_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        
        state_tf = tf.convert_to_tensor(
            np.reshape(state, (1, self.state_dim)).astype(np.float32)
        )
        action = self._predict_fn(state_tf).numpy()[0]
        
        noise = self.exploration_noise() if use_noise else np.zeros(self.action_dim, dtype=np.float32)
        return action + noise, noise

    def update(self, state: np.ndarray, action: np.ndarray, reward: float,
               done: bool, next_state: np.ndarray, update_actor: bool = True) -> Tuple[Optional[float], Optional[float]]:
        
        # Add to replay buffer
        self.replay_buffer.add(
            np.reshape(state, (self.state_dim,)),
            np.reshape(action, (self.action_dim,)),
            reward, done,
            np.reshape(next_state, (self.state_dim,))
        )
        
        # Train if buffer has enough samples
        if self.replay_buffer.size() < self.batch_size:
            return None, None
        
        # Sample batch
        s_batch, a_batch, r_batch, d_batch, s2_batch = \
            self.replay_buffer.sample_batch(self.batch_size)
        
        # Convert to tensors
        s_batch = tf.convert_to_tensor(s_batch, dtype=tf.float32)
        a_batch = tf.convert_to_tensor(a_batch, dtype=tf.float32)
        r_batch = tf.convert_to_tensor(r_batch, dtype=tf.float32)
        d_batch = tf.convert_to_tensor(d_batch, dtype=tf.bool)
        s2_batch = tf.convert_to_tensor(s2_batch, dtype=tf.float32)
        
        # Train networks
        if update_actor:
            actor_loss, critic_loss = self._train_both_fn(
                s_batch, a_batch, r_batch, d_batch, s2_batch
            )
            return float(actor_loss.numpy()), float(critic_loss.numpy())
        else:
            critic_loss = self._train_critic_fn(
                s_batch, a_batch, r_batch, d_batch, s2_batch
            )
            return None, float(critic_loss.numpy())

    def _load_pretrained_weights(self, path: str):
        
        try:
            data = np.load(path, allow_pickle=True)
            weights = list(data['arr_0'])
            
            actor_weights = self.actor.get_weights()
            n = len(actor_weights)
            
            if len(weights) == n:
                self.actor.set_weights(weights)
                self.actor_target.set_weights(weights)
            elif len(weights) == 2 * n:
                self.actor.set_weights(weights[:n])
                self.actor_target.set_weights(weights[n:])
            
            hard_update(self.critic_target, self.critic)
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights from {path}: {e}")

    def init_target_network(self):
        
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def save_checkpoint(self, checkpoint_dir: str, max_to_keep: int = 3) -> str:
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = tf.train.Checkpoint(
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer
        )
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=max_to_keep)
        return manager.save()

    def load_checkpoint(self, checkpoint_dir: str) -> str:
        
        checkpoint = tf.train.Checkpoint(
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer
        )
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
        if manager.latest_checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in: {checkpoint_dir}")
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        return manager.latest_checkpoint

class DDPGAgentWithTracking(DDPGAgent):
    
    
    def __init__(self, user_config: dict, train_config: dict):
        super().__init__(user_config, train_config)
        
        # Tracking metrics cho IDE
        self.last_critic_loss = None
        self.last_td_error = None
        self.last_actor_loss = None
    
    def update(self, state: np.ndarray, action: np.ndarray, reward: float,
               done: bool, next_state: np.ndarray, update_actor: bool = True):
        
        # Add to replay buffer
        self.replay_buffer.add(
            np.reshape(state, (self.state_dim,)),
            np.reshape(action, (self.action_dim,)),
            reward,
            done,
            np.reshape(next_state, (self.state_dim,))
        )

        # Train if buffer has enough samples
        if self.replay_buffer.size() < self.batch_size:
            return None, None

        # Sample batch
        s_batch, a_batch, r_batch, d_batch, s2_batch = \
            self.replay_buffer.sample_batch(self.batch_size)

        # Convert to tensors
        s_batch = tf.convert_to_tensor(s_batch, dtype=tf.float32)
        a_batch = tf.convert_to_tensor(a_batch, dtype=tf.float32)
        r_batch = tf.convert_to_tensor(r_batch, dtype=tf.float32)
        d_batch = tf.convert_to_tensor(d_batch, dtype=tf.bool)
        s2_batch = tf.convert_to_tensor(s2_batch, dtype=tf.float32)

        if update_actor:
            # Train both actor and critic
            actor_loss, critic_loss = self._train_both_fn(
                s_batch, a_batch, r_batch, d_batch, s2_batch
            )
            
            self.last_actor_loss = float(actor_loss.numpy())
            self.last_critic_loss = float(critic_loss.numpy())
            
            # Calculate TD error for stability tracking
            self.last_td_error = self._compute_td_error(
                s_batch, a_batch, r_batch, d_batch, s2_batch
            )
            
            return self.last_actor_loss, self.last_critic_loss
        else:
            # Train critic only
            critic_loss = self._train_critic_fn(
                s_batch, a_batch, r_batch, d_batch, s2_batch
            )
            
            self.last_critic_loss = float(critic_loss.numpy())
            
            # Calculate TD error
            self.last_td_error = self._compute_td_error(
                s_batch, a_batch, r_batch, d_batch, s2_batch
            )
            
            return None, self.last_critic_loss
    
    def _compute_td_error(self, states, actions, rewards, dones, next_states):
        
        # Target Q-value
        next_actions = self.actor_target(next_states, training=False)
        target_q = self.critic_target(next_states, next_actions, training=False)
        target_q = tf.squeeze(target_q, axis=1)
        
        not_done = tf.cast(tf.logical_not(dones), tf.float32)
        y = rewards + self.gamma * not_done * target_q
        
        # Current Q-value
        q_pred = self.critic(states, actions, training=False)
        q_pred = tf.squeeze(q_pred, axis=1)
        
        # TD error
        td_error = tf.reduce_mean(tf.abs(y - q_pred))
        
        return float(td_error.numpy())
    
    def get_tracking_metrics(self):
        
        return {
            'actor_loss': self.last_actor_loss,
            'critic_loss': self.last_critic_loss,
            'td_error': self.last_td_error
        }
