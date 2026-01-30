from __future__ import annotations
from typing import List, Tuple, Optional

import numpy as np

from system.helper import MarkovChannelModel, ARChannelModel
from drl_lib.ddpg_lib import  DDPGAgent
from system.IDE_extension import IDEActionOptimizer, IDEScheduler, fitness_advanced

import numpy as np

# Local processing parameters
COMPUTATION_EFFICIENCY = 1e-27  # k: energy coefficient for local computation
TIME_SLOT_DURATION = 0.001      # t: time slot duration (seconds)
TASK_SIZE = 500                  # L: task size (bits)


class MecTerm:
        
    def __init__(self, user_config: dict, train_config: dict):
        # User identification and task parameters
        self.user_id = user_config['id']
        self.task_arrival_rate = user_config['rate']
        self.distance = user_config['dis']
        
        # State and action space
        self.state_dim = int(user_config['state_dim'])
        self.action_dim = int(user_config['action_dim'])
        self.action_bound = float(user_config['action_bound'])
        
        # Buffer and reward parameters
        self.data_buffer_size = user_config['data_buf_size']
        self.tradeoff_factor = user_config['t_factor']  # Balance between power and buffer
        self.overflow_penalty = user_config['penalty']
        
        # Training parameters
        self.noise_power = float(train_config['sigma2'])
        
        # Agent initialization (to be set by subclasses)
        self.agent = None
        self.update_actor = True
        self.init_sequence_count = 0
        
        # Initialize channel model
        self.channel_model = self._create_channel_model(user_config, train_config)
        
        # State variables
        self.data_buffer = 0.0
        self.channel_coefficient = self.channel_model.get_channel()
        self.sinr = 0.0
        self.power_allocation = np.zeros(self.action_dim, dtype=np.float32)
        self.reward = 0.0
        self.state = np.zeros(self.state_dim, dtype=np.float32)

    def _create_channel_model(self, user_config: dict, train_config: dict):
        if 'model' not in user_config:
            return MarkovChannelModel(
                distance=self.distance,
                seed=int(train_config['random_seed'])
            )
        else:
            n_transmit = 1
            n_receive = int(user_config['num_r'])
            return ARChannelModel(
                distance=self.distance,
                n_transmit=n_transmit,
                n_receive=n_receive,
                seed=int(train_config['random_seed'])
            )

    
    # Local Computation Methods  
    def compute_local_processing_bits(self, power: float) -> float:
        cpu_cycles = np.power(power / COMPUTATION_EFFICIENCY, 1.0 / 3.0)
        bits_processed = cpu_cycles * TIME_SLOT_DURATION / TASK_SIZE / 1000 # chuẩn hóa sang Kbits
        return bits_processed

    def compute_power_from_bits(self, bits: float) -> float:
        # Hàm ngược của compute_local_processing_bits
        cpu_cycles_needed = bits * 1000 * TASK_SIZE / TIME_SLOT_DURATION
        power = np.power(cpu_cycles_needed, 3.0) * COMPUTATION_EFFICIENCY
        return power

    def compute_offload_power_from_rate(self, transmission_rate: float) -> float:
        
        channel_gain = np.power(np.linalg.norm(self.channel_coefficient), 2)
        power = (np.power(2.0, transmission_rate) - 1) * self.noise_power / channel_gain
        return power

    def compute_offload_power_from_rate_sinr(self, transmission_rate: float) -> float:
    
        if self.sinr <= 1e-12:
            return self.action_bound
        return (np.power(2.0, transmission_rate) - 1) / self.sinr

    
    def get_channel_coefficient(self) -> np.ndarray:
        return self.channel_coefficient

    def update_channel_sample(self):
        self.channel_coefficient = self.channel_model.sample_next_channel()
        return self.channel_coefficient

    def set_sinr_and_update_state(self, sinr: float):
        self.sinr = float(sinr)
        self.update_channel_sample()
        
        channel_gain = np.power(np.linalg.norm(self.channel_coefficient), 2) / self.noise_power
        self.state = np.array([self.data_buffer, self.sinr, channel_gain], dtype=np.float32)

        
    def process_tasks_and_sample_arrivals(self) -> Tuple[float, float, float, float]:
       
        offloaded_bits = np.log2(1 + self.power_allocation[0] * self.sinr)
        
        locally_processed_bits = self.compute_local_processing_bits(self.power_allocation[1])
        
        
        self.data_buffer -= (offloaded_bits + locally_processed_bits)
        
        
        wasted_power = 0.0
        if self.data_buffer < 0:
            
            needed_bits = max(0.0, self.data_buffer + locally_processed_bits)
            needed_power = self.compute_power_from_bits(needed_bits)
            wasted_power = self.power_allocation[1] - needed_power
            self.data_buffer = 0.0
        
        
        arrived_bits = np.random.poisson(self.task_arrival_rate)
        self.data_buffer += arrived_bits
        
        return  locally_processed_bits, offloaded_bits, arrived_bits, wasted_power

    
    
    def compute_reward(self) -> float:
        total_power = np.sum(self.power_allocation)
        power_cost = self.tradeoff_factor * total_power * 10
        buffer_cost = (1 - self.tradeoff_factor) * self.data_buffer
        return -(power_cost + buffer_cost)

    def reset(self, arrival_rate: float, sequence_count: int) -> float:

        self.task_arrival_rate = arrival_rate
        self.data_buffer = np.random.randint(0, self.data_buffer_size - 1) / 2.0
        self.update_channel_sample()
        
        if sequence_count >= self.init_sequence_count:
            self.update_actor = True
        
        return self.data_buffer
    
    def predict(self, is_random: bool) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Subclass must implement predict()")

    def feedback(self, sinr: float, done: bool) -> Tuple[float, ...]:
        raise NotImplementedError("Subclass must implement feedback()")


# RL TERMINAL (Training Mode)
class MecTermRL(MecTerm):
    
    def __init__(self, user_config: dict, train_config: dict):
        super().__init__(user_config, train_config)
        
        agent_type = train_config.get('agent_type', 'ddpg').lower() # Mặc định là 'ddpg'
        
        if agent_type == 'td3':
            self.agent = TD3Agent(user_config, train_config)
            print(f"User {self.user_id}: Initialized TD3 Agent")
        elif agent_type == 'ddpg':
            self.agent = DDPGAgent(user_config, train_config)
            print(f"User {self.user_id}: Initialized DDPG Agent")
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}. Use 'ddpg' or 'td3'.")
        
        # Optional: Load pretrained weights
        if 'init_path' in user_config and user_config['init_path']:
            self.init_path = user_config['init_path']
            self.init_sequence_count = user_config.get('init_seqCnt', 0)
            self.update_actor = False

    def predict(self, is_random: bool) -> Tuple[np.ndarray, np.ndarray]:
        power, noise = self.agent.predict(self.state, use_noise=self.update_actor)
        
        # Clip to valid range [0, action_bound]
        self.power_allocation = np.clip(power, 0.0, self.action_bound).astype(np.float32)
        
        return self.power_allocation, noise

    def feedback(self, sinr: float, done: bool) -> Tuple[float, ...]:
        self.sinr = float(sinr)
        
        # Process tasks and calculate metrics
        processed_bits, offloaded_bits, arrived_bits, wasted_power = \
            self.process_tasks_and_sample_arrivals()
        
        # Compute reward
        self.reward = self.compute_reward()
        
        # Update channel and state
        self.update_channel_sample()
        channel_gain = np.power(np.linalg.norm(self.channel_coefficient), 2) / self.noise_power
        next_state = np.array([self.data_buffer, self.sinr, channel_gain], dtype=np.float32)
        
        # Update RL agent
        self.agent.update(
            state=self.state,
            action=self.power_allocation,
            reward=self.reward,
            done=done,
            next_state=next_state,
            update_actor=self.update_actor
        )
        
        self.state = next_state
        
        # Calculate metrics
        total_power = np.sum(self.power_allocation) - wasted_power
        is_overflow = 0  
        
        return (
            self.reward,
            total_power,
            wasted_power,
            offloaded_bits,
            processed_bits,
            arrived_bits,
            self.data_buffer,
            channel_gain,
            is_overflow
        )



class MecTermLD(MecTerm):
        
    def __init__(self, user_config: dict, train_config: dict):
        super().__init__(user_config, train_config)
        
        # Load pretrained agent
        ckpt_dir = user_config.get('ckpt_dir', '')
        if not ckpt_dir:
            raise ValueError("MecTermLD requires 'ckpt_dir' in user_config")
        
        agent_type = train_config.get('agent_type', 'ddpg').lower()
        
        
        load_config = train_config.copy()
        load_config.update({
            'sigma2': self.noise_power,
            'minibatch_size': 1,      
            'buffer_size': 1,         
            'random_seed': 123, 
            'noise_sigma': 0.0,      
            'actor_lr': 1e-6,         
            'critic_lr': 1e-6,       
            'tau': 0.001,             
            'gamma': 0.99,           
            'is_training': False     
        })
        
        if agent_type == 'td3':
            load_config.update({
                'policy_noise': 0.0,  
                'noise_clip': 0.0,    
                'policy_delay': 2      
            })

       
        if agent_type == 'td3':
            self.agent = TD3Agent(user_config, load_config)
            print(f"User {self.user_id}: Loading TD3 Agent from {ckpt_dir}")
        elif agent_type == 'ddpg':
            self.agent = DDPGAgent(user_config, load_config)
            print(f"User {self.user_id}: Loading DDPG Agent from {ckpt_dir}")
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")
        
        try:
            self.agent.load_checkpoint(ckpt_dir)
            print(f"Successfully loaded checkpoint from {ckpt_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {ckpt_dir}: {e}")

    def predict(self, is_random: bool) -> Tuple[np.ndarray, np.ndarray]:
        power, _ = self.agent.predict(self.state, use_noise=False)
        self.power_allocation = power.astype(np.float32)
        
        # Return zero noise vector
        return self.power_allocation, np.zeros(self.action_dim, dtype=np.float32)

    def feedback(self, sinr: float, done: bool) -> Tuple[float, ...]:
        self.sinr = float(sinr)
        
        # Process tasks
        processed_bits, offloaded_bits, arrived_bits, wasted_power = \
            self.process_tasks_and_sample_arrivals()
        
        # Compute reward
        self.reward = self.compute_reward()
        
        # Update channel and state (no agent update)
        self.update_channel_sample()
        channel_gain = np.power(np.linalg.norm(self.channel_coefficient), 2) / self.noise_power
        next_state = np.array([self.data_buffer, self.sinr, channel_gain], dtype=np.float32)
        self.state = next_state
        
        # Calculate metrics
        total_power = np.sum(self.power_allocation) - wasted_power
        is_overflow = 0
        
        return (
            self.reward,
            total_power,
            wasted_power,
            offloaded_bits,
            processed_bits,
            arrived_bits,
            self.data_buffer,
            channel_gain,
            is_overflow
        )

class MecSvrEnv:
    def __init__(self, user_list: List[MecTerm], num_antennas: int, 
                 noise_power: float, max_episode_length: int):
    
        self.user_list = user_list
        self.num_users = len(user_list)
        self.num_antennas = int(num_antennas)
        self.noise_power = float(noise_power)
        self.max_episode_length = int(max_episode_length)
        
        # Episode tracking
        self.current_step = 0
        self.episode_count = 0

    def init_target_networks(self):
        
        for user in self.user_list:
            if hasattr(user, 'agent') and hasattr(user.agent, 'init_target_network'):
                user.agent.init_target_network()

    
    
    def compute_sinr(self, channel_matrix: np.ndarray, transmit_powers: np.ndarray) -> np.ndarray:
        
        channel_pinv = np.linalg.pinv(channel_matrix)
        
        
        noise_amplification = np.power(np.linalg.norm(channel_pinv, axis=1), 2) * self.noise_power
        
        
        sinr_values = 1.0 / noise_amplification
        
        return sinr_values.astype(np.float32)


       
    def step(self, is_random: bool = True, 
             external_powers: Optional[np.ndarray] = None,
             external_noises: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
        
        # Collect channel vectors from all users
        channel_matrix = np.transpose([user.get_channel_coefficient() for user in self.user_list])
        
        # Get or collect power allocations
        if external_powers is None:
            
            powers = []
            noises = []
            for user in self.user_list:
                power, noise = user.predict(is_random)
                powers.append(power.copy())
                noises.append(noise.copy())
            powers = np.array(powers, dtype=np.float32)
            noises = np.array(noises, dtype=np.float32)
        else:
            # External mode: use provided powers (for IDE optimization)
            powers = np.asarray(external_powers, dtype=np.float32)
            
            if powers.ndim != 2 or powers.shape[0] != self.num_users:
                raise ValueError(
                    f"external_powers must have shape ({self.num_users}, action_dim). "
                    f"Got {powers.shape}"
                )
            
            # Set power for each user (important for feedback step)
            for i, user in enumerate(self.user_list):
                user.power_allocation = powers[i].astype(np.float32)
            
            # Handle noises
            if external_noises is None:
                noises = np.zeros_like(powers, dtype=np.float32)
            else:
                noises = np.asarray(external_noises, dtype=np.float32)
                if noises.shape != powers.shape:
                    raise ValueError(
                        f"external_noises must match external_powers shape. "
                        f"Got {noises.shape} vs {powers.shape}"
                    )
        
        # Calculate SINR for all users
        transmit_powers = powers[:, 0]  # First dimension is transmission power
        sinr_values = self.compute_sinr(channel_matrix, transmit_powers)
        
        # Collect feedback from all users
        self.current_step += 1
        done = self.current_step >= self.max_episode_length
        
        # Initialize metric arrays
        rewards = np.zeros(self.num_users, dtype=np.float32)
        total_powers = np.zeros(self.num_users, dtype=np.float32)
        wasted_powers = np.zeros(self.num_users, dtype=np.float32)
        offloaded_bits = np.zeros(self.num_users, dtype=np.float32)
        processed_bits = np.zeros(self.num_users, dtype=np.float32)
        arrived_bits = np.zeros(self.num_users, dtype=np.float32)
        buffer_sizes = np.zeros(self.num_users, dtype=np.float32)
        channel_gains = np.zeros(self.num_users, dtype=np.float32)
        overflows = np.zeros(self.num_users, dtype=np.float32)
        
        # Collect feedback from each user
        for i, user in enumerate(self.user_list):
            metrics = user.feedback(sinr_values[i], done)
            (rewards[i], total_powers[i], wasted_powers[i],
             offloaded_bits[i], processed_bits[i], arrived_bits[i],
             buffer_sizes[i], channel_gains[i], overflows[i]) = metrics
        
        return (
            rewards, done, total_powers, wasted_powers, noises,
            offloaded_bits, processed_bits, arrived_bits,
            buffer_sizes, channel_gains, overflows
        )

    
    
    def reset(self, is_train: bool = True) -> List[float]:
        
        self.current_step = 0
        
        if is_train:
            
            initial_buffers = [
                user.reset(user.task_arrival_rate, self.episode_count)
                for user in self.user_list
            ]
            
            
            channel_matrix = np.transpose([user.get_channel_coefficient() for user in self.user_list])
            random_powers = [
                np.random.uniform(0, user.action_bound)
                for user in self.user_list
            ]
            sinr_values = self.compute_sinr(channel_matrix, random_powers)
        else:
            
            initial_buffers = [0.0 for _ in self.user_list]
            sinr_values = [0.0 for _ in self.user_list]
        
        
        for i, user in enumerate(self.user_list):
            user.set_sinr_and_update_state(sinr_values[i])
        
        self.episode_count += 1
        return initial_buffers
    

        
class MecTermRL_IDE(MecTermRL):

    
    def __init__(self, user_config: dict, train_config: dict, ide_config: dict):
        
        super().__init__(user_config, train_config)
        
       
        if ide_config is None:
            raise ValueError("ide_config must be provided for MecTermRL_IDE")
        
        self.ide_config = ide_config
     
        self._init_ide_components()
        
        self.total_steps = 0
        self.current_episode = 0
        self.ide_application_count = 0
        self.last_ide_action = None
        self.last_base_action = None
        self.last_blend_alpha = 0.0
        
        # Log confirmation
        agent_type = train_config.get('agent_type', 'ddpg').lower()
        print(f"User {self.user_id}: Initialized {agent_type.upper()} + IDE (Deterministic)")
        print(f"  > Warmup: {self.ide_config.get('warmup_episodes', 100)} episodes")
        print(f"  > Conservative: {self.ide_config.get('conservative_interval', 30)} steps (3.3%)")
        print(f"  > Aggressive: {self.ide_config.get('aggressive_interval', 20)} steps (5%)")

    def _init_ide_components(self):
        
        opt_keys = ['pop_size', 'gen', 'pbest_rate', 'archive_size', 
                    'memory_size', 'F_init', 'CR_init']
        optimizer_config = {k: self.ide_config[k] for k in opt_keys if k in self.ide_config}
        
        self.ide_optimizer = IDEActionOptimizer(
            action_dim=self.action_dim,
            action_bound=self.action_bound,
            config=optimizer_config
        )
        
        
        self.ide_scheduler = IDEScheduler(self.ide_config)

    def _fitness_function(self, user, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return fitness_advanced(user, state, actions)

    def predict(self, is_random: bool, step_in_episode: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        
        self.total_steps += 1
        
        
        base_action, _ = self.agent.predict(self.state, use_noise=False)
        base_action = np.clip(base_action, 0.0, self.action_bound)
        self.last_base_action = base_action.copy()
        
        
        buffer_size = self.agent.replay_buffer.size()
        
        should_apply_ide = self.ide_scheduler.should_use_ide(
            episode=self.current_episode,
            step_in_episode=step_in_episode,  
            buffer_size=buffer_size
            
        )
        
        if should_apply_ide:
            try:
                ide_action = self.ide_optimizer.optimize(
                    user=self,
                    state=self.state,
                    base_action=base_action,
                    fitness_func=self._fitness_function
                )
                
                # Blend IDE action with base action
                alpha = self.ide_scheduler.get_blend_factor(self.current_episode)
                final_action = alpha * ide_action + (1.0 - alpha) * base_action
                final_action = np.clip(final_action, 0.0, self.action_bound)
                
                # Save for logging
                self.last_ide_action = ide_action.copy()
                self.last_blend_alpha = alpha
                self.ide_application_count += 1
                
            except Exception as e:
                print(f"Warning: IDE optimization failed: {e}. Using base action.")
                final_action = base_action
                self.last_ide_action = None
                self.last_blend_alpha = 0.0
        else:
            final_action = base_action
            self.last_ide_action = None
            self.last_blend_alpha = 0.0
        
        
        if self.update_actor:
            noise = self.agent.exploration_noise()
            final_action = final_action + noise
        else:
            noise = np.zeros(self.action_dim, dtype=np.float32)
        
        
        self.power_allocation = np.clip(final_action, 0.0, self.action_bound).astype(np.float32)
        
        return self.power_allocation, noise

    def feedback(self, sinr: float, done: bool) -> Tuple[float, ...]:
        
        self.sinr = float(sinr)
        
        # Process tasks
        locally_processed_bits, offloaded_bits, arrived_bits, wasted_power = \
            self.process_tasks_and_sample_arrivals()
        
        # Compute reward
        self.reward = self.compute_reward()
        
        # Update channel and state
        self.update_channel_sample()
        channel_gain = np.power(np.linalg.norm(self.channel_coefficient), 2) / self.noise_power
        next_state = np.array([self.data_buffer, self.sinr, channel_gain], dtype=np.float32)
        
      
        self.agent.update(
            state=self.state,
            action=self.power_allocation,
            reward=self.reward,
            done=done,
            next_state=next_state,
            update_actor=self.update_actor
        )
        
     
        
        self.state = next_state
        
        total_power = np.sum(self.power_allocation) - wasted_power
        is_overflow = 0
        
        return (
            self.reward,
            total_power,
            wasted_power,
            offloaded_bits,
            locally_processed_bits,
            arrived_bits,
            self.data_buffer,
            channel_gain,
            is_overflow
        )

    def reset(self, arrival_rate: float, sequence_count: int) -> float:
        
        self.current_episode = sequence_count
        initial_buffer = super().reset(arrival_rate, sequence_count)
        return initial_buffer

    def get_ide_stats(self) -> dict:
        
        stats = self.ide_scheduler.get_statistics()
        
        return {
            'total_steps': self.total_steps,
            'current_episode': self.current_episode,
            'ide_applications': self.ide_application_count,
            'ide_application_rate': self.ide_application_count / max(1, self.total_steps),
            
            
            
            # Scheduler info
            'current_interval': self.ide_scheduler.get_current_interval(self.current_episode),
            'current_blend_alpha': self.ide_scheduler.get_blend_factor(self.current_episode),
            'current_phase': stats['current_phase'],
            
            # Agent metrics (from agent directly)
            'critic_loss': self.agent.last_critic_loss if hasattr(self.agent, 'last_critic_loss') else None,
            'actor_loss': self.agent.last_actor_loss if hasattr(self.agent, 'last_actor_loss') else None,
            
            # Buffer info
            'replay_buffer_size': self.agent.replay_buffer.size(),
        }
    
    def get_last_action_info(self) -> dict:
        
        return {
            'base_action': self.last_base_action,
            'ide_action': self.last_ide_action,
            'final_action': self.power_allocation,
            'blend_alpha': self.last_blend_alpha,
            'used_ide': self.last_ide_action is not None
        }