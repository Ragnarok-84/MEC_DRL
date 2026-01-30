from __future__ import annotations
import numpy as np
import tensorflow as tf
from collections import deque


class IDEActionOptimizer:    
    def __init__(self, action_dim, action_bound, config=None):
        self.action_dim = action_dim
        self.action_bound = action_bound
        
        default_config = {
            'pop_size': 20,
            'gen': 5,
            'pbest_rate': 0.2,
            'archive_size': 20,
            'memory_size': 5,
            'F_init': 0.2,
            'CR_init': 1.0,
        }
        
        if config:
            default_config.update(config)
        self.config = default_config
        
        self.archive = deque(maxlen=self.config['archive_size'])
        self.operator_count = 2
        
        self.memory_F = np.full((self.operator_count, self.config['memory_size']), 
                                self.config['F_init'])
        self.memory_CR = np.full((self.operator_count, self.config['memory_size']), 
                                 self.config['CR_init'])
        
        self.operator_success = np.zeros(self.operator_count)
        self.operator_applications = np.ones(self.operator_count)
        self.operator_best = 0
        
        self.success_F = [[] for _ in range(self.operator_count)]
        self.success_CR = [[] for _ in range(self.operator_count)]
        self.success_weights = [[] for _ in range(self.operator_count)]
        
        self.generation_count = 0
    
    def generate_F_CR(self, operator_idx):
        mem_idx = np.random.randint(0, self.config['memory_size'])
        
        while True:
            F = self.memory_F[operator_idx, mem_idx] + \
                0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
            if F > 0:
                break
        F = min(F, 1.0)
        
        CR = np.random.normal(self.memory_CR[operator_idx, mem_idx], 0.1)
        CR = np.clip(CR, 0, 1)
        
        return F, CR
    
    def select_operator(self):
        r = np.random.rand()
        for op in range(self.operator_count):
            if (op / self.operator_count) <= r < ((op + 1) / self.operator_count):
                return op
        return self.operator_best
    
    def mutation_pbest1bin(self, population, pbest, F, target_idx):
        pop_size = len(population)
        r1_idx = np.random.randint(0, pop_size)
        r1 = population[r1_idx]
        
        if len(self.archive) > 0:
            p_archive = len(self.archive) / (len(self.archive) + pop_size)
            if np.random.rand() < p_archive:
                r2 = self.archive[np.random.randint(0, len(self.archive))]
            else:
                r2_idx = np.random.randint(0, pop_size)
                while r2_idx == r1_idx:
                    r2_idx = np.random.randint(0, pop_size)
                r2 = population[r2_idx]
        else:
            r2_idx = np.random.randint(0, pop_size)
            while r2_idx == r1_idx:
                r2_idx = np.random.randint(0, pop_size)
            r2 = population[r2_idx]
        
        mutant = pbest + F * (r1 - r2)
        return mutant
    
    def mutation_current_to_pbest1bin(self, population, pbest, F, target_idx):
        pop_size = len(population)
        target = population[target_idx]
        r1_idx = np.random.randint(0, pop_size)
        r1 = population[r1_idx]
        
        if len(self.archive) > 0:
            p_archive = len(self.archive) / (len(self.archive) + pop_size)
            if np.random.rand() < p_archive:
                r2 = self.archive[np.random.randint(0, len(self.archive))]
            else:
                r2_idx = np.random.randint(0, pop_size)
                while r2_idx == r1_idx:
                    r2_idx = np.random.randint(0, pop_size)
                r2 = population[r2_idx]
        else:
            r2_idx = np.random.randint(0, pop_size)
            while r2_idx == r1_idx:
                r2_idx = np.random.randint(0, pop_size)
            r2 = population[r2_idx]
        
        mutant = target + F * (pbest - target) + F * (r1 - r2)
        return mutant
    
    def crossover(self, target, mutant, CR):
        trial = np.copy(target)
        jrand = np.random.randint(0, self.action_dim)
        
        for j in range(self.action_dim):
            if np.random.rand() < CR or j == jrand:
                trial[j] = mutant[j]
        
        return trial
    
    def optimize(self, user, state, base_action, fitness_func):
        """VECTORIZED optimization - keep as is"""
        pop_size = self.config['pop_size']
        pbest_count = max(1, int(self.config['pbest_rate'] * pop_size))
        
        population = np.clip(
            base_action + np.random.uniform(-0.3, 0.3, size=(pop_size, self.action_dim)),
            0.0, self.action_bound
        )
        
        fitness = fitness_func(user, state, population)
        
        for gen in range(self.config['gen']):
            sorted_indices = np.argsort(fitness)[::-1]
            pbest_indices = sorted_indices[:pbest_count]
            
            trials = np.zeros_like(population)
            operator_indices = np.zeros(pop_size, dtype=int)
            F_values = np.zeros(pop_size)
            CR_values = np.zeros(pop_size)
            
            for i in range(pop_size):
                operator_idx = self.select_operator()
                operator_indices[i] = operator_idx
                
                F, CR = self.generate_F_CR(operator_idx)
                F_values[i] = F
                CR_values[i] = CR
                
                pbest = population[pbest_indices[np.random.randint(0, pbest_count)]]
                
                if operator_idx == 0:
                    mutant = self.mutation_pbest1bin(population, pbest, F, i)
                else:
                    mutant = self.mutation_current_to_pbest1bin(population, pbest, F, i)
                
                trial = self.crossover(population[i], mutant, CR)
                trials[i] = np.clip(trial, 0.0, self.action_bound)
            
            trials_fitness = fitness_func(user, state, trials)
            
            improved_mask = trials_fitness >= fitness
            better_mask = trials_fitness > fitness
            
            for i in range(pop_size):
                op_idx = operator_indices[i]
                self.operator_applications[op_idx] += 1
                
                if improved_mask[i]:
                    self.archive.append(np.copy(population[i]))
                    
                    if better_mask[i]:
                        improvement = trials_fitness[i] - fitness[i]
                        self.operator_success[op_idx] += 1
                        self.success_F[op_idx].append(F_values[i])
                        self.success_CR[op_idx].append(CR_values[i])
                        self.success_weights[op_idx].append(improvement)
            
            population = np.where(improved_mask[:, None], trials, population)
            fitness = np.where(improved_mask, trials_fitness, fitness)
        
        self._update_memory()
        
        self.generation_count += 1
        if self.generation_count % 10 == 0:
            self._update_best_operator()
        
        best_idx = np.argmax(fitness)
        return np.clip(population[best_idx], 0.0, self.action_bound)
    
    def _update_memory(self):
        pos = self.generation_count % self.config['memory_size']
        
        for op in range(self.operator_count):
            if len(self.success_F[op]) > 0:
                weights = np.array(self.success_weights[op])
                weights = weights / np.sum(weights)
                
                F_arr = np.array(self.success_F[op])
                CR_arr = np.array(self.success_CR[op])
                
                self.memory_F[op, pos] = np.sum(weights * F_arr**2) / np.sum(weights * F_arr)
                
                denominator_CR = np.sum(weights * CR_arr)
                if denominator_CR > 0:
                    self.memory_CR[op, pos] = np.sum(weights * CR_arr**2) / denominator_CR
                else:
                    self.memory_CR[op, pos] = 0
                
                self.success_F[op] = []
                self.success_CR[op] = []
                self.success_weights[op] = []
    
    def _update_best_operator(self):
        success_rates = self.operator_success / self.operator_applications
        self.operator_best = np.argmax(success_rates)
        self.operator_success = np.zeros(self.operator_count)
        self.operator_applications = np.ones(self.operator_count)



class IDEScheduler:    
    def __init__(self, config):
        
        self.warmup_episodes = config.get('warmup_episodes', 100)
        
        # Progressive schedule
        self.phase_transition = config.get('phase_transition', 500)
        self.conservative_interval = config.get('conservative_interval', 30)  
        self.aggressive_interval = config.get('aggressive_interval', 20)    
        
        # Blend alpha schedule
        self.decay_episodes = config.get('decay_episodes', 1000)
        
        # Safety
        self.min_buffer_size = config.get('min_buffer_size', 20000)
        
        # Statistics
        self.total_checks = 0
        self.ide_calls = 0
        self.current_phase = "warmup"
    
    def should_use_ide(self, episode, step_in_episode, buffer_size=None):
        self.total_checks += 1
        
        # Phase 1: Warmup
        if episode < self.warmup_episodes:
            self.current_phase = "warmup"
            return False
        
        # Safety: Buffer size
        if buffer_size is not None and buffer_size < self.min_buffer_size:
            return False
        
        # Determine interval based on phase
        if episode < self.phase_transition:
            self.current_phase = "conservative"
            interval = self.conservative_interval
        else:
            self.current_phase = "aggressive"
            interval = self.aggressive_interval
        
        should_run = (step_in_episode % interval == 0)
        
        if should_run:
            self.ide_calls += 1
        
        return should_run
    
    def get_blend_factor(self, episode):
        
        if episode < self.warmup_episodes:
            return 0.0
        
        if episode < self.phase_transition:
            progress = (episode - self.warmup_episodes) / (self.phase_transition - self.warmup_episodes)
            return 0.1 + progress * 0.2
        else:
            progress = min(1.0, (episode - self.phase_transition) / self.decay_episodes)
            return 0.3 + progress * 0.2
    
    def get_current_interval(self, episode):
        
        if episode < self.warmup_episodes:
            return -1
        elif episode < self.phase_transition:
            return self.conservative_interval
        else:
            return self.aggressive_interval
    
    def get_statistics(self):
        
        usage_rate = (self.ide_calls / self.total_checks * 100) if self.total_checks > 0 else 0.0
        return {
            'total_checks': self.total_checks,
            'ide_calls': self.ide_calls,
            'usage_rate': usage_rate,
            'current_phase': self.current_phase
        }




def fitness_advanced(user, state, actions):
    
    if len(actions.shape) == 1:
        actions = actions.reshape(1, -1)
    
    pop_size = len(actions)
    
    s = np.tile(np.reshape(state, (1, -1)), (pop_size, 1)).astype(np.float32)
    a = actions.astype(np.float32)
    
    s_tf = tf.convert_to_tensor(s)
    a_tf = tf.convert_to_tensor(a)
    
    agent = user.agent
    
    if hasattr(agent, 'critic1') and hasattr(agent, 'critic2'):
        q1 = agent.critic1(s_tf, a_tf, training=False)
        q2 = agent.critic2(s_tf, a_tf, training=False)
        q_vals = tf.minimum(q1, q2).numpy()
    elif hasattr(agent, 'critic'):
        q_vals = agent.critic(s_tf, a_tf, training=False).numpy()
    else:
        raise AttributeError("Agent must have critic(s)")
    
    q_vals = np.reshape(q_vals, (-1,))
    return q_vals


