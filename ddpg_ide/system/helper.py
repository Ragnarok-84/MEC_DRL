from __future__ import annotations

import numpy as np

# Markov channel state average gains
STATE_AVG_GAINS = np.array([
    0.031, 0.153, 0.399, 0.772, 1.274, 
    1.911, 2.694, 3.630, 4.730, 6.021, 7.902
])

# Markov channel transition probabilities [stay, lower, higher]
TRANSITION_PROBS = np.array([
    [0.514, 0.514, 1.000],
    [0.513, 0.696, 1.000],
    [0.513, 0.745, 1.000],
    [0.515, 0.776, 1.000],
    [0.513, 0.799, 1.000],
    [0.514, 0.821, 1.000],
    [0.516, 0.842, 1.000],
    [0.511, 0.858, 1.000],
    [0.516, 0.880, 1.000],
    [0.512, 0.897, 1.000],
    [0.671, 1.000, 1.000],
])

# Path loss parameters
PATH_LOSS_EXPONENT = 3.0
REFERENCE_LOSS = 0.001



class MarkovChannelModel:
    
    def __init__(self, distance: float, seed: int = 123):
        self.distance = float(distance)
        self.path_loss = REFERENCE_LOSS * np.power(1.0 / distance, PATH_LOSS_EXPONENT)
        
        np.random.seed(seed)
        self.state = np.random.randint(0, len(STATE_AVG_GAINS))

    def get_channel(self) -> np.ndarray:
        """Get current channel coefficient."""
        gain = STATE_AVG_GAINS[self.state]
        channel_coeff = np.sqrt(self.path_loss * gain)
        return np.array([channel_coeff], dtype=np.float32)

    def sample_next_channel(self) -> np.ndarray:
        rand_val = np.random.random()
        
        # Determine state transition
        if rand_val >= TRANSITION_PROBS[self.state, 1]:
            # Move to higher state
            self.state = min(self.state + 1, len(STATE_AVG_GAINS) - 1)
        elif rand_val >= TRANSITION_PROBS[self.state, 0]:
            # Move to lower state
            self.state = max(self.state - 1, 0)
        # else: stay in current state
        
        return self.get_channel()


class ARChannelModel:
      
    def __init__(self, distance: float, n_transmit: int = 1, n_receive: int = 1, 
                 rho: float = 0.95, seed: int = 123):

        self.distance = float(distance)
        self.n_transmit = int(n_transmit)
        self.n_receive = int(n_receive)
        self.path_loss = REFERENCE_LOSS * np.power(1.0 / distance, PATH_LOSS_EXPONENT)
        
        np.random.seed(seed)
        self.rho = float(rho)
        self.rho_complement_sqrt = np.sqrt(1.0 - rho * rho)
        
        self.channel_matrix = self._generate_complex_gaussian()

    def _generate_complex_gaussian(self) -> np.ndarray:
        
        real_part = np.random.normal(size=(self.n_transmit, self.n_receive)) * np.sqrt(0.5)
        imag_part = np.random.normal(size=(self.n_transmit, self.n_receive)) * np.sqrt(0.5)
        return (real_part + 1j * imag_part)[0]

    def get_channel(self) -> np.ndarray:
        
        return self.channel_matrix * np.sqrt(self.path_loss)

    def sample_next_channel(self) -> np.ndarray:
    
        noise = self._generate_complex_gaussian()
        self.channel_matrix = self.rho * self.channel_matrix + self.rho_complement_sqrt * noise
        return self.get_channel()



def create_channel_model(model_type: str, distance: float, **kwargs):
    model_type = model_type.lower()
    
    if model_type == 'markov':
        return MarkovChannelModel(distance, **kwargs)
    elif model_type == 'ar':
        return ARChannelModel(distance, **kwargs)
    else:
        raise ValueError(f"Unknown channel model type: {model_type}")