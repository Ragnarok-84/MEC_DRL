from __future__ import annotations

import numpy as np

from helper import MarkovModel, ARModel, DDPGAgent, DDPGAgentLD


class MecTerm:
    
    def __init__(self, user_config, train_config):
        self.rate = user_config['rate']
        self.dis = user_config['dis']
        self.id = user_config['id']
        self.state_dim = int(user_config['state_dim'])
        self.action_dim = int(user_config['action_dim'])
        self.action_bound = float(user_config['action_bound'])
        self.data_buf_size = user_config['data_buf_size']
        self.t_factor = user_config['t_factor']
        self.penalty = user_config['penalty']

        self.sigma2 = float(train_config['sigma2'])
        self.init_path = ''
        self.isUpdateActor = True
        self.init_seqCnt = 0

        if 'model' not in user_config:
            self.channelModel = MarkovModel(self.dis, seed=int(train_config['random_seed']))
        else:
            n_t = 1
            n_r = int(user_config['num_r'])
            self.channelModel = ARModel(self.dis, n_t, n_r, seed=int(train_config['random_seed']))

        self.DataBuf = 0.0
        self.Channel = self.channelModel.getCh()
        self.SINR = 0.0
        self.Power = np.zeros(self.action_dim, dtype=np.float32)
        self.Reward = 0.0
        self.State = np.zeros(self.state_dim, dtype=np.float32)

        # pre-defined parameters
        self.k = 1e-27
        self.t = 0.001
        self.L = 500

    def localProc(self, p):
        return np.power(p / self.k, 1.0 / 3.0) * self.t / self.L / 1000

    def localProcRev(self, b):
        return np.power(b * 1000 * self.L / self.t, 3.0) * self.k

    def offloadRev(self, b):
        return (np.power(2.0, b) - 1) * self.sigma2 / np.power(np.linalg.norm(self.Channel), 2)

    def offloadRev2(self, b):
        return self.action_bound if self.SINR <= 1e-12 else (np.power(2.0, b) - 1) / self.SINR

    def getCh(self):
        return self.Channel

    def setSINR(self, sinr):
        self.SINR = float(sinr)
        self.sampleCh()
        channel_gain = np.power(np.linalg.norm(self.Channel), 2) / self.sigma2
        self.State = np.array([self.DataBuf, self.SINR, channel_gain], dtype=np.float32)

    def sampleData(self):
        data_t = np.log2(1 + self.Power[0] * self.SINR)
        data_p = self.localProc(self.Power[1])
        over_power = 0.0

        self.DataBuf -= data_t + data_p
        if self.DataBuf < 0:
            over_power = self.Power[1] - self.localProcRev(np.fmax(0.0, self.DataBuf + data_p))
            self.DataBuf = 0.0

        data_r = np.random.poisson(self.rate)
        self.DataBuf += data_r
        return data_t, data_p, data_r, over_power

    def sampleCh(self):
        self.Channel = self.channelModel.sampleCh()
        return self.Channel

    def reset(self, rate, seqCount):
        self.rate = rate
        self.DataBuf = np.random.randint(0, self.data_buf_size - 1) / 2.0
        self.sampleCh()

        if seqCount >= self.init_seqCnt:
            self.isUpdateActor = True

        return self.DataBuf


class MecTermLD(MecTerm):
    def __init__(self, user_config, train_config):
        super().__init__(user_config, train_config)
        ckpt_dir = user_config.get('ckpt_dir', '')
        if not ckpt_dir:
            raise ValueError("MecTermLD requires user_config['ckpt_dir'] for TF2 checkpoints.")
        self.agent = DDPGAgentLD(user_config, ckpt_dir)

    def feedback(self, sinr, done):
        isOverflow = 0
        self.SINR = float(sinr)

        data_t, data_p, data_r, over_power = self.sampleData()
        self.Reward = -self.t_factor * np.sum(self.Power) * 10 - (1 - self.t_factor) * self.DataBuf

        self.sampleCh()
        channel_gain = np.power(np.linalg.norm(self.Channel), 2) / self.sigma2
        next_state = np.array([self.DataBuf, self.SINR, channel_gain], dtype=np.float32)
        self.State = next_state

        sum_power = np.sum(self.Power) - over_power
        return self.Reward, sum_power, over_power, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow

    def predict(self, isRandom):
        self.Power = self.agent.predict(self.State).astype(np.float32)
        return self.Power, np.zeros(self.action_dim, dtype=np.float32)





class MecTermRL(MecTerm):
    
    def __init__(self, user_config, train_config):
        super().__init__(user_config, train_config)
        self.agent = DDPGAgent(user_config, train_config)

        if 'init_path' in user_config and len(user_config['init_path']) > 0:
            self.init_path = user_config['init_path']
            self.init_seqCnt = user_config.get('init_seqCnt', 0)
            self.isUpdateActor = False

    def feedback(self, sinr, done):
        isOverflow = 0
        self.SINR = float(sinr)

        data_t, data_p, data_r, over_power = self.sampleData()
        self.Reward = -self.t_factor * np.sum(self.Power) * 10 - (1 - self.t_factor) * self.DataBuf

        self.sampleCh()
        channel_gain = np.power(np.linalg.norm(self.Channel), 2) / self.sigma2
        next_state = np.array([self.DataBuf, self.SINR, channel_gain], dtype=np.float32)

        self.agent.update(self.State, self.Power, self.Reward, done, next_state, self.isUpdateActor)

        self.State = next_state
        sum_power = np.sum(self.Power) - over_power
        return self.Reward, sum_power, over_power, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow

    def predict(self, isRandom):
        power, noise = self.agent.predict(self.State, self.isUpdateActor)
        self.Power = np.fmax(0.0, np.fmin(self.action_bound, power)).astype(np.float32)
        return self.Power, noise




class MecSvrEnv:
    
    def __init__(self, user_list, num_att, sigma2, max_len):
        self.user_list = user_list
        self.num_user = len(user_list)
        self.num_att = num_att
        self.sigma2 = float(sigma2)
        self.count = 0
        self.seqCount = 0
        self.max_len = int(max_len)

    def init_target_network(self):
        for user in self.user_list:
            if hasattr(user, "agent") and hasattr(user.agent, "init_target_network"):
                user.agent.init_target_network()

    def step_transmit(self, isRandom=True):
        # channel vectors
        channels = np.transpose([user.getCh() for user in self.user_list])
        powers = []
        noises = []

        for i in range(self.num_user):
            p, n = self.user_list[i].predict(isRandom)
            powers.append(p.copy())
            noises.append(n.copy())

        powers = np.array(powers, dtype=np.float32)
        noises = np.array(noises, dtype=np.float32)

        sinr_list = self.compute_sinr(channels, powers[:, 0])

        rewards = np.zeros(self.num_user, dtype=np.float32)
        sum_powers = np.zeros(self.num_user, dtype=np.float32)
        over_powers = np.zeros(self.num_user, dtype=np.float32)
        data_ts = np.zeros(self.num_user, dtype=np.float32)
        data_ps = np.zeros(self.num_user, dtype=np.float32)
        data_rs = np.zeros(self.num_user, dtype=np.float32)
        data_buf_sizes = np.zeros(self.num_user, dtype=np.float32)
        next_channels = np.zeros(self.num_user, dtype=np.float32)
        isOverflows = np.zeros(self.num_user, dtype=np.float32)

        self.count += 1
        done = self.count >= self.max_len

        for i in range(self.num_user):
            (rewards[i], sum_powers[i], over_powers[i],
             data_ts[i], data_ps[i], data_rs[i],
             data_buf_sizes[i], next_channels[i], isOverflows[i]) = self.user_list[i].feedback(sinr_list[i], done)

        return rewards, done, sum_powers, over_powers, noises, data_ts, data_ps, data_rs, data_buf_sizes, next_channels, isOverflows

    def compute_sinr(self, channels, powers):
       
        H_inv = np.linalg.pinv(channels)
        noise = np.power(np.linalg.norm(H_inv, axis=1), 2) * self.sigma2
        sinr_list = 1.0 / noise
        return sinr_list.astype(np.float32)

    def reset(self, isTrain=True):
        self.count = 0

        if isTrain:
            init_data_buf_size = [user.reset(user.rate, self.seqCount) for user in self.user_list]
            channels = np.transpose([user.getCh() for user in self.user_list])
            powers = [np.random.uniform(0, user.action_bound) for user in self.user_list]
            sinr_list = self.compute_sinr(channels, powers)
        else:
            init_data_buf_size = [0 for _ in self.user_list]
            sinr_list = [0 for _ in self.user_list]

        for i in range(self.num_user):
            self.user_list[i].setSINR(sinr_list[i])

        self.seqCount += 1
        return init_data_buf_size
