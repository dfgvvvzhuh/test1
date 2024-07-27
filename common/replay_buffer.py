import threading
import numpy as np


class Buffer:
    def __init__(self, args):
        self.size = args.buffer_size
        self.args = args
        self.current_size = 0
        self.buffer = dict()
        for i in range(self.args.n_agents):
            self.buffer['o_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
            self.buffer['u_%d' % i] = np.empty([self.size, self.args.action_shape[i]])
            self.buffer['r_%d' % i] = np.empty([self.size])
            self.buffer['o_next_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
        self.lock = threading.Lock()
        self.point = 0
    def store_episode(self, o, u, r, o_next):
        idxs = self._get_storage_idx(inc=1) 
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]
                self.buffer['o_next_%d' % i][idxs] = o_next[i]
        self.point += 1
    
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx

class embedding_Buffer(object):
    def __init__(self, max_size=1e6):
        self.pos_storage = []
        self.pos_storage_reward = []

        self.pos_epoch_reward = []
        self.neg_storage = []
        self.neg_storage_reward = []
        self.neg_epoch_reward = []
        self.worse_storage = []
        self.worse_storage_reward = []
        self.storage = []

        self.max_max_size = max_size
        self.max_size = max_size
        self.total_max = max_size
        self.pos_ptr = 0
        self.neg_ptr = 0
        self.ptr = 0
        self.worse_ptr = 0
        self.max_reward = -100000000
        self.min_reward = 1000000000

    def rank_storage(self, current_policy_reward):

        index = np.argsort(self.pos_storage_reward)
        self.pos_storage_reward = list(np.array(self.pos_storage_reward)[index])

        pos_subset = [self.pos_storage[i] for i in index]
        self.pos_storage = pos_subset

        start_index = 0
        for i, d in enumerate(self.pos_storage_reward):
            if d > current_policy_reward:
                break;
            else:
                start_index = i + 1
        self.pos_storage_reward = self.pos_storage_reward[start_index::]
        self.pos_storage = self.pos_storage[start_index::]

        if len(self.pos_storage) == self.max_size:
            self.pos_ptr = 0
        else:
            self.pos_ptr = self.pos_ptr - start_index

        return self.pos_storage_reward[0]

    def greater(self, reward):
        if reward > self.max_reward:
            self.max_reward = reward
            return True
        else:
            return False

    def can_save(self, reward):
        if reward > self.max_reward * 0.8:
            return True
        else:
            return False

    def get_baseline(self):
        return self.max_reward * 0.8

    def get_MI(self, policy, deafult=0):
        pos_mean_mi_list = []
        neg_mena_mi_list = []
        for index in range(0, len(self.pos_storage), 100):
            temp_list = []
            temp_obs = []
            temp_state = []
            for i in range(100):
                if index + i < len(self.pos_storage):
                    temp_list.append(self.pos_storage[index + i][0])
                    temp_obs.append(self.pos_storage[index + i][1])
                    temp_state.append(self.pos_storage[index + i][2])
            mean_mi = policy.get_mi_from_a(np.array(temp_obs), np.array(temp_list), deafult)
            pos_mean_mi_list.append(mean_mi)
        for index in range(0, len(self.neg_storage), 100):
            temp_list = []
            temp_state = []
            temp_obs = []
            for i in range(100):
                if index + i < len(self.neg_storage):
                    temp_list.append(self.neg_storage[index + i][0])
                    temp_obs.append(self.neg_storage[index + i][1])
                    temp_state.append(self.neg_storage[index + i][2])
            mean_mi = policy.get_mi_from_a(np.array(temp_obs), np.array(temp_list), deafult)
            neg_mena_mi_list.append(mean_mi)
        return np.mean(pos_mean_mi_list), np.mean(neg_mena_mi_list)

    def clear_pos(self):
        self.pos_storage_reward = []
        self.pos_storage = []
        self.pos_ptr = 0

    def get_mean_pos_reward(self):
        return np.mean(self.pos_epoch_reward)

    def get_mean_neg_reward(self):
        return np.mean(self.neg_epoch_reward)

    def add_pos(self, data, rollout_reward):
        if len(self.pos_storage) == self.max_size:
            self.pos_storage_reward[int(self.pos_ptr)] = rollout_reward
            self.pos_storage[int(self.pos_ptr)] = data
            self.pos_ptr = (self.pos_ptr + 1) % self.max_size
        else:
            self.pos_storage.append(data)
            self.pos_storage_reward.append(rollout_reward)
            self.pos_ptr = (self.pos_ptr + 1) % self.max_size

    def add(self, data):
        if len(self.storage) == self.total_max:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.total_max
        else:
            self.storage.append(data)
            self.ptr = (self.ptr + 1) % self.total_max

    def add_neg(self, data, rollout_reward):
        if len(self.neg_storage) == self.max_max_size:
            self.neg_storage[int(self.neg_ptr)] = data
            self.neg_storage_reward[int(self.neg_ptr)] = rollout_reward
            self.neg_ptr = (self.neg_ptr + 1) % self.max_max_size
        else:
            self.neg_storage_reward.append(rollout_reward)
            self.neg_storage.append(data)
            self.neg_ptr = (self.neg_ptr + 1) % self.max_max_size

    def sample_pos(self, batch_size):
        ind = np.random.randint(0, len(self.pos_storage), size=min(batch_size,len(self.pos_storage)))
        actions_list, state_list, rewards_list = [], [], []
        for i in ind:
            actions, state, rewards = self.pos_storage[i]
            rewards_list.append(np.array(rewards, copy=False))
            actions_list.append(np.array(actions, copy=False))
            state_list.append(np.array(state, copy=False))
        return np.array(actions_list), np.array(state_list), np.array(rewards_list)

    def sample_neg(self, batch_size):
        ind = np.random.randint(0, len(self.neg_storage), size=batch_size)
        embedding_list, discount_reward_list, obs_list, state_list, Q_list, adv = [], [], [], [], [], []
        for i in ind:
            embedding, obs, state = self.neg_storage[i]
            discount_reward = self.neg_storage_reward[i]
            state_list.append(np.array(state, copy=False))
            embedding_list.append(np.array(embedding, copy=False))
            obs_list.append(np.array(obs, copy=False))
            discount_reward_list.append(discount_reward)
        return np.array(embedding_list), np.array(discount_reward_list), np.array(obs_list), np.array(state_list)

    def sample_rencently(self, batch_size, recent_num=10000):
        if self.ptr == 0:
            ind = np.random.randint(int(self.max_size - recent_num), int(self.max_size), size=batch_size)
        else:
            ind = np.random.randint(max(int(self.ptr - recent_num), 0), self.ptr, size=batch_size)
        embedding_list, obs_list, state_list = [], [], []
        for i in ind:
            embedding, obs, state = self.storage[i]

            state_list.append(np.array(state, copy=False))
            embedding_list.append(np.array(embedding, copy=False))
            obs_list.append(np.array(obs, copy=False))

        return np.array(embedding_list), np.array(obs_list), np.array(state_list)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        embedding_list, obs_list, state_list = [], [], []
        for i in ind:
            embedding, obs, state = self.storage[i]
            state_list.append(np.array(state, copy=False))
            embedding_list.append(np.array(embedding, copy=False))
            obs_list.append(np.array(obs, copy=False))
        return np.array(embedding_list), np.array(obs_list), np.array(state_list)

    def refresh_pos_data(self, mean_reward):
        new_pos_storage = []
        new_pos_storage_reward = []
        for index, reward in enumerate(self.pos_storage_reward):
            if reward > mean_reward:
                new_pos_storage.append(self.pos_storage[index])
                new_pos_storage_reward.append(reward)
        self.pos_storage = new_pos_storage
        self.pos_storage_reward = new_pos_storage_reward
        self.pos_ptr = len(new_pos_storage) % self.max_max_size

    def refresh_neg_data(self, mean_reward):
        new_neg_storage = []
        new_neg_storage_reward = []
        for index, reward in enumerate(self.neg_storage_reward):
            if reward < mean_reward:
                new_neg_storage.append(self.neg_storage[index])
                new_neg_storage_reward.append(reward)
        self.neg_storage = new_neg_storage
        self.neg_storage_reward = new_neg_storage_reward
        self.neg_ptr = len(new_neg_storage) % self.max_max_size
