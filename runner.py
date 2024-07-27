from tqdm import tqdm
from agent import Agent
from baseline.maddpg_with_two_buffer.common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from baseline.maddpg_with_two_buffer.common.environment.state_normalization import StateNormalization

bad_buffer_size = 64
good_buffer_size = 64
limit_bad_buffer = 64
limit_good_buffer = 64

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.epsilon = args.epsilon
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        writer = SummaryWriter(log_dir='result', comment='mec')
        s_normal = StateNormalization()
        ep_reward_list = []
        returns = []
        MAX_EPISODES = 600
        time_step = 100

        for i in tqdm(range(MAX_EPISODES)):
            s = self.env.reset()
            ep_reward = 0
            actions_list, state_list, rewards_list = [],[],[]
            for j in range(time_step):
                u = []
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s_normal.state_normal(s[agent_id]), self.epsilon)
                        u.append(action)
                        actions.append(action)
                s_next, r, done, info , step_redo, offloading_ratio_change, reset_dist= self.env.step(actions)
                for agent in range(self.env.uav_number):
                    if step_redo[agent]:
                        continue
                    if reset_dist[agent]:
                        u[agent][2] = -1
                    if offloading_ratio_change[agent]:
                        u[agent][3] = -1
                self.buffer.store_episode(s_normal.state_normal(s[:self.args.n_agents]), u, r[:self.args.n_agents], s_normal.state_normal(s_next[:self.args.n_agents]))
                actions_list.append(u)
                state_list.append(s)
                rewards_list.append(r)

                s = s_next
                ep_reward = ep_reward + r[0]
                if self.buffer.current_size >= self.args.buffer_size:
                    transitions = self.buffer.sample(self.args.batch_size)
                    for agent in self.agents:
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        agent.learn(transitions, other_agents)

                flag = 0
                for agent in range(self.env.uav_number):
                    if done[agent]:
                        ep_reward_list = np.append(ep_reward_list, ep_reward)
                        flag = 1
                        break

                if flag == 1:
                    break
            if len(self.agents[0].policy.good_data_buffer.pos_storage_reward) != 0:
                lowest_reward = self.agents[0].policy.good_data_buffer.rank_storage(-100000)
            else:
                lowest_reward = -10000
            if lowest_reward < ep_reward:
                for two_buffer_i in range(len(actions_list)):
                    self.agents[0].policy.good_data_buffer.add_pos((actions_list[two_buffer_i], state_list[two_buffer_i], rewards_list[two_buffer_i]), ep_reward)
            else:
                for two_buffer_i in range(len(actions_list)):
                    self.agents[0].policy.bad_data_buffer.add_pos(
                        (actions_list[two_buffer_i], state_list[two_buffer_i], rewards_list[two_buffer_i]), ep_reward)
            if len(self.agents[0].policy.bad_data_buffer.pos_storage) > limit_bad_buffer:
                self.agents[0].policy.train_club(self.agents[0].policy.bad_data_buffer, 1,
                                                 batch_size=bad_buffer_size)
            if len(self.agents[0].policy.good_data_buffer.pos_storage) > limit_good_buffer:
                self.agents[0].policy.train_mine(self.agents[0].policy.good_data_buffer, 1,
                                                     batch_size=good_buffer_size)
            if  i % 50 == 0:
                for agent in self.agents:
                    agent.policy.save_model(i * 50)
        writer.close()