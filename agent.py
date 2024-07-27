import numpy as np
import torch
import os
from baseline.maddpg_with_two_buffer.maddpg.maddpg import MADDPG


class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    def select_action(self, o, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(self.args.low_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor_network(inputs).squeeze(0)
            u = pi.cpu().numpy()
            u = np.clip(u, self.args.low_action, self.args.high_action)
        return u.copy()

    def learn(self, transitions, other_agents):
        actor_loss,critic_loss = self.policy.train(transitions, other_agents)
        return actor_loss,critic_loss

