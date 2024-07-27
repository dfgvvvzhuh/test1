import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 10)
        self.fc4 = nn.Linear(10, args.action_dim)
        self.a_bound = args.action_bound

    def forward(self, x):
        net = torch.relu(self.fc1(x))
        net = torch.relu(self.fc2(net))
        net = torch.relu(self.fc3(net))
        a = torch.tanh(self.fc4(net))
        return a * self.a_bound[1]


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.obs_shape[0] + args.action_shape[0], 700)
        self.fc2 = nn.Linear(700, 300)
        self.fc3 = nn.Linear(300, 10)
        self.fc4 = nn.Linear(10, 1)

        self.n_agents = args.n_agents
        self.mlp = nn.ModuleList(
            [nn.Linear(64, 1) for _ in range(self.n_agents)])
        self.rnn = self.rnn = nn.GRU(
            input_size=300,
            num_layers=1,
            hidden_size=64,
            batch_first=True,
        )

    def forward(self, state, action):
        state = torch.stack(state)
        action = torch.stack(action)
        s_a = torch.cat((state , action), dim=2)
        net = torch.relu(self.fc1(s_a))
        net = torch.relu(self.fc2(net))
        gru_out, _ = self.rnn(net)
        local_q = torch.stack([mlp(gru_out[id, :, :])
        for id, mlp in enumerate(self.mlp)], dim=1)
        local_q = local_q.permute(1, 0, 2)
        net = torch.relu(self.fc3(net))
        q_value = self.fc4(net)
        q_value = q_value +  local_q
        return q_value