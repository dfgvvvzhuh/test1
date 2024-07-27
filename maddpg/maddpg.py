import torch
import os
from baseline.maddpg_with_two_buffer.maddpg.actor_critic import Actor, Critic
from baseline.maddpg_with_two_buffer.maddpg.mine_club import NEW_MINE,CLUB
from baseline.maddpg_with_two_buffer.common.replay_buffer import embedding_Buffer


club_hidden_size = 64
mine_learning_rate = 0.0001
club_learning_rate = 0.0001
MI_upper_bound_discount_rate = 0.5
MI_lower_bound_discount_rate = 0.5

class MADDPG:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.actor_network = Actor(args, agent_id)
        self.critic_network = Critic(args)
        self.actor_target_network = Actor(args, agent_id)
        self.critic_target_network = Critic(args)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        self.good_data_buffer = embedding_Buffer(args.two_buffer_size)
        self.bad_data_buffer = embedding_Buffer(args.two_buffer_size)
        self.mine_policy = NEW_MINE(self.args.obs_shape[0], self.args.action_shape[0])
        self.mine_optimizer = torch.optim.Adam([{'params': self.mine_policy.parameters()}], lr=mine_learning_rate)
        self.club_policy = CLUB(self.args.obs_shape[0], self.args.action_shape[0], club_hidden_size)
        self.club_optimizer = torch.optim.Adam([{'params': self.club_policy.parameters()}], lr=club_learning_rate)
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))

    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)
        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    def train_mine(self, replay_buffer, iterations, batch_size=64):
        pos_action, pos_state, _  = replay_buffer.sample_pos(batch_size)
        pos_action = torch.from_numpy(pos_action)
        pos_state = torch.FloatTensor(pos_state)
        pos_loaa, pos_MI = self.mine_policy(pos_state, pos_action)

        loss = pos_loaa.mean()
        self.mine_optimizer.zero_grad()
        loss.backward()
        self.mine_optimizer.step()

    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]
        o, u, o_next = [], [], []
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])
        u_temp = torch.stack(u)
        o_next_temp = torch.stack(o_next)
        o_temp = torch.stack(o)
        with torch.no_grad():
            MI_upper_bound = self.club_policy(o_temp, u_temp).detach()
        MI_upper_bound = torch.sum(MI_upper_bound, dim=0)/2
        MI_upper_bound = torch.reshape(MI_upper_bound,(self.args.batch_size, 1))

        with torch.no_grad():
            neg_MI, half_MI = self.mine_policy(o_temp, u_temp)
            neg_MI = neg_MI.detach()
            MI_lower_bound = - neg_MI
        MI_lower_bound = MI_lower_bound.unsqueeze(1)
        MI_lower_bound = torch.sum(MI_lower_bound, dim=0)/2
        new_tensor = torch.ones((self.args.batch_size, 1))
        MI_lower_bound = new_tensor * MI_lower_bound
        u_next = []
        with torch.no_grad():
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next[agent_id]))
                else:
                    u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                    index += 1
            q_next = self.critic_target_network(o_next, u_next).detach()
            temp = (r.unsqueeze(1) + self.args.gamma * q_next).detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach() - MI_upper_bound_discount_rate * MI_upper_bound + MI_lower_bound_discount_rate * MI_lower_bound

        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        actor_loss = - self.critic_network(o, u).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        self._soft_update_target_network()
        return actor_loss,critic_loss

    def train_club(self, replay_buffer, iterations, batch_size=64):
        pos_action, pos_state, _  = replay_buffer.sample_pos(batch_size)
        pos_action = torch.from_numpy(pos_action)
        pos_state = torch.FloatTensor(pos_state)
        club_loss = self.club_policy.learning_loss(pos_state, pos_action)
        self.club_optimizer.zero_grad()
        club_loss.backward(torch.ones_like(club_loss))
        self.club_optimizer.step()

    def save_model(self, train_step):
        num = str(train_step // 100)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')


