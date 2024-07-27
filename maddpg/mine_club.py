import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_negative_expectation(q_samples, measure, average=True):
    log_2 = math.log(2.)
    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        #
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        q_samples = torch.clamp(q_samples, -1e6, 9.5)
        Eq = torch.exp(q_samples - 1.)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        assert 1 == 2

    if average:
        return Eq.mean()
    else:
        return Eq


def get_positive_expectation(p_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)  # Note JSD will be shifted
        # Ep =  - F.softplus(-p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples

    elif measure == 'RKL':

        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        assert 1 == 2

    if average:
        return Ep.mean()
    else:
        return Ep

def fenchel_dual_loss(l, m, measure=None):
    N, agent_num, units = l.size()

    l = l.view(N * agent_num, units)
    u = torch.mm(m.view(N * agent_num, units), l.t())

    u_new = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            row_start, col_start = i * 5, j * 5
            row_end, col_end = row_start + 5, col_start + 5
            u_new[i, j] = u[row_start:row_end, col_start:col_end].mean()
    u = u_new

    mask = torch.eye(N)
    n_mask = 1 - mask
    E_pos = get_positive_expectation(u, measure, average=False)
    E_neg = get_negative_expectation(u, measure, average=False)
    MI = (E_pos * mask).sum(1)
    E_pos_term = (E_pos * mask).sum(1)
    E_neg_term = (E_neg * n_mask).sum(1) / (N - 1)
    loss = E_neg_term - E_pos_term
    return loss, MI

class NEW_MINE(nn.Module):
    def __init__(self,state_size,com_a_size,measure ="JSD"):
        super(NEW_MINE, self).__init__()
        self.measure = measure
        self.com_a_size = com_a_size
        self.state_size = state_size
        self.nonlinearity = F.leaky_relu
        self.l1 = nn.Linear(self.state_size, 32)
        self.l2 = nn.Linear(self.com_a_size, 32)

    def forward(self, state, joint_action,params =None):
        em_1 = self.nonlinearity(self.l1(state),inplace=True)
        joint_action = joint_action.type(torch.FloatTensor)
        em_2 = self.nonlinearity(self.l2(joint_action),inplace=True)
        two_agent_embedding = [em_1,em_2]
        loss, MI = fenchel_dual_loss(two_agent_embedding[0], two_agent_embedding[1], measure=self.measure)
        return loss ,MI


class CLUB(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(hidden_size // 2, y_dim))
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()
        prediction_1 = mu.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()
        return  positive.sum(dim=-1) - negative.sum(dim=-1)

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
