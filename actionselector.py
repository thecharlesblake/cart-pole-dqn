import random
import math
import torch


class ActionSelector:
    def __init__(self, net, n_actions, eps_start, eps_end, eps_decay, device):
        self.net = net
        self.n_actions = n_actions
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.device = device
        self.steps_done = 0

    def select_action(self, state):
        self.steps_done += 1
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                return self.net(state).max(1)[1].view(1, 1)
        else:
            global device
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)