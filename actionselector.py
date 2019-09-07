import random
import math
import torch


class ActionSelector:
    def __init__(self, net, n_actions, eps_start, eps_end, eps_steps, device):
        self.net = net
        self.n_actions = n_actions
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.device = device
        self.steps_done = 0
        self.eps_threshold = eps_start

    def select_action(self, state):
        self.steps_done += 1
        sample = random.random()
        if self.eps_threshold > self.eps_end:
            self.eps_threshold = self.eps_start + (self.eps_end - self.eps_start) * self.steps_done / self.eps_steps
        if sample > self.eps_threshold:
            with torch.no_grad():
                return self.net(state).max(1)[1].view(1, 1)
        else:
            global device
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)