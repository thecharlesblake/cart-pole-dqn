from dataclasses import dataclass

import torch
from gym import Env
from torch import nn


@dataclass
class Evaluator:
    env: Env
    net: nn.Module
    device: torch.device

    def single_attempt(self):
        with torch.no_grad():
            state = self.env.reset()
            cum_reward, done = 0, False
            while not done:
                action = self.net(state).max(1)[1].view(1, 1)
                state, reward, done, _ = self.env.step(action.item())
                cum_reward += reward
                if done:
                    break
        return cum_reward

    def run(self, trials=100, ave_reward_threshold=195.0):
        cum_reward = 0
        for t in range(trials):
            reward = self.single_attempt()
            cum_reward += reward
            print("Attempt: {}, Reward: {}".format(t, reward))
        ave_reward = cum_reward / trials
        print("Average reward: {}, Target: {}".format(ave_reward, ave_reward_threshold))
        if ave_reward > ave_reward_threshold:
            print("Success!")
        else:
            print("Failure :(")