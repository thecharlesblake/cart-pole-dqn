import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def seed_all(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env:
        env.seed(seed)


def plot(episode_durations, episode_losses, episode_rewards, episode_q_values, replay_full_episode=None, average_sz=10):
    def get_means(values):
        return torch.cat((torch.zeros(average_sz - 1), torch.tensor(values, dtype=torch.float).unfold(0, average_sz, 1)
                          .mean(1).view(-1)))

    if len(episode_durations) >= average_sz:
        reward_means = get_means(episode_rewards)
        loss_means = get_means(episode_losses)
        q_value_means = get_means(episode_q_values)

        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(reward_means.numpy(), 'b-', label='Actual reward')
        ax1.plot(q_value_means.numpy(), 'c-', label='Max action values')
        ax2.plot(loss_means.numpy(), 'y-')

        ax1.set_xlim(average_sz)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward', color='b')
        ax2.set_ylabel('Loss', color='y')

        fig.legend(loc='upper center')

        if replay_full_episode:
            plt.axvline(x=replay_full_episode, color='r')

        plt.show()
