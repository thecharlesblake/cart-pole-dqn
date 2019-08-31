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


def plot(episode_durations, episode_losses):
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    loss_t = torch.tensor(episode_losses, dtype=torch.float)

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(durations_t.numpy(), 'g-')
    ax2.plot(loss_t.numpy(), 'b-')

    ax1.set_xlabel('Episode / Batch')
    ax1.set_ylabel('Steps', color='g')
    ax2.set_ylabel('Loss', color='b')
    # Take 100 episode averages and plot them too
    # if len(durations_t) >= 100:
    #    means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #    means = torch.cat((torch.zeros(99), means))
    #    plt.plot(means.numpy())
