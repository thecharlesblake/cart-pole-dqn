from actionselector import ActionSelector
from replaymemory import ReplayMemory
from transition import Transition
import torch
import torch.nn.functional as F
import numpy as np
from myutil import plot
import matplotlib.pyplot as plt
from IPython.display import clear_output
import random
from time import time


class ReplayDqnOptimizer:
    def __init__(self, policy_net, target_net, optimizer, replay_memory_size, batch_size, gamma, device):
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.memory = ReplayMemory(replay_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.batches_processed = 0
        self.debug_dqn = False

    def get_transition_batches(self):
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation)
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        return state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask

    def q(self, s, a):
        return self.policy_net(s).gather(1, a).squeeze(1)

    def q_max(self, s):
        return self.target_net(s).max(1)[0].detach()

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return 0, 0
        s0, a0, r0, s1, s1_non_final_mask = self.get_transition_batches()

        # -- The Q-learning algorithm -- #
        q0 = self.q(s0, a0)

        q1_max = torch.zeros(self.batch_size, device=self.device)
        q1_max[s1_non_final_mask] = self.q_max(s1)
        y1 = r0 + self.gamma * q1_max

        loss = F.smooth_l1_loss(y1, q0)
        # --- #

        self.optimize_loss(loss)

        if self.debug_dqn:
            self.log_batch(s0, a0, r0, s1, s1_non_final_mask, q0, q1_max, y1, loss, batch_sample_size=3)
        self.batches_processed += 1
        return loss.item(), q1_max.mean().item()

    def optimize_loss(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def add_transition(self, transition):
        self.memory.push(transition)

    def log_batch(self, s0, a0, r0, s1, s1_non_final_mask, q0, q1_max, y1, loss, batch_sample_size):
        input()
        batch_sample = random.sample(range(self.batch_size), batch_sample_size)
        _s1 = torch.zeros((self.batch_size, s1.shape[1], s1.shape[2], s1.shape[3]), device=self.device)
        _s1[s1_non_final_mask] = s1
        for i in batch_sample:
            s1_i = _s1[i] if s1_non_final_mask[i] else None
            loss_i = F.smooth_l1_loss(y1[i], q0[i])
            plot_states(s0[i].unsqueeze(0), s1_i.unsqueeze(0) if s1_i is not None else None,
                        "Batch: {},\nAction: {}\nReward: {}\nQ0: {:.2f}\nQ1_max: {:.2f}\nY1: {:.2f}\nLoss: {:.2f}"
                        .format(self.batches_processed, get_action_string(a0[i]), r0[i].item(), q0[i].item(),
                                q1_max[i].item(), y1[i].item(), loss_i.item()), offset=22)
        print("Batch: {}, Loss: {}".format(self.batches_processed, loss))


class DqnTrainer:
    def __init__(self, env, policy_net, target_net, optimizer, hyperparameters, device):
        self.env = env
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.action_selector = ActionSelector(policy_net, env.action_space.n, hyperparameters['eps_start'],
                                              hyperparameters['eps_end'], hyperparameters['eps_decay'], device)
        self.replay_optimizer = ReplayDqnOptimizer(policy_net, target_net, optimizer,
                                                   hyperparameters['replay_memory_size'], hyperparameters['batch_size'],
                                                   hyperparameters['gamma'], device)
        self.hyperparameters = hyperparameters
        self.device = device
        self.debug_episodes = False
        self.replay_full_episode = None

    def train(self, n_episodes):
        start_time = time()
        episode_durations, episode_losses, episode_rewards, episode_q_values = [], [], [], []
        for i_episode in range(n_episodes):  # Per-episode loop
            print("Episode: {}".format(i_episode), end=', ')

            state = self.env.reset()
            step_losses, step_rewards, step_q_values, i_step, done = [], [], [], 0, False
            while not done:  # Per-step loop
                action = self.action_selector.select_action(state)

                next_state, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                if done:
                    next_state = None

                transition = Transition(state, action, next_state, reward)
                self.replay_optimizer.add_transition(transition)

                loss, average_q_value = self.replay_optimizer.optimize()

                step_losses.append(loss)
                step_rewards.append(reward)
                step_q_values.append(average_q_value)

                if self.debug_episodes:
                    self.log_transition(transition)

                state = next_state
                i_step += 1

            episode_durations.append(i_step)
            episode_losses.append(np.mean(step_losses))
            episode_rewards.append(np.sum(step_rewards))
            episode_q_values.append(np.mean(step_q_values))

            if self.replay_optimizer.memory.at_capacity() and self.replay_full_episode is None:
                self.replay_full_episode = i_episode

            print("Reward: {}, Mean q1 value: {:.2f}".format(np.sum(step_rewards).item(), np.mean(step_q_values).item()))

            if i_episode % self.hyperparameters['target_update'] == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if i_episode % 12 == 11:
                clear_output()
                plot(episode_durations, episode_losses, episode_rewards, episode_q_values, self.replay_full_episode)

        self.env.close()
        print('--- Training complete in {:.2f} seconds ---'.format(time() - start_time))

        plot(episode_durations, episode_losses, episode_rewards, episode_q_values, self.replay_full_episode)

    def log_transition(self, transition):
        input()
        plot_states(transition.state, transition.next_state,
                    "Action: {}\nReward: {}".format(get_action_string(transition.action), transition.reward.item()))


def plot_states(s0, s1, text, offset=7):
    plt.figure()
    plt.subplot(3, 2, 1)
    plt.text(2, offset, text)
    plot_state(s0[:,0])
    plt.subplot(3, 2, 3)
    plot_state(s0[:,1])
    plt.subplot(3, 2, 5)
    plot_state(s0[:,2])
    if s1 is not None:
        plt.subplot(3, 2, 2)
        plot_state(s1[:,0])
        plt.subplot(3, 2, 4)
        plot_state(s1[:,1])
        plt.subplot(3, 2, 6)
        plot_state(s1[:,2])
    plt.show()

def plot_state(state):
    plt.imshow(state.cpu().squeeze(0).numpy(), interpolation='none')

def get_action_string(action):
    return 'Left' if action.item() == 0 else 'Right'
