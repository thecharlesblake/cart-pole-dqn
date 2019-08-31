from actionselector import ActionSelector
from replaymemory import ReplayMemory
from transition import Transition
import torch
import torch.nn.functional as F
import numpy as np
from myutil import plot
import matplotlib.pyplot as plt
from IPython.display import clear_output


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
            return 0
        s0, a0, r0, s1, s1_non_final_mask = self.get_transition_batches()

        # -- The Q-learning algorithm -- #
        q0 = self.q(s0, a0)

        q1_max = torch.zeros(self.batch_size, device=self.device)
        q1_max[s1_non_final_mask] = self.q_max(s1)

        loss = F.smooth_l1_loss(r0 + self.gamma * q1_max, q0)
        # --- #

        self.optimize_loss(loss)

        self.batches_processed += 1
        return loss.item()

    def optimize_loss(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def add_transition(self, *args):
        self.memory.push(*args)


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
        self.debug = False

    def train(self, n_episodes):
        episode_durations = []
        episode_losses = []

        for i_episode in range(n_episodes):  # Per-episode loop
            if self.debug:
                print("Episode: {}".format(i_episode))

            state = current_screen = self.env.reset()
            step_losses, i_step, done = [], 1, False
            while not done:  # Per-step loop
                action = self.action_selector.select_action(state)
                screen, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)

                next_state = screen - current_screen if not done else None
                self.replay_optimizer.add_transition(state, action, next_state, reward)

                loss = self.replay_optimizer.optimize()
                step_losses.append(loss)

                state = next_state
                i_step += 1

            episode_durations.append(i_step)
            episode_losses.append(np.mean(step_losses))

            if i_episode % self.hyperparameters['target_update'] == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if self.debug:
                transitions_processed = self.replay_optimizer.batches_processed * self.hyperparameters['batch_size']
                print("\tDuration: {}, Total transitions processed: {}".format(i_step, transitions_processed))
                if i_episode % 100 == 0:
                    clear_output()
                    plot(episode_durations, episode_losses)
                    plt.show()

        self.env.close()

        if self.debug:
            print('--- Training complete ---')
            plot(episode_durations, episode_losses)
            plt.show()