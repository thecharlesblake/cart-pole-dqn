import gym
import torch
from torch.optim.rmsprop import RMSprop

from evaluator import Evaluator
from myutil import seed_all
from model import CNN
from pixelwrapper import PixelWrapper
from training import DqnTrainer


# %% General setup
colab = False
debug = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = PixelWrapper(gym.make('CartPole-v0'), device)
seed_all(0, env)

# %% DQN setup
hyperparameters = {
    "batch_size": 128,
    "gamma": 0.999,
    "eps_start": 0.9,
    "eps_end": 0.05,
    "eps_decay": 500,
    "target_update": 10,
    "replay_memory_size": 10000
}

init_obs = env.reset()
_, _, h, w = init_obs.shape
n_actions = env.action_space.n

policy_net = CNN(h, w, n_actions).to(device)
target_net = CNN(h, w, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = RMSprop(policy_net.parameters())

trainer = DqnTrainer(env, policy_net, target_net, optimizer, hyperparameters, device)

# %% Run the training loop
n_episodes = 40
trainer.train(n_episodes)

evaluator = Evaluator(env, policy_net, device)
evaluator.run()

# TODO:
#
# Main aims:
# a) verify it's doing what I expect it to!
# b) improve the performance to get it to pass the test!
# c) shift the architecture to make it close to the Atari paper

# (a):
# 1) GIT! :)
# 2) Understand the loss fn :)
# 3) Debug statements in relevant places so that I can run through the algo step by step for multiple episodes :)
# 4) Figure out what I actually want to plot :)
# 6) Make sure that generally all parts of the program are doing what I expect and the debug mode demonstrates this :)

# (b): some ideas
# 1) Get it easily working and transferable to colab
# 2) Record some initial benchmarks (and keep testing against them!!)
# 3) Get system ready where experimental changes can be toggled while implementing
# luminescense thing
# 3/4 frames stacked
# square image
# not moving image
# other things to align with atari paper
