import gym
import torch
from torch.optim.rmsprop import RMSprop
from importlib import reload

import evaluator
import myutil
import model
import pixelwrapper
import training

# %%
evaluator = reload(evaluator)
myutil = reload(myutil)
model = reload(model)
pixelwrapper = reload(pixelwrapper)
training = reload(training)

# %% Setup
hyperparameters = {
    "batch_size": 32,
    "gamma": 0.99,
    "eps_start": 0.9,
    "eps_end": 0.05,
    "eps_steps": 300,
    "target_update": 500,
    "update_frequency": 1,
    "replay_memory_size": 100000,
    "state_frame_count": 3,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = pixelwrapper.PixelWrapper(gym.make('CartPole-v0'), device, hyperparameters["state_frame_count"])
myutil.seed_all(0, env)

init_obs = env.reset()
_, _, h, w = init_obs.shape
n_actions = env.action_space.n

policy_net = model.CNN(hyperparameters["state_frame_count"], h, w, n_actions).to(device)
target_net = model.CNN(hyperparameters["state_frame_count"], h, w, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = RMSprop(policy_net.parameters())

trainer = training.DqnTrainer(env, policy_net, target_net, optimizer, hyperparameters, device)

# %% Run the training loop
n_episodes = 400
trainer.train(n_episodes)

evaluator = evaluator.Evaluator(env, policy_net, device)
evaluator.run()

# TODO:
# replay start size different to batch size
# clipping error term (plus gradients?)
# no batch norm
# experiment again with hyperparameter values

# Paper things I'm not doing:
# Clipping rewards, multiple steps per action

# Benchmarks:
# initial-basic-algo: score=37.5, time=210s, learning=q-value plateau after 100 episodes, reward and loss all over the place
# stacked-frames-1: score=15.45, time=354s, max mem issues, overfitting, q1 going mad & actual reward dropping, some very good episodes though
# stacked-frames-2: score=61.58, time=432s, fixed max mem issues, much better! Success seems quite binary - v long or v short episodes
# 1-channel: score=48.16, time=141s, no obvious drop in quality but huge jump in speed!
# whole-frame: score=48.29, time=316s, time inc but still about as good as 3-channel, bad eps gone (presumably because no centring), q dipping after a while (over-training??)