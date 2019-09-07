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
    "batch_size": 64,
    "gamma": 0.999,
    "eps_start": 0.9,
    "eps_end": 0.05,
    "eps_decay": 500,
    "target_update": 10,
    "replay_memory_size": 100000,
    "state_frame_count": 3
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
# extract the Y channel (luminance) from the RGB frame
# not moving image / whole screen
# scale & square image? (84x84)
# similar hyperparameter values (parameterise optimizer) - then experiment
# same CNN arch (see below)
# linear epsilon drop
# replay start size different to batch size
# try square loss
# try rmsprop vs adam
# clipping error term
# experiment again with hyperparameter values

# . The input to
# the neural network consists of an 843 843 4 image produced by the preprocessing map w. The first hidden layer
# convolves 32 filters of 8 3 8 with stride 4with the
# input image and applies a rectifier nonlinearity31,32
# . The second hidden layer convolves 64 filters of 4 3 4 with stride 2, again followed by a rectifier nonlinearity.
# Thisisfollowed by a third convolutional layerthat convolves 64 filters of 3 3 3with
# stride 1 followed by a rectifier. The final hidden layer is fully-connected and consists of 512 rectifier units. The
# output layer is a fully-connected linear layer with a single output for each valid action

# Paper things I'm not doing:
# Clipping rewards, multiple steps per action

# Benchmarks: (episode, duration, loss)
# initial-basic-algo: score=37.5, time=210s, learning=q-value plateau after 100 episodes, reward and loss all over the place
# stacked-frames-1: score=15.45, time=354s, max mem issues, overfitting, q1 going mad & actual reward dropping, some very good episodes though
# stacked-frames-2: score=61.58, time=432s, fixed max mem issues, much better! Success seems quite binary - v long or v short episodes
# 1-channel: score=48.16, time=141s, no obvious drop in quality but huge jump in speed!
