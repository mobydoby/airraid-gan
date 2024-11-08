import gym
import torch
from numpy import moveaxis
import numpy as np
from torch import nn
import cv2
from typing import List

from torch.onnx.symbolic_opset9 import detach

IMAGE_SIZE = 64
LATENT_SIZE = 4

# there are 2 models, the discriminator tries to
class Discriminator(nn.Module):
    def __init__(self, input_size = IMAGE_SIZE):
        super(Discriminator, self).__init__()
        self.pipe = nn.Sequential(
            nn.Conv2d(input_size, 16, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(8 * 8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.pipe(x)


# given random noisy inputs, try to recover/generate an atari frame.
class Generator(nn.Module):
    def __init__(self, input_size = LATENT_SIZE):
        super(Generator, self).__init__()
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(input_size, 16, kernel_size=3, stride=2, padding=1),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1),
        )
    def forward(self, x):
        return self.pipe(x)


class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ObsWrapper, self).__init__(env)
    def observation(self, obs):
        # # reshape (downsample)
        new_obs = cv2.resize(obs, (IMAGE_SIZE, IMAGE_SIZE))
        # # move color axis
        new_obs = moveaxis(new_obs, -1, 0)
        return new_obs.astype(np.uint8)


BATCH_SIZE = 64
class Batcher:
    def __init__(self, batch_size = BATCH_SIZE):
        self.batch_size = batch_size
    def create_batch(self, env) -> np.array:
        batch = []
        while len(batch) < self.batch_size:
            obs, _, done, _ = env.step(env.action_space.sample())[0]
            batch.append(obs)
            if done: env.reset()
        return np.array(batch)
    def create_latent_batch(self, env):


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    e = ObsWrapper(gym.make('AirRaid-v0'))
    print(e.reset().shape)
    print(e.step(e.action_space.sample())[0].shape)
    print(type(e.step(e.action_space.sample())[0]))
    img = e.reset()
    print(img.shape)
    print(np.sum(img))
    cv2.imshow('image', np.moveaxis(img, 0, -1))
    cv2.waitKey(0)

    # test
    gen = Generator(LATENT_SIZE).to(device)
    disc = Discriminator(IMAGE_SIZE).to(device)
    loss_fn = nn.BCELoss()

    # optimizers
    dis_optim = torch.optim.Adam(disc.parameters(), lr=1e-4)
    gen_optim = torch.optim.Adam(gen.parameters(), lr=1e-4)

    # params
    epochs = 100
    for epoch in range(epochs):
        batcher = Batcher()
        batch = batcher.create_batch(e)
        gen_out = gen(torch.stack(batch))

        # first we start with a latent space, we generate images from the latent space
        # create BATCH_SIZE number of images in the latent space.
        # optimize the generator first
        dis_optim.zero_grad()
        loss = loss_fn(generated_set, target_set)
        loss.backward()
        gen_optim.step()
        gen_optim.zero_grad()
        loss_fn.step()

        out = nn.Softmax(dim=1)(gen(e.observation_space.sample()))



        # optimize the discriminator

    epochs = 100
    while True:
        if
