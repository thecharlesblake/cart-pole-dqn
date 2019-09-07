from gym import Wrapper
from PIL import Image
import torchvision.transforms as T
import numpy as np
import torch


class PixelWrapper(Wrapper):
    def __init__(self, env, device, n_state_frames, n_channels=1):
        super().__init__(env)
        self.device = device
        self.n_state_frames = n_state_frames
        self.n_channlels = n_channels
        self.frame_memory = None

    def reset(self):
        super().reset()
        frame = self.get_frame()
        self.frame_memory = frame.repeat(1, self.n_state_frames, 1, 1)
        return self.frame_memory

    def step(self, action):
        _, reward, done, info = super().step(action)
        frame = self.get_frame()
        self.frame_memory = self.frame_memory[:, self.n_channlels:]
        self.frame_memory = torch.cat([self.frame_memory, frame], dim=1)
        return self.frame_memory, reward, done, info

    def get_frame(self):
        resize = T.Compose([T.ToPILImage(),
                            lambda i : i.convert("L"),
                            T.Resize(84, interpolation=Image.CUBIC),
                            T.ToTensor()])

        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))

        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize and add batch dimension

        return resize(screen).unsqueeze(0).to(self.device)
