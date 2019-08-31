from gym import Wrapper
from PIL import Image
import torchvision.transforms as T
import numpy as np
import torch


class PixelWrapper(Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.device = device

    def reset(self):
        super().reset()
        return self.__get_screen()

    def step(self, action):
        _, reward, done, info = super().step(action)
        return self.__get_screen(), reward, done, info

    def __get_screen(self):
        resize = T.Compose([T.ToPILImage(),
                            T.Resize(40, interpolation=Image.CUBIC),
                            T.ToTensor()])

        def get_cart_location(screen_width):
            world_width = self.env.x_threshold * 2
            scale = screen_width / world_width
            return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))

        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]

        view_width = int(screen_width * 0.6)
        cart_location = get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]

        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)

        return resize(screen).unsqueeze(0).to(self.device)
