"""
自定义 CNN 特征提取器：适配 84x84 灰度 4 帧堆叠输入。
参考计划附录 8.2 与 NatureCNN 结构，支持 channels_first / channels_last。
"""
import torch as th
import torch.nn as nn

try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    通用 CNN 特征提取器，输入 84x84 图像（4 帧堆叠）。
    支持 (C,H,W) 或 (H,W,C) 格式，与 VecFrameStack channels_order 一致。
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # 自动判断通道顺序：若最后一维<=4则为 channels_last
        shape = observation_space.shape
        if len(shape) == 3:
            if shape[-1] in (1, 3, 4):
                n_input_channels = shape[-1]
                self._channels_last = True
            else:
                n_input_channels = shape[0]
                self._channels_last = False
        else:
            n_input_channels = shape[0]
            self._channels_last = False

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            if self._channels_last:
                sample = sample.permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        self._features_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if self._channels_last and observations.dim() == 4:
            observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))
