"""
马里奥环境工厂：统一观测 84x84 灰度，4 帧堆叠由 VecFrameStack 处理。
"""
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace

from envs.skip_frame import SkipFrameWrapper
from envs.action_mapping import ActionMappingWrapper

OBS_SIZE = (84, 84)
SKIP_FRAMES = 4


def make_mario_env():
    """创建单机马里奥环境，观测 84x84 灰度，已应用动作映射。"""
    env = gym_super_mario_bros.make("SuperMarioBros-v2")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)  # 12 动作：含左+跳、左+B、蹲、上等
    env = SkipFrameWrapper(env, SKIP_FRAMES)  # 每 SKIP_FRAMES 帧执行一次动作，加速训练
    env = GrayScaleObservation(env, keep_dim=True)  # 灰度化，keep_dim=True 保留通道维便于 CNN 输入
    env = ResizeObservation(env, OBS_SIZE)  # 缩放观测到 84x84，保留更多细节

    env = ActionMappingWrapper(env, "mario")  # 将动作映射到项目统一的动作空间
    return env


def make_mario_env_for_recording():
    """
    创建用于人类数据采集的马里奥环境，不含 ActionMappingWrapper。
    返回 Mario 原生动作 (0-11)，采集时需用 MARIO_TO_UNIFIED 转为统一动作。
    不含 SkipFrame，游玩时正常帧率；保存时按每 4 帧取一帧组成 4 帧，与 PPO 一致。
    """
    env = gym_super_mario_bros.make("SuperMarioBros-v2")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, OBS_SIZE)
    return env
