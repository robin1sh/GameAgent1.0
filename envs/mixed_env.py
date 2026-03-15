"""
联合训练环境：混合 Mario 与 Jumper，用于实验组 B。
"""
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, DummyVecEnv, VecFrameStack

from .mario_env import make_mario_env
from .jumper_env import make_jumper_vec_env, PROCGEN_AVAILABLE


class MixedVecEnv(VecEnv):
    """
    将 mario 与 jumper 的 VecEnv 拼接为一个 VecEnv。
    前 n_mario 个 env 为 mario，后 n_jumper 个为 jumper。
    """

    def __init__(self, mario_vec, jumper_vec):
        n_mario = mario_vec.num_envs
        n_jumper = jumper_vec.num_envs
        n_total = n_mario + n_jumper
        obs_space = mario_vec.observation_space
        act_space = mario_vec.action_space
        super().__init__(n_total, obs_space, act_space)
        self.mario_vec = mario_vec
        self.jumper_vec = jumper_vec
        self.n_mario = n_mario
        self.n_jumper = n_jumper

    def reset(self):
        mario_obs = self.mario_vec.reset()
        jumper_obs = self.jumper_vec.reset()
        if isinstance(mario_obs, tuple):
            mario_obs = mario_obs[0]
        if isinstance(jumper_obs, tuple):
            jumper_obs = jumper_obs[0]
        return np.concatenate([mario_obs, jumper_obs], axis=0)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        mario_actions = self._actions[: self.n_mario]
        jumper_actions = self._actions[self.n_mario :]
        mario_result = self.mario_vec.step(mario_actions)
        jumper_result = self.jumper_vec.step(jumper_actions)
        mario_obs, mario_rewards, mario_dones, mario_infos = mario_result[:4]
        jumper_obs, jumper_rewards, jumper_dones, jumper_infos = jumper_result[:4]
        if isinstance(mario_obs, tuple):
            mario_obs = mario_obs[0]
        if isinstance(jumper_obs, tuple):
            jumper_obs = jumper_obs[0]
        obs = np.concatenate([mario_obs, jumper_obs], axis=0)
        rewards = np.concatenate([mario_rewards, jumper_rewards])
        dones = np.concatenate([mario_dones, jumper_dones])
        infos = mario_infos + jumper_infos
        return obs, rewards, dones, infos

    def close(self):
        self.mario_vec.close()
        self.jumper_vec.close()

    def get_attr(self, attr_name, indices=None):
        if indices is None:
            indices = list(range(self.num_envs))
        results = []
        for i in indices:
            if i < self.n_mario:
                try:
                    r = self.mario_vec.get_attr(attr_name, [i])
                    results.append(r[0] if r else None)
                except Exception:
                    results.append(None)
            else:
                try:
                    r = self.jumper_vec.get_attr(attr_name, [i - self.n_mario])
                    results.append(r[0] if r else None)
                except Exception:
                    results.append(None)
        return results

    def set_attr(self, attr_name, value, indices=None):
        if indices is None:
            indices = list(range(self.num_envs))
        for i in indices:
            if i < self.n_mario:
                self.mario_vec.set_attr(attr_name, value, [i])
            else:
                self.jumper_vec.set_attr(attr_name, value, [i - self.n_mario])

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if indices is None:
            indices = list(range(self.num_envs))
        results = []
        for i in indices:
            if i < self.n_mario:
                try:
                    r = self.mario_vec.env_method(method_name, *args, indices=[i], **kwargs)
                    results.append(r[0] if r else None)
                except Exception:
                    results.append(None)
            else:
                try:
                    r = self.jumper_vec.env_method(method_name, *args, indices=[i - self.n_mario], **kwargs)
                    results.append(r[0] if r else None)
                except Exception:
                    results.append(None)
        return results

    def env_is_wrapped(self, wrapper_class, indices=None):
        if indices is None:
            indices = list(range(self.num_envs))
        return [False] * len(indices)


def make_mixed_vec_env(n_envs=10, frame_stack=4, mario_ratio=0.5):
    """
    创建混合 VecEnv：mario_ratio 比例为 mario，其余为 jumper。
    """
    if not PROCGEN_AVAILABLE:
        raise ImportError("procgen 未安装，无法使用混合环境。")
    n_mario = max(1, int(n_envs * mario_ratio))
    n_jumper = max(1, n_envs - n_mario)
    mario_vec = SubprocVecEnv([lambda: make_mario_env() for _ in range(n_mario)])
    jumper_vec = make_jumper_vec_env(n_envs=n_jumper)
    mario_vec = VecFrameStack(mario_vec, n_stack=frame_stack, channels_order="last")
    jumper_vec = VecFrameStack(jumper_vec, n_stack=frame_stack, channels_order="last")
    return MixedVecEnv(mario_vec, jumper_vec)
