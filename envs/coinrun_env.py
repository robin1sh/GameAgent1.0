"""
Procgen CoinRun 环境入口：复用 Jumper 的 Procgen 观测预处理包装器。
CoinRun 与 Jumper 同属 Procgen，统一采用 84x84 灰度 + 跳帧接口。
"""
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

from .jumper_env import (
    OBS_SIZE,
    SKIP_FRAMES,
    PROCGEN_AVAILABLE,
    ProcgenEnv,
    ProcgenObsWrapper,
)


class CoinRunAlignedRewardWrapper(VecEnvWrapper):
    """CoinRun 单任务统一奖励语义包装器。"""

    def __init__(
        self,
        venv,
        progress_coef=0.02,
        success_bonus=100.0,
        fail_penalty=20.0,
        step_penalty=0.002,
    ):
        super().__init__(venv)
        self.progress_coef = float(progress_coef)
        self.success_bonus = float(success_bonus)
        self.fail_penalty = float(fail_penalty)
        self.step_penalty = float(step_penalty)

    @staticmethod
    def _coinrun_success(info):
        return bool(
            info.get("level_complete", False)
            or info.get("prev_level_complete", False)
            or info.get("carrot_get", False)
        )

    def _aligned_reward(self, raw_reward, done, success):
        """统一奖励模板：progress + success - fail - step。"""
        progress = max(0.0, float(raw_reward)) * self.progress_coef
        shaped = progress - self.step_penalty
        if success:
            shaped += self.success_bonus
        elif bool(done):
            shaped -= self.fail_penalty
        return max(0.0, shaped)

    def reset(self):
        """透传 reset，满足 VecEnvWrapper 抽象接口。"""
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        aligned_rewards = np.zeros_like(rewards, dtype=np.float32)
        for i in range(len(rewards)):
            info = infos[i] if isinstance(infos[i], dict) else {}
            success = self._coinrun_success(info)
            aligned_rewards[i] = self._aligned_reward(
                raw_reward=rewards[i],
                done=dones[i],
                success=success,
            )
            info["use_aligned_reward"] = True
            info["progress_coef"] = self.progress_coef
            info["success_bonus"] = self.success_bonus
            info["fail_penalty"] = self.fail_penalty
            info["step_penalty"] = self.step_penalty
            info["game"] = "coinrun"
        return obs, aligned_rewards, dones, infos


def make_coinrun_vec_env(
    n_envs=10,
    fixed_level=False,
    start_level=0,
    num_levels=None,
    distribution_mode="easy",
    skip_frames=SKIP_FRAMES,
    max_episode_steps=3000,
    use_aligned_reward=False,
    progress_coef=0.02,
    success_bonus=100.0,
    fail_penalty=20.0,
    step_penalty=0.002,
    **procgen_kwargs,
):
    """
    创建 Procgen CoinRun 的 VecEnv，观测已统一为 84x84 灰度。
    默认每个动作重复执行 4 帧，与 Mario 训练节奏保持一致。

    :param n_envs: 并行环境数
    :param fixed_level: 若为 True，使用固定关卡（num_levels=1），不随机关卡
    :param start_level: 固定关卡 ID（fixed_level=True 时生效）
    :param num_levels: 可选关卡数量（fixed_level=False 时可传）
    :param distribution_mode: Procgen 难度分布（默认 easy）
    :param skip_frames: 动作重复执行帧数，默认 4
    :param procgen_kwargs: 其他 ProcgenEnv 参数，如 rand_seed 等
    """
    if not PROCGEN_AVAILABLE:
        raise ImportError("procgen 未安装，无法使用 coinrun 环境。请运行: pip install procgen")
    kwargs = dict(procgen_kwargs)
    kwargs.setdefault("distribution_mode", distribution_mode)
    # max_episode_steps 不透传给 ProcgenEnv（部分版本不支持 max_steps 参数）
    if fixed_level:
        kwargs["start_level"] = int(start_level)
        kwargs["num_levels"] = 1
    elif num_levels is not None:
        kwargs["num_levels"] = int(num_levels)
    venv = ProcgenEnv(num_envs=n_envs, env_name="coinrun", **kwargs)
    venv = ProcgenObsWrapper(venv, obs_size=OBS_SIZE, skip_frames=skip_frames)
    if use_aligned_reward:
        venv = CoinRunAlignedRewardWrapper(
            venv,
            progress_coef=progress_coef,
            success_bonus=success_bonus,
            fail_penalty=fail_penalty,
            step_penalty=step_penalty,
        )
    return venv
