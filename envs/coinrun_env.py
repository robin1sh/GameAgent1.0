"""
Procgen CoinRun 环境入口：复用 Jumper 的 Procgen 观测预处理包装器。
CoinRun 与 Jumper 同属 Procgen，统一采用 84x84 灰度 + 跳帧接口。
"""
from .jumper_env import (
    OBS_SIZE,
    SKIP_FRAMES,
    PROCGEN_AVAILABLE,
    ProcgenEnv,
    ProcgenObsWrapper,
)


def make_coinrun_vec_env(
    n_envs=10,
    fixed_level=False,
    start_level=0,
    num_levels=None,
    distribution_mode="easy",
    skip_frames=SKIP_FRAMES,
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
    if fixed_level:
        kwargs["start_level"] = int(start_level)
        kwargs["num_levels"] = 1
    elif num_levels is not None:
        kwargs["num_levels"] = int(num_levels)
    venv = ProcgenEnv(num_envs=n_envs, env_name="coinrun", **kwargs)
    venv = ProcgenObsWrapper(venv, obs_size=OBS_SIZE, skip_frames=skip_frames)
    return venv
