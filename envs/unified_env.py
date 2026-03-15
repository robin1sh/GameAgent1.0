"""
统一环境入口：make_mario_env、make_jumper_env、make_vec_env。
"""
from .mario_env import make_mario_env
from .jumper_env import make_jumper_vec_env, PROCGEN_AVAILABLE
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv


def make_unified_env(env_name: str):
    """
    创建单机环境（用于测试等）。
    env_name: "mario" | "jumper"
    """
    if env_name.lower() == "mario":
        return make_mario_env()
    if env_name.lower() == "jumper":
        if not PROCGEN_AVAILABLE:
            raise ImportError("procgen 未安装，无法使用 jumper。请运行: pip install procgen")
        venv = make_jumper_vec_env(n_envs=1)
        return venv
    raise ValueError(f"未知环境: {env_name}")


def make_vec_env(
    env_name: str,
    n_envs: int = 10,
    use_subproc: bool = True,
    frame_stack: int = 4,
    fixed_level: bool = False,
    start_level: int = 0,
    distribution_mode: str = "easy",
):
    """
    创建向量化环境，观测统一为 (64, 64, 4) channels_last。
    env_name: "mario" | "jumper" | "both"
    """
    if env_name.lower() == "mario":
        if use_subproc and n_envs > 1:
            vec = SubprocVecEnv([lambda: make_mario_env() for _ in range(n_envs)])
        else:
            vec = DummyVecEnv([lambda: make_mario_env() for _ in range(max(1, n_envs))])
        vec = VecFrameStack(vec, n_stack=frame_stack, channels_order="last")
        return vec

    if env_name.lower() == "jumper":
        if not PROCGEN_AVAILABLE:
            raise ImportError("procgen 未安装，无法使用 jumper。请运行: pip install procgen")
        vec = make_jumper_vec_env(
            n_envs=n_envs,
            fixed_level=fixed_level,
            start_level=start_level,
            distribution_mode=distribution_mode,
        )
        vec = VecFrameStack(vec, n_stack=frame_stack, channels_order="last")
        return vec

    if env_name.lower() == "both":
        from .mixed_env import make_mixed_vec_env
        return make_mixed_vec_env(n_envs=n_envs, frame_stack=frame_stack)

    raise ValueError(f"未知环境: {env_name}")


def make_jumper_env(n_envs=10):
    """兼容接口：创建 jumper VecEnv。"""
    return make_jumper_vec_env(n_envs=n_envs)
