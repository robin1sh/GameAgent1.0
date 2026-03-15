"""
运行 PPO 模型并可视化，支持 mario / jumper，使用与训练一致的预处理。
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from stable_baselines3 import PPO

from envs import make_vec_env


def resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    root_path = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(root_path):
        return root_path
    return path


def main():
    parser = argparse.ArgumentParser(description="运行 PPO 模型并可视化")
    parser.add_argument(
        "--model",
        default="best_model/best_model.zip",
        help="SB3 模型路径",
    )
    parser.add_argument(
        "--env",
        default="mario",
        choices=["mario", "jumper"],
        help="测试环境",
    )
    parser.add_argument("--steps", type=int, default=10000, help="总步数")
    parser.add_argument("--render", action="store_true", help="渲染画面")
    parser.add_argument("--deterministic", action="store_true", help="确定性动作")
    args = parser.parse_args()

    args.model = resolve_path(args.model)
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"找不到模型文件: {args.model}\n"
            "请用 --model 指定正确路径，例如：best_model/best_model.zip"
        )

    # 使用与训练一致的预处理：64x64 灰度 + 4 帧堆叠
    try:
        env_for_model = make_vec_env(args.env, n_envs=1, use_subproc=False, frame_stack=4)
    except ImportError as e:
        raise SystemExit(f"环境 {args.env} 需要额外依赖: {e}") from e

    model = PPO.load(args.model, env=env_for_model)

    obs = env_for_model.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, rewards, dones, infos = env_for_model.step(action)
        if isinstance(obs, tuple):
            obs = obs[0]
        if args.render and hasattr(env_for_model, "render"):
            env_for_model.render()
        if np.any(dones):
            obs = env_for_model.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

    env_for_model.close()


if __name__ == "__main__":
    main()
