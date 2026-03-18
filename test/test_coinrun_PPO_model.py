"""
专门用于测试 CoinRun 的 PPO 模型（Stable-Baselines3 .zip）。
支持可视化（通过 info["rgb"] + pygame），避免依赖 env.render()。
"""
import argparse
import os
import sys

import numpy as np
import pygame
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs import make_vec_env


def resolve_path(path: str) -> str:
    """若传入路径不存在，则尝试在项目根目录下查找。"""
    if os.path.exists(path):
        return path
    root_path = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(root_path):
        return root_path
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="测试 CoinRun PPO 模型并可视化")
    parser.add_argument(
        "--model",
        default="best_model/coinrun/ppo_coinrun_final.zip",
        help="SB3 模型路径",
    )
    parser.add_argument("--steps", type=int, default=10000, help="总步数")
    parser.add_argument("--deterministic", action="store_true", help="确定性动作")
    parser.add_argument("--render", action="store_true", help="是否显示 pygame 画面")
    parser.add_argument("--render-scale", type=int, default=6, help="画面放大倍数")
    parser.add_argument(
        "--fixed-level",
        action="store_true",
        default=True,
        help="使用固定关卡（默认开启，便于复现）",
    )
    parser.add_argument(
        "--no-fixed-level",
        action="store_false",
        dest="fixed_level",
        help="关闭固定关卡，使用随机关卡",
    )
    parser.add_argument("--start-level", type=int, default=0, help="固定关卡 ID（第一关为 0）")
    parser.add_argument("--distribution-mode", default="easy", help="Procgen 难度分布，建议 easy")
    return parser.parse_args()


def main():
    args = parse_args()
    args.model = resolve_path(args.model)
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"找不到模型文件: {args.model}\n"
            "请用 --model 指定正确路径，例如：best_model/coinrun/<exp-id>/ppo_coinrun_final.zip"
        )

    # 与训练保持一致：CoinRun + frame_stack=4
    vec_env = make_vec_env(
        env_name="coinrun",
        n_envs=1,
        use_subproc=False,
        frame_stack=4,
        fixed_level=args.fixed_level,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode,
    )
    model = PPO.load(args.model, env=vec_env)

    screen = None
    screen_size = (64 * args.render_scale, 64 * args.render_scale)
    if args.render:
        pygame.init()
        screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("CoinRun PPO 预览")

    obs = vec_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    running = True
    for _ in range(args.steps):
        if not running:
            break

        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, _, dones, infos = vec_env.step(action)
        if isinstance(obs, tuple):
            obs = obs[0]

        if args.render and screen is not None:
            rgb = infos[0].get("rgb") if infos and isinstance(infos[0], dict) else None
            if rgb is not None:
                frame = np.transpose(rgb, (1, 0, 2))
                surface = pygame.surfarray.make_surface(frame)
                surface = pygame.transform.scale(surface, screen_size)
                screen.blit(surface, (0, 0))
                pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        if np.any(dones):
            obs = vec_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

    vec_env.close()
    if screen is not None:
        pygame.quit()


if __name__ == "__main__":
    main()
