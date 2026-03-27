"""
测试混合训练（train_unified.py）保存的 PPO 模型。

与训练一致：VecMonitor + VecTransposeImage +（可选）VecNormalize，
双动作头模型需加 --dual-head（与 train_unified --dual-head 一致）；仅 Mario / 仅 CoinRun 时
在双头模式下会注入 game_id 像素（与 mixed_env 一致）。默认按单头 CnnPolicy 加载。
"""
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage, VecNormalize

from envs import make_vec_env
from policies import UnifiedDualHeadPolicy


def resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    root_path = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(root_path):
        return root_path
    return path


def _inject_game_id_chw(obs: np.ndarray, coinrun: bool) -> None:
    """
    在 CHW 观测上写入 game_id，与 mixed_env._inject_game_id_channel 语义一致。
    浮点且值域在 [0,1] 附近时用 0 / 1；否则用 0 / 255。
    """
    if obs.dtype.kind == "f" and float(np.max(obs)) <= 1.0 + 1e-5:
        obs[:, 0, 0, 0] = 1.0 if coinrun else 0.0
    else:
        obs[:, 0, 0, 0] = 255 if coinrun else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="测试 train_unified 混合训练模型")
    p.add_argument(
        "--model",
        required=True,
        help="PPO .zip 路径，例如 best_model/unified_from_fused_v1/best_model.zip",
    )
    p.add_argument(
        "--vecnorm",
        default=None,
        help="vecnormalize.pkl；默认与模型同目录下的 vecnormalize.pkl",
    )
    p.add_argument(
        "--no-vecnorm",
        action="store_true",
        help="不使用 VecNormalize（对应训练时 --no-normalize-reward）",
    )
    p.add_argument(
        "--env",
        default="both",
        choices=["both", "mario", "coinrun"],
        help="评测环境：both=混合（与训练一致，自动注入 game_id）；mario/coinrun=单任务",
    )
    p.add_argument(
        "--dual-head",
        action="store_true",
        help="与 train_unified --dual-head 一致：用 UnifiedDualHeadPolicy 加载；"
        "默认关闭（单头 CnnPolicy），与训练脚本默认一致",
    )
    p.add_argument("--n-envs", type=int, default=2, help="both 模式下的并行数（与训练一致可调大）")
    p.add_argument("--mario-ratio", type=float, default=0.5, help="both：Mario 子环境占比")
    p.add_argument("--steps", type=int, default=10000, help="总环境步数（所有子环境合计步进次数）")
    p.add_argument("--deterministic", action="store_true", help="确定性策略")
    p.add_argument("--render", action="store_true", help="可视化（CoinRun 用 info['rgb']+pygame；Mario 尝试 render）")
    p.add_argument("--render-scale", type=int, default=6, help="CoinRun pygame 放大倍数")
    p.add_argument("--fps", type=int, default=12, help="渲染帧率上限")
    p.add_argument("--fixed-level", action="store_true", default=True)
    p.add_argument("--no-fixed-level", action="store_false", dest="fixed_level")
    p.add_argument("--start-level", type=int, default=0)
    p.add_argument("--distribution-mode", default="easy")
    p.add_argument("--max-episode-steps", type=int, default=3000)
    p.add_argument("--coinrun-reward-scale", type=float, default=200.0)
    p.add_argument("--coinrun-progress-coef", type=float, default=0.05)
    p.add_argument("--coinrun-success-bonus", type=float, default=10.0)
    p.add_argument("--coinrun-fail-penalty", type=float, default=2.0)
    p.add_argument("--coinrun-step-penalty", type=float, default=0.002)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.model = resolve_path(args.model)
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"找不到模型: {args.model}")

    vecnorm_path = args.vecnorm
    if vecnorm_path is None:
        candidate = os.path.join(os.path.dirname(os.path.abspath(args.model)), "vecnormalize.pkl")
        vecnorm_path = candidate if os.path.isfile(candidate) else None

    if args.env == "both":
        vec_env = make_vec_env(
            "both",
            n_envs=args.n_envs,
            use_subproc=False,
            frame_stack=4,
            mario_ratio=args.mario_ratio,
            coinrun_reward_scale=args.coinrun_reward_scale,
            coinrun_progress_coef=args.coinrun_progress_coef,
            coinrun_success_bonus=args.coinrun_success_bonus,
            coinrun_fail_penalty=args.coinrun_fail_penalty,
            coinrun_step_penalty=args.coinrun_step_penalty,
            fixed_level=args.fixed_level,
            start_level=args.start_level,
            distribution_mode=args.distribution_mode,
            max_episode_steps=args.max_episode_steps,
        )
    elif args.env == "mario":
        vec_env = make_vec_env(
            "mario",
            n_envs=1,
            use_subproc=False,
            frame_stack=4,
            max_episode_steps=args.max_episode_steps,
        )
    else:
        vec_env = make_vec_env(
            "coinrun",
            n_envs=1,
            use_subproc=False,
            frame_stack=4,
            coinrun_reward_scale=args.coinrun_reward_scale,
            coinrun_progress_coef=args.coinrun_progress_coef,
            coinrun_success_bonus=args.coinrun_success_bonus,
            coinrun_fail_penalty=args.coinrun_fail_penalty,
            coinrun_step_penalty=args.coinrun_step_penalty,
            fixed_level=args.fixed_level,
            start_level=args.start_level,
            distribution_mode=args.distribution_mode,
            max_episode_steps=args.max_episode_steps,
        )

    vec_env = VecMonitor(VecTransposeImage(vec_env))

    if not args.no_vecnorm:
        if vecnorm_path is None or not os.path.isfile(vecnorm_path):
            raise FileNotFoundError(
                "需要 vecnormalize.pkl（与 train_unified 默认 --normalize-reward 一致）。"
                "请放在模型同目录或显式传入 --vecnorm，若训练未使用归一化则加 --no-vecnorm。"
            )
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    load_kw = {}
    if args.dual_head:
        load_kw["custom_objects"] = {"policy_class": UnifiedDualHeadPolicy}

    try:
        model = PPO.load(args.model, env=vec_env, **load_kw)
    except RuntimeError as e:
        err = str(e)
        if args.dual_head and "mario_action_net" in err:
            raise RuntimeError(
                "按双头策略加载失败：该 checkpoint 很可能是单头 CnnPolicy（训练时未加 --dual-head）。"
                "请去掉 --dual-head 后重试。"
            ) from e
        raise

    screen = None
    clock = None
    screen_size = (64 * args.render_scale, 64 * args.render_scale)
    if args.render and args.env == "coinrun":
        pygame.init()
        screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("Unified PPO — CoinRun")
        clock = pygame.time.Clock()

    obs = vec_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    if args.env == "mario" and args.dual_head:
        _inject_game_id_chw(obs, coinrun=False)
    if args.env == "coinrun" and args.dual_head:
        _inject_game_id_chw(obs, coinrun=True)

    running = True
    for _ in range(args.steps):
        if not running:
            break

        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, _rewards, dones, infos = vec_env.step(action)
        if isinstance(obs, tuple):
            obs = obs[0]

        if args.env == "mario" and args.dual_head:
            _inject_game_id_chw(obs, coinrun=False)
        if args.env == "coinrun" and args.dual_head:
            _inject_game_id_chw(obs, coinrun=True)

        if args.render and screen is not None and args.env == "coinrun":
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
            if clock is not None and args.fps > 0:
                clock.tick(args.fps)

        if args.render and args.env == "mario" and hasattr(vec_env, "render"):
            try:
                vec_env.render()
            except Exception:
                pass

        if args.render and args.env == "both":
            # 混合环境下仅尝试绘制 CoinRun 子环境的 rgb（若有）
            if infos:
                for info in infos:
                    if not isinstance(info, dict):
                        continue
                    if info.get("game") != "coinrun":
                        continue
                    rgb = info.get("rgb")
                    if rgb is None:
                        continue
                    if screen is None:
                        pygame.init()
                        screen = pygame.display.set_mode(screen_size)
                        pygame.display.set_caption("Unified PPO — mixed (CoinRun)")
                        clock = pygame.time.Clock()
                    frame = np.transpose(rgb, (1, 0, 2))
                    surface = pygame.surfarray.make_surface(frame)
                    surface = pygame.transform.scale(surface, screen_size)
                    screen.blit(surface, (0, 0))
                    pygame.display.flip()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                    if clock is not None and args.fps > 0:
                        clock.tick(args.fps)
                    break

        if np.any(dones):
            obs = vec_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            if args.env == "mario" and args.dual_head:
                _inject_game_id_chw(obs, coinrun=False)
            if args.env == "coinrun" and args.dual_head:
                _inject_game_id_chw(obs, coinrun=True)

    vec_env.close()
    if screen is not None:
        pygame.quit()


if __name__ == "__main__":
    main()
