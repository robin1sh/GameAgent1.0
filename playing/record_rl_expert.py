"""
RL 专家轨迹采集：加载 PPO 模型，在 Mario / Jumper / CoinRun 上游玩，收集 state-action 对。
输出到 data_imitation_unified/rl_expert_{env}/
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from stable_baselines3 import PPO

from envs import make_vec_env

SAVE_EVERY = 2  # 每 N 步保存一次


def resolve_path(path: str) -> str:
    """若传入路径不存在，则尝试在项目根目录下查找。"""
    if os.path.exists(path):
        return path
    root_path = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(root_path):
        return root_path
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="RL 专家轨迹采集")
    parser.add_argument("--model-path", required=True, help="PPO 模型路径，如 ppo_mario_expert.zip")
    parser.add_argument("--env", required=True, choices=["mario", "jumper", "coinrun"], help="环境")
    parser.add_argument("--out-dir", default=None, help="输出目录，默认 data_imitation_unified/rl_expert_{env}/")
    parser.add_argument("--episodes", type=int, default=10, help="采集 episode 数")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument(
        "--fixed-level",
        action="store_true",
        default=True,
        help="Procgen 环境使用固定关卡（默认开启，便于复现）",
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
    args.model_path = resolve_path(args.model_path)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"找不到模型文件: {args.model_path}")

    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "data_imitation_unified", f"rl_expert_{args.env}")
    os.makedirs(out_dir, exist_ok=True)

    vec_env = make_vec_env(
        args.env,
        n_envs=1,
        use_subproc=False,
        frame_stack=4,
        fixed_level=args.fixed_level,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode,
    )
    model = PPO.load(args.model_path, env=vec_env)

    sample_id = 0
    for ep in range(args.episodes):
        obs = vec_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            result = vec_env.step(action)
            obs, rewards, dones, infos = result[:4]
            if isinstance(obs, tuple):
                obs = obs[0]
            done = dones[0]
            act = int(action[0])

            step += 1
            if step % args.save_every == 0:
                sample_id += 1
                sample_dir = os.path.join(out_dir, f"{sample_id:06d}")
                os.makedirs(sample_dir, exist_ok=True)
                frames = obs[0]
                for i in range(4):
                    f = frames[:, :, i] if frames.ndim == 3 else frames
                    cv2.imwrite(os.path.join(sample_dir, f"frame_{i}.png"), f)
                with open(os.path.join(sample_dir, "label.txt"), "w") as f:
                    f.write(str(act))

            if done:
                break

    vec_env.close()
    print(f"已采集 {sample_id} 个样本到 {out_dir}")


if __name__ == "__main__":
    main()
