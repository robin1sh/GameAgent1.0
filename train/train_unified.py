"""
联合训练脚本：支持交替训练（B1）与混合训练（B2）。
支持 --pretrain-path 加载 backbone，用于阶段 4 通用智能体微调。
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from envs import make_vec_env
from model_script import CustomCNN


def parse_args():
    parser = argparse.ArgumentParser(description="联合训练：交替或混合")
    parser.add_argument("--mode", default="mixed", choices=["alternating", "mixed"],
                        help="alternating=交替训练, mixed=混合训练")
    parser.add_argument("--n-envs", type=int, default=10)
    parser.add_argument("--total-timesteps", type=int, default=int(1e7))
    parser.add_argument("--alternate-rounds", type=int, default=5,
                        help="交替模式下每环境训练轮数（百万步）")
    parser.add_argument("--exp-id", default="unified")
    parser.add_argument("--save-path", default="./best_model/")
    parser.add_argument("--log-path", default="./logs/")
    parser.add_argument("--pretrain-path", default="", help="模仿学习 backbone 路径，用于初始化")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "mixed":
        vec_env = make_vec_env("both", n_envs=args.n_envs, frame_stack=4)
    else:
        vec_env = make_vec_env("mario", n_envs=args.n_envs, frame_stack=4)

    log_dir = os.path.join(args.log_path, args.exp_id)
    save_path = os.path.join(args.save_path, args.exp_id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=save_path,
        log_path="./callback_logs/",
        eval_freq=max(10000 // args.n_envs, 1),
        n_eval_episodes=5,
    )

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=8192,
        n_epochs=5,
        ent_coef=0.1,
        gamma=0.95,
        verbose=1,
        tensorboard_log=log_dir,
    )

    if args.pretrain_path and os.path.isfile(args.pretrain_path):
        pretrain = torch.load(args.pretrain_path, map_location="cpu")
        model.policy.features_extractor.load_state_dict(pretrain, strict=False)
        print(f"已加载预训练 backbone: {args.pretrain_path}")

    if args.mode == "alternating":
        steps_per_round = args.alternate_rounds * 1_000_000
        for round_idx in range(max(1, args.total_timesteps // steps_per_round)):
            env_name = "jumper" if round_idx % 2 == 1 else "mario"
            vec_env.close()
            vec_env = make_vec_env(env_name, n_envs=args.n_envs, frame_stack=4)
            model.set_env(vec_env)
            eval_cb = EvalCallback(
                vec_env,
                best_model_save_path=save_path,
                log_path="./callback_logs/",
                eval_freq=max(10000 // args.n_envs, 1),
                n_eval_episodes=5,
            )
            model.learn(total_timesteps=steps_per_round, callback=eval_cb)
    else:
        model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

    model.save(f"ppo_unified_{args.mode}")
    vec_env.close()


if __name__ == "__main__":
    main()
