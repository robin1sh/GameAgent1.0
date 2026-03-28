"""
联合训练脚本：支持交替训练（B1）与混合训练（B2）。
支持 --pretrain-path 加载 backbone，用于阶段 4 的 Mario + CoinRun 通用智能体微调。
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage, VecNormalize
from envs import make_vec_env
from CNN_TEMPLATE import CustomCNN
from callbacks import MetricsEvalCallback, PrefixedEvalCallback
from policies import UnifiedDualHeadPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="联合训练：交替或混合")
    parser.add_argument("--mode", default="mixed", choices=["alternating", "mixed"],
                        help="alternating=交替训练, mixed=混合训练")
    parser.add_argument("--n-envs", type=int, default=10)
    parser.add_argument(
        "--mario-ratio",
        type=float,
        default=0.5,
        help="mixed 模式中 Mario 环境占比（其余为 CoinRun）",
    )
    parser.add_argument("--total-timesteps", type=int, default=int(1e7))
    parser.add_argument("--alternate-rounds", type=int, default=5,
                        help="交替模式下每环境训练轮数（百万步）")
    parser.add_argument("--exp-id", default="unified")
    parser.add_argument("--save-path", default="./best_model/")
    parser.add_argument("--log-path", default="./logs/")
    parser.add_argument("--pretrain-path", default="", help="模仿学习 backbone 路径，用于初始化")
    parser.add_argument("--resume", default="", help="从已有 PPO .zip 恢复并继续训练")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO 学习率初值（配合 --learning-rate-schedule 使用）",
    )
    parser.add_argument(
        "--learning-rate-schedule",
        choices=["constant", "linear"],
        default="constant",
        help="学习率调度：constant=常数，linear=线性衰减到 0",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.1,
        help="PPO 熵系数，适度降低可减弱后期过度探索",
    )
    parser.add_argument(
        "--dual-head",
        action="store_true",
        default=False,
        help="启用共享 CNN + 双动作头（Mario/CoinRun）策略",
    )
    parser.add_argument(
        "--fixed-level",
        action="store_true",
        default=True,
        help="CoinRun 使用固定关卡（默认开启，便于复现）",
    )
    parser.add_argument(
        "--no-fixed-level",
        action="store_false",
        dest="fixed_level",
        help="关闭 CoinRun 固定关卡，使用随机关卡",
    )
    parser.add_argument("--start-level", type=int, default=0, help="CoinRun 固定关卡 ID（第一关为 0）")
    parser.add_argument("--distribution-mode", default="easy", help="CoinRun 难度分布，建议 easy")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=3000,
        help="统一单回合最大时间步（Mario/CoinRun）",
    )
    parser.add_argument(
        "--normalize-reward",
        action="store_true",
        default=True,
        help="开启训练奖励归一化（默认开启）",
    )
    parser.add_argument(
        "--no-normalize-reward",
        action="store_false",
        dest="normalize_reward",
        help="关闭训练奖励归一化",
    )
    parser.add_argument(
        "--coinrun-reward-scale",
        type=float,
        default=200.0,
        help="混合训练时 CoinRun 奖励缩放系数（用于贴近 Mario 量级）",
    )
    parser.add_argument(
        "--coinrun-progress-coef",
        type=float,
        default=0.05,
        help="CoinRun 向右进度奖励系数（仅 mixed 且非 aligned 模式）",
    )
    parser.add_argument(
        "--coinrun-success-bonus",
        type=float,
        default=10.0,
        help="CoinRun 通关奖励（仅 mixed 且非 aligned 模式）",
    )
    parser.add_argument(
        "--coinrun-fail-penalty",
        type=float,
        default=2.0,
        help="CoinRun 失败惩罚（仅 mixed 且非 aligned 模式）",
    )
    parser.add_argument(
        "--coinrun-step-penalty",
        type=float,
        default=0.002,
        help="CoinRun 每步时间惩罚（仅 mixed 且非 aligned 模式）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    eval_freq_steps = max(100000 // args.n_envs, 1)

    # SB3 约定 progress_remaining 从 1 线性衰减到 0。
    if args.learning_rate_schedule == "linear":
        learning_rate = lambda progress_remaining: float(progress_remaining) * args.learning_rate
    else:
        learning_rate = args.learning_rate

    if args.mode == "mixed":
        vec_env = make_vec_env(
            "both",
            n_envs=args.n_envs,
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
        eval_env = make_vec_env(
            "both",
            n_envs=1,
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
        mario_eval_env = make_vec_env(
            "mario",
            n_envs=1,
            use_subproc=False,
            frame_stack=4,
            max_episode_steps=args.max_episode_steps,
        )
        coinrun_eval_env = make_vec_env(
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
    else:
        vec_env = make_vec_env(
            "mario",
            n_envs=args.n_envs,
            frame_stack=4,
            max_episode_steps=args.max_episode_steps,
        )
        eval_env = make_vec_env(
            "mario",
            n_envs=1,
            use_subproc=False,
            frame_stack=4,
            max_episode_steps=args.max_episode_steps,
        )

    # 训练与评估环境保持相同包装，避免图像布局与统计方式不一致。
    vec_env = VecMonitor(VecTransposeImage(vec_env))
    eval_env = VecMonitor(VecTransposeImage(eval_env))
    if args.mode == "mixed":
        mario_eval_env = VecMonitor(VecTransposeImage(mario_eval_env))
        coinrun_eval_env = VecMonitor(VecTransposeImage(coinrun_eval_env))
    if args.normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
            gamma=0.95,
            training=True,
        )

    log_dir = os.path.join(args.log_path, args.exp_id)
    save_path = os.path.join(args.save_path, args.exp_id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    if args.mode == "mixed":
        callback_log_root = os.path.join("./callback_logs/", args.exp_id)
        os.makedirs(callback_log_root, exist_ok=True)
        mario_eval_callback = PrefixedEvalCallback(
            mario_eval_env,
            best_model_save_path=save_path,
            log_path=os.path.join(callback_log_root, "mario"),
            eval_freq=eval_freq_steps,
            n_eval_episodes=5,
            metric_prefix="eval_mario",
        )
        coinrun_eval_callback = PrefixedEvalCallback(
            coinrun_eval_env,
            best_model_save_path=save_path,
            log_path=os.path.join(callback_log_root, "coinrun"),
            eval_freq=eval_freq_steps,
            n_eval_episodes=5,
            metric_prefix="eval_coinrun",
        )
        metrics_callback = MetricsEvalCallback(verbose=1)
        callback = CallbackList([mario_eval_callback, coinrun_eval_callback, metrics_callback])
    else:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path="./callback_logs/",
            eval_freq=eval_freq_steps,
            n_eval_episodes=5,
        )
        callback = eval_callback

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )
    policy_class = UnifiedDualHeadPolicy if args.dual_head else "CnnPolicy"

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"--resume 指定的模型不存在: {args.resume}")
        model = PPO.load(args.resume, env=vec_env)
        model.n_steps = 2048
        model.batch_size = 2048
        model.n_epochs = 10
        model.ent_coef = args.ent_coef
        model.gamma = 0.95
        model.learning_rate = learning_rate
        model._setup_lr_schedule()
        model.tensorboard_log = log_dir
        print(f"已从 checkpoint 恢复并继续训练: {args.resume}")
    else:
        model = PPO(
            policy=policy_class,
            env=vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=2048,
            n_epochs=10,
            ent_coef=args.ent_coef,
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
            env_name = "coinrun" if round_idx % 2 == 1 else "mario"
            vec_env.close()
            if env_name == "coinrun":
                vec_env = make_vec_env(
                    env_name,
                    n_envs=args.n_envs,
                    frame_stack=4,
                    fixed_level=args.fixed_level,
                    start_level=args.start_level,
                    distribution_mode=args.distribution_mode,
                    max_episode_steps=args.max_episode_steps,
                )
            else:
                vec_env = make_vec_env(
                    env_name,
                    n_envs=args.n_envs,
                    frame_stack=4,
                    max_episode_steps=args.max_episode_steps,
                )
            model.set_env(vec_env)
            eval_cb = EvalCallback(
                vec_env,
                best_model_save_path=save_path,
                log_path="./callback_logs/",
                eval_freq=eval_freq_steps,
                n_eval_episodes=5,
            )
            model.learn(total_timesteps=steps_per_round, callback=eval_cb)
    else:
        model.learn(total_timesteps=args.total_timesteps, callback=callback)

    final_model_path = os.path.join(save_path, "final_model.zip")
    model.save(final_model_path)
    print(f"最终模型已保存: {final_model_path}")
    if args.normalize_reward:
        vecnorm_path = os.path.join(save_path, "vecnormalize.pkl")
        vec_env.save(vecnorm_path)
        print(f"VecNormalize 统计已保存: {vecnorm_path}")
    vec_env.close()
    eval_env.close()
    if args.mode == "mixed":
        mario_eval_env.close()
        coinrun_eval_env.close()


if __name__ == "__main__":
    main()
