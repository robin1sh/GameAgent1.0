"""
PPO 训练脚本：支持 mario / jumper / coinrun / both，使用自定义 CNN，统一 15 维动作空间。
支持 --pretrain-path 加载模仿学习 backbone 初始化。
"""
import argparse
import os
import sys
from datetime import datetime

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage

from envs import make_vec_env
from CNN_TEMPLATE import CustomCNN
from callbacks import MetricsEvalCallback


def build_linear_schedule(initial_value: float, final_value: float):
    """线性学习率调度：训练开始时为 initial_value，结束时衰减到 final_value。"""
    def schedule(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return schedule


def parse_args():
    parser = argparse.ArgumentParser(description="PPO 训练：Mario / Jumper / CoinRun / 联合")
    parser.add_argument("--env", default="mario", choices=["mario", "jumper", "coinrun", "both"],
                        help="训练环境")
    parser.add_argument("--n-envs", type=int, default=10, help="并行环境数")
    parser.add_argument("--total-timesteps", type=int, default=int(1e7), help="总训练步数")
    parser.add_argument("--exp-id", default="", help="实验 ID，用于日志目录")
    parser.add_argument("--save-path", default="./best_model/", help="最佳模型保存路径")
    parser.add_argument("--log-path", default="./logs/", help="TensorBoard 日志路径")
    parser.add_argument("--no-subproc", action="store_true", help="使用 DummyVecEnv 替代 SubprocVecEnv")
    parser.add_argument("--pretrain-path", default="", help="模仿学习 backbone 路径，用于初始化特征提取器")
    parser.add_argument("--resume", default="", help="从已有 PPO .zip 恢复并继续训练（与 --pretrain-path 二选一）")
    parser.add_argument("--callback-log-path", default="./callback_logs/", help="EvalCallback 日志根目录")
    parser.add_argument(
        "--fixed-level",
        action="store_true",
        default=True,
        help="procgen 环境使用固定关卡（默认开启，可复现训练）",
    )
    parser.add_argument(
        "--no-fixed-level",
        action="store_false",
        dest="fixed_level",
        help="关闭 procgen 固定关卡，使用随机关卡",
    )
    parser.add_argument("--start-level", type=int, default=0, help="procgen 固定关卡 ID（第一关为 0）")
    parser.add_argument("--distribution-mode", default="easy", help="procgen 关卡分布模式，建议 easy")
    # PPO 超参数（允许从入口脚本注入，便于单任务独立微调）
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="PPO 学习率")
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "linear"],
        default="constant",
        help="学习率调度方式：constant=常数，linear=线性衰减",
    )
    parser.add_argument(
        "--final-learning-rate",
        type=float,
        default=1e-5,
        help="当 --lr-schedule=linear 时使用的最终学习率",
    )
    parser.add_argument("--n-steps", type=int, default=2048, help="每次 rollout 步数")
    parser.add_argument("--batch-size", type=int, default=8192, help="PPO 批大小")
    parser.add_argument("--n-epochs", type=int, default=5, help="每次更新 epoch 数")
    parser.add_argument("--ent-coef", type=float, default=0.1, help="熵正则系数")
    parser.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="评估间隔（按环境总步数计，内部会自动按 n_envs 折算为回调步数）",
    )
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="每次评估的回合数")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.exp_id:
        args.exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.lr_schedule == "linear":
        learning_rate = build_linear_schedule(args.learning_rate, args.final_learning_rate)
    else:
        learning_rate = args.learning_rate

    vec_env = make_vec_env(
        env_name=args.env,
        n_envs=args.n_envs,
        use_subproc=not args.no_subproc,
        frame_stack=4,
        fixed_level=args.fixed_level,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode,
    )
    # 评估环境与训练环境分离，避免状态与统计相互污染
    eval_env = make_vec_env(
        env_name=args.env,
        n_envs=1,
        use_subproc=False,
        frame_stack=4,
        fixed_level=args.fixed_level,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode,
    )
    # 保证 train/eval 观测布局与统计封装一致，避免类型不一致与 Monitor 警告
    vec_env = VecMonitor(VecTransposeImage(vec_env))
    eval_env = VecMonitor(VecTransposeImage(eval_env))

    log_dir = args.log_path
    if args.exp_id:
        log_dir = os.path.join(log_dir, args.exp_id)
    os.makedirs(log_dir, exist_ok=True)

    save_path = args.save_path
    if args.exp_id:
        save_path = os.path.join(save_path, args.exp_id)
    os.makedirs(save_path, exist_ok=True)

    callback_log_dir = args.callback_log_path
    if args.exp_id:
        callback_log_dir = os.path.join(callback_log_dir, args.exp_id)
    os.makedirs(callback_log_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=callback_log_dir,
        # SB3 的 eval_freq 按 callback 调用次数计；这里折算为“环境总步数”更直观。
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
    )
    metrics_callback = MetricsEvalCallback(verbose=1)
    callback = CallbackList([eval_callback, metrics_callback])

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"--resume 指定的模型不存在: {args.resume}")
        model = PPO.load(args.resume, env=vec_env)
        # 恢复训练时，显式覆盖可调超参数，避免沿用 checkpoint 内旧配置
        if getattr(model, "n_envs", args.n_envs) != args.n_envs:
            raise ValueError(
                f"--resume 模型的 n_envs={model.n_envs} 与当前 --n-envs={args.n_envs} 不一致。"
                "请使用相同 n_envs 继续训练，或不使用 --resume 重新训练。"
            )
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
        model.n_epochs = args.n_epochs
        model.ent_coef = args.ent_coef
        model.gamma = args.gamma
        model.learning_rate = learning_rate
        model.lr_schedule = get_schedule_fn(learning_rate)
        # 重新构建 rollout buffer，确保 n_steps / n_envs / gamma 覆盖后形状与配置一致
        # 兼容不同 SB3 版本：新版本可能没有 rollout_buffer_class 字段
        rollout_buffer_cls = getattr(model, "rollout_buffer_class", type(model.rollout_buffer))
        model.rollout_buffer = rollout_buffer_cls(
            model.n_steps,
            model.observation_space,
            model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=args.n_envs,
        )
        model.tensorboard_log = log_dir
        print(f"已从 checkpoint 恢复并继续训练: {args.resume}")
    else:
        model = PPO(
            policy="CnnPolicy",
            env=vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            ent_coef=args.ent_coef,
            gamma=args.gamma,
            verbose=1,
            tensorboard_log=log_dir,
        )

        if args.pretrain_path and os.path.isfile(args.pretrain_path):
            pretrain = torch.load(args.pretrain_path, map_location="cpu")
            model.policy.features_extractor.load_state_dict(pretrain, strict=False)
            print(f"已加载预训练 backbone: {args.pretrain_path}")

    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    final_model_path = os.path.join(save_path, f"ppo_{args.env}_final")
    model.save(final_model_path)
    print(f"最终模型已保存: {final_model_path}.zip")
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
