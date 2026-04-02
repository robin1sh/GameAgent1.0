"""
PPO 权重融合工具（Model Soup）：
1) 融合两个同结构 PPO 模型
2) 在 Mario / CoinRun 上分别评估
3) 按加权总分自动选择最佳 alpha
"""
import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage

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


def parse_alpha_grid(alpha_grid: str) -> List[float]:
    """
    解析 alpha 列表，格式示例：
    "0.5,0.6,0.7,0.8,0.9,0.95"
    """
    values = []
    for token in alpha_grid.split(","):
        token = token.strip()
        if not token:
            continue
        alpha = float(token)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"alpha 必须在 [0,1] 区间，收到: {alpha}")
        values.append(alpha)
    if not values:
        raise ValueError("alpha 网格为空，请至少提供一个 alpha")
    return values


def should_fuse_key(param_key: str, fuse_scope: str) -> bool:
    """
    控制融合范围：
    - full: 融合 policy 的所有浮点参数（包含 actor/critic）
    - actor_feature: 仅融合特征提取器 + actor 头，减少 value 头干扰
    """
    if fuse_scope == "full":
        # full: policy 内所有浮点参数都参与融合（含 actor/critic）
        return True
    if fuse_scope == "actor_feature":
        # actor_feature: 只融合共享视觉特征 + actor 分支，避免 value 头干扰
        return (
            param_key.startswith("features_extractor.")
            or param_key.startswith("mlp_extractor.policy_net.")
            or param_key.startswith("action_net.")
        )
    raise ValueError(f"未知 fuse_scope: {fuse_scope}")


def fuse_policy_state_dict(
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
    alpha: float,
    fuse_scope: str,
) -> Dict[str, torch.Tensor]:
    """对 policy 参数做线性融合。"""
    # 先保证两个 state_dict 的参数键完全一致，否则无法一一对应融合
    if set(state_dict_a.keys()) != set(state_dict_b.keys()):
        missing_a = sorted(set(state_dict_b.keys()) - set(state_dict_a.keys()))
        missing_b = sorted(set(state_dict_a.keys()) - set(state_dict_b.keys()))
        raise ValueError(
            "两个模型 policy 参数键不一致，无法融合。\n"
            f"A 缺失: {missing_a[:5]}\n"
            f"B 缺失: {missing_b[:5]}"
        )

    fused = {}
    # 逐个参数名进行融合
    for key in state_dict_a.keys():
        tensor_a = state_dict_a[key]
        tensor_b = state_dict_b[key]
        # 同名参数的形状必须一致，避免发生错误广播
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(f"参数形状不一致: {key}, {tensor_a.shape} vs {tensor_b.shape}")

        # 非浮点参数（如整型 buffer）不做加权，直接继承 A。
        if not torch.is_floating_point(tensor_a) or not should_fuse_key(key, fuse_scope):
            # 非浮点参数或不在融合范围内的参数，直接继承 A
            fused[key] = tensor_a.clone()
            continue
        # 核心融合公式：W_fused = alpha * W_a + (1 - alpha) * W_b
        fused[key] = alpha * tensor_a + (1.0 - alpha) * tensor_b
    return fused


def evaluate_model(
    model: PPO,
    env_name: str,
    episodes: int,
    deterministic: bool,
    coinrun_use_aligned_reward: bool,
    coinrun_progress_coef: float,
    coinrun_success_bonus: float,
    coinrun_fail_penalty: float,
    coinrun_step_penalty: float,
) -> float:
    """在指定环境上评估平均回报。"""
    eval_env = make_vec_env(
        env_name=env_name,
        n_envs=1,
        use_subproc=False,
        frame_stack=4,
        fixed_level=True if env_name == "coinrun" else False,
        start_level=0,
        distribution_mode="easy",
        use_aligned_reward=coinrun_use_aligned_reward if env_name == "coinrun" else False,
        progress_coef=coinrun_progress_coef if env_name == "coinrun" else 0.02,
        success_bonus=coinrun_success_bonus if env_name == "coinrun" else 100.0,
        fail_penalty=coinrun_fail_penalty if env_name == "coinrun" else 20.0,
        step_penalty=coinrun_step_penalty if env_name == "coinrun" else 0.002,
    )
    eval_env = VecMonitor(VecTransposeImage(eval_env))
    mean_return = 0.0
    try:
        episode_returns: List[float] = []
        completed = 0

        obs = eval_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        cur_ret = 0.0

        while completed < episodes:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, _ = eval_env.step(action)
            if isinstance(obs, tuple):
                obs = obs[0]

            cur_ret += float(np.asarray(rewards).item())
            if bool(np.asarray(dones).item()):
                episode_returns.append(cur_ret)
                completed += 1
                cur_ret = 0.0
                obs = eval_env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]

        mean_return = float(np.mean(episode_returns)) if episode_returns else 0.0
    finally:
        eval_env.close()
    return mean_return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPO 权重融合 + 双环境评估选 alpha")
    parser.add_argument("--model-a", required=True, help="模型 A 路径（alpha 对应权重）")
    parser.add_argument("--model-b", required=True, help="模型 B 路径（1-alpha 对应权重）")
    parser.add_argument("--output-dir", default="best_model/fused", help="融合输出目录")
    parser.add_argument("--output-name", default="best_model.zip", help="最佳融合模型文件名")
    parser.add_argument(
        "--alpha-grid",
        default="0.5,0.6,0.7,0.8,0.9,0.95",
        help="逗号分隔 alpha 网格",
    )
    parser.add_argument(
        "--fuse-scope",
        default="full",
        choices=["full", "actor_feature"],
        help="full=融合 policy 全部浮点参数，actor_feature=仅融合特征+actor",
    )
    parser.add_argument("--mario-episodes", type=int, default=5, help="Mario 评估回合数")
    parser.add_argument("--coinrun-episodes", type=int, default=5, help="CoinRun 评估回合数")
    parser.add_argument("--mario-weight", type=float, default=0.7, help="综合分中 Mario 权重")
    parser.add_argument("--coinrun-weight", type=float, default=0.3, help="综合分中 CoinRun 权重")
    parser.add_argument(
        "--coinrun-use-aligned-reward",
        action="store_true",
        default=False,
        help="CoinRun 评估启用 aligned 奖励（默认关闭，保持 Mario 原生 + CoinRun 塑形口径）",
    )
    parser.add_argument(
        "--no-coinrun-use-aligned-reward",
        action="store_false",
        dest="coinrun_use_aligned_reward",
        help="显式关闭 CoinRun 评估 aligned 奖励",
    )
    parser.add_argument("--coinrun-progress-coef", type=float, default=0.02, help="CoinRun 评估进度项系数")
    parser.add_argument("--coinrun-success-bonus", type=float, default=100.0, help="CoinRun 评估通关奖励")
    parser.add_argument("--coinrun-fail-penalty", type=float, default=20.0, help="CoinRun 评估失败惩罚")
    parser.add_argument("--coinrun-step-penalty", type=float, default=0.002, help="CoinRun 评估每步惩罚")
    parser.add_argument("--deterministic", action="store_true", help="评估时使用确定性动作")
    parser.add_argument("--device", default="cpu", help="推理设备，如 cpu / cuda")
    parser.add_argument(
        "--score-norm",
        default="none",
        choices=["none", "minmax"],
        help="综合分归一化方式：none=原始回报加权，minmax=按 alpha 网格内最小-最大归一化后加权",
    )
    return parser


def main():
    args = build_parser().parse_args()
    model_a_path = resolve_path(args.model_a)
    model_b_path = resolve_path(args.model_b)
    if not os.path.isfile(model_a_path):
        raise FileNotFoundError(f"找不到模型 A: {model_a_path}")
    if not os.path.isfile(model_b_path):
        raise FileNotFoundError(f"找不到模型 B: {model_b_path}")

    # 解析 alpha 网格（例如 0.5,0.6,0.7）
    alphas = parse_alpha_grid(args.alpha_grid)
    os.makedirs(resolve_path(args.output_dir), exist_ok=True)

    model_a = PPO.load(model_a_path, device=args.device)
    model_b = PPO.load(model_b_path, device=args.device)

    # 只融合 policy 参数（特征提取器/actor/critic 都在此字典中）
    sd_a = model_a.policy.state_dict()
    sd_b = model_b.policy.state_dict()

    best: Tuple[float, float, float, float] = (-1.0, -1e18, -1e18, -1e18)
    # best = (alpha, score, mario_mean, coinrun_mean)
    raw_results: List[Tuple[float, float, float]] = []
    # raw_results 元素为 (alpha, mario_mean, coinrun_mean)

    print("开始融合评估...")
    # 网格搜索不同 alpha，评估后自动选最优
    for alpha in alphas:
        fused_sd = fuse_policy_state_dict(
            state_dict_a=sd_a,
            state_dict_b=sd_b,
            alpha=alpha,
            fuse_scope=args.fuse_scope,
        )

        # 用 model_a 作为容器加载融合参数，不改动原始文件。
        # 这一步只修改内存中的参数，不会覆盖 model_a_path 的原模型文件
        model_a.policy.load_state_dict(fused_sd, strict=True)

        mario_mean = evaluate_model(
            model=model_a,
            env_name="mario",
            episodes=args.mario_episodes,
            deterministic=args.deterministic,
            coinrun_use_aligned_reward=args.coinrun_use_aligned_reward,
            coinrun_progress_coef=args.coinrun_progress_coef,
            coinrun_success_bonus=args.coinrun_success_bonus,
            coinrun_fail_penalty=args.coinrun_fail_penalty,
            coinrun_step_penalty=args.coinrun_step_penalty,
        )
        coinrun_mean = evaluate_model(
            model=model_a,
            env_name="coinrun",
            episodes=args.coinrun_episodes,
            deterministic=args.deterministic,
            coinrun_use_aligned_reward=args.coinrun_use_aligned_reward,
            coinrun_progress_coef=args.coinrun_progress_coef,
            coinrun_success_bonus=args.coinrun_success_bonus,
            coinrun_fail_penalty=args.coinrun_fail_penalty,
            coinrun_step_penalty=args.coinrun_step_penalty,
        )
        # 记录该 alpha 在双环境下的原始均值回报
        raw_results.append((alpha, mario_mean, coinrun_mean))

    mario_vals = [x[1] for x in raw_results]
    coinrun_vals = [x[2] for x in raw_results]
    mario_min, mario_max = min(mario_vals), max(mario_vals)
    coinrun_min, coinrun_max = min(coinrun_vals), max(coinrun_vals)
    eps = 1e-8

    # 计算综合分：可选 min-max 归一化后按任务权重加权
    for alpha, mario_mean, coinrun_mean in raw_results:
        if args.score_norm == "minmax":
            # 将两任务回报归一到 [0,1]，避免数值量级大的任务主导综合分。
            mario_for_score = (mario_mean - mario_min) / (mario_max - mario_min + eps)
            coinrun_for_score = (coinrun_mean - coinrun_min) / (coinrun_max - coinrun_min + eps)
        else:
            mario_for_score = mario_mean
            coinrun_for_score = coinrun_mean

        # 综合评分（论文中可表述为双任务加权目标）
        score = args.mario_weight * mario_for_score + args.coinrun_weight * coinrun_for_score

        print(
            f"[alpha={alpha:.4f}] "
            f"mario={mario_mean:.3f}, coinrun={coinrun_mean:.3f}, "
            f"score={score:.3f} (norm={args.score_norm})"
        )
        if score > best[1]:
            best = (alpha, score, mario_mean, coinrun_mean)

    # 用最优 alpha 重新融合一次，并保存为最终 fused 初始化模型
    best_alpha = best[0]
    best_sd = fuse_policy_state_dict(
        state_dict_a=sd_a,
        state_dict_b=sd_b,
        alpha=best_alpha,
        fuse_scope=args.fuse_scope,
    )
    model_a.policy.load_state_dict(best_sd, strict=True)

    output_dir = resolve_path(args.output_dir)
    output_path = os.path.join(output_dir, args.output_name)
    model_a.save(output_path)

    print("\n融合完成。")
    print(
        f"最佳 alpha={best[0]:.4f}, score={best[1]:.3f}, "
        f"mario={best[2]:.3f}, coinrun={best[3]:.3f}"
    )
    print(f"最佳融合模型已保存: {output_path}")


if __name__ == "__main__":
    main()
