"""
加载训练好的模型，接入游戏环境并用模型输出动作。
"""
import argparse
import json
import os
import sys
from datetime import datetime

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from PIL import Image
import torch
from torchvision import transforms

# 使脚本可以导入 train.train_cnn_imitation
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train.train_cnn_imitation import build_model, build_model_legacy  # noqa: E402


def resolve_path(path: str) -> str:
    """若传入路径不存在，则尝试在项目根目录下查找。"""
    if os.path.exists(path):
        return path
    root_path = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(root_path):
        return root_path
    return path


def load_class_map(path: str) -> dict:
    """加载 class_to_idx.json，并返回 idx->class 的映射。"""
    with open(path, "r") as f:
        class_to_idx = json.load(f)
    return {int(v): k for k, v in class_to_idx.items()}


def build_action_mapping(env, class_names):
    """
    根据 env.get_action_meanings() 构建：模型类别名 -> 动作索引 的映射。
    """
    action_names = env.get_action_meanings()
    name_map = {
        "NOOP": "noop",
        "A": "jump",
        "B": "run",
        "right": "right",
        "left": "left",
        "right A": "right_jump",
        "right B": "right_run",
        "right A B": "right_run_jump",
    }
    action_name_to_index = {}
    for idx, name in enumerate(action_names):
        safe_name = name_map.get(name, name.replace(" ", "_").lower())
        action_name_to_index[safe_name] = idx

    mapping = {}
    for cname in class_names:
        if cname in action_name_to_index:
            mapping[cname] = action_name_to_index[cname]
    return mapping


def resolve_jump_action(action_mapping):
    """选择一个可用的跳跃动作（优先向右跳）。"""
    for name in ("right_jump", "right_run_jump", "jump", "right"):
        if name in action_mapping:
            return action_mapping[name]
    return 0


def main():
    parser = argparse.ArgumentParser(description="用模型控制马里奥环境")
    parser.add_argument("--model", default="imitation_cnn.pt", help="模型权重文件路径")
    parser.add_argument("--class-map", default="class_to_idx.json", help="类别映射文件路径")
    parser.add_argument("--size", type=int, default=128, help="模型输入尺寸")
    parser.add_argument("--episodes", type=int, default=1, help="运行的回合数")
    parser.add_argument("--max-steps", type=int, default=2000, help="每回合最大步数")
    parser.add_argument("--render", action="store_true", help="渲染游戏画面")
    parser.add_argument(
        "--debug-steps",
        type=int,
        default=10,
        help="前 N 步打印预测动作与映射结果（0 表示不打印）",
    )
    parser.add_argument(
        "--debug-probs",
        action="store_true",
        help="打印每步前几类的概率（用于确认是否全部偏向 noop）",
    )
    parser.add_argument(
        "--min-jump-frames",
        type=int,
        default=5,
        help="跳跃类动作最短持续帧数",
    )
    parser.add_argument(
        "--record-file",
        default="",
        help="将每回合是否通关第一关的结果追加写入该文件（不指定则不写）",
    )
    args = parser.parse_args()

    # 加载模型（先尝试新结构，失败则回退旧结构）
    args.class_map = resolve_path(args.class_map)
    args.model = resolve_path(args.model)
    if not os.path.exists(args.class_map):
        raise FileNotFoundError(
            f"找不到类别映射文件: {args.class_map}\n"
            f"请确认 `class_to_idx.json` 在项目根目录，或用 --class-map 指定路径。"
        )
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"找不到模型权重文件: {args.model}\n"
            "这个脚本只支持模仿学习 CNN 的 PyTorch state_dict（通常叫 imitation_cnn.pt）。\n"
            "如果你要跑 Stable-Baselines3 的 best_model.zip，请改用：\n"
            "  python test/test_model.py --model best_model/best_model.zip --render\n"
        )
    idx_to_class = load_class_map(args.class_map)
    try:
        state = torch.load(args.model, map_location="cpu")
    except Exception as e:
        suffix = os.path.splitext(args.model)[1].lower()
        hint = (
            "无法用 torch.load 读取该文件。\n"
            "该脚本期望的是 `torch.save(model.state_dict(), ...)` 产生的权重文件（.pt）。\n"
        )
        if suffix in {".zip", ".tgz", ".gz"}:
            hint += (
                f"你给的看起来是压缩包/Stable-Baselines3 模型（{suffix}），不是 imitation_cnn.pt。\n"
                "若要跑 SB3 的 best_model.zip，请用：\n"
                "  python test/test_model.py --model best_model/best_model.zip --render\n"
            )
        raise RuntimeError(f"{hint}\n原始错误: {repr(e)}") from e
    try:
        model = build_model(num_classes=len(idx_to_class))
        model.load_state_dict(state)
        model_type = "new"
    except RuntimeError:
        model = build_model_legacy(num_classes=len(idx_to_class))
        model.load_state_dict(state)
        model_type = "legacy"
    model.eval()

    # 预处理：保证 1 通道、固定尺寸
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
        ]
    )

    env = gym_super_mario_bros.make("SuperMarioBros-v2")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    class_names = [idx_to_class[i] for i in sorted(idx_to_class)]
    action_mapping = build_action_mapping(env, class_names)
    if not action_mapping:
        raise RuntimeError("未能把模型类别映射到环境动作，请检查 class_to_idx.json")
    print("环境动作列表:", env.get_action_meanings())
    print("模型结构:", model_type)
    print("模型类别 -> 动作索引映射:", action_mapping)

    jump_action = resolve_jump_action(action_mapping)
    jump_hold_steps = 0

    # 记录每回合是否通关第一关（1-1），用于最后汇总
    episode_results = []

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        step = 0
        cleared_1_1 = False  # 本回合是否通关 1-1（碰到旗杆）
        while not done and step < args.max_steps:
            if args.render:
                env.render()

            img = Image.fromarray(obs)
            x = transform(img).unsqueeze(0)  # [1, 1, H, W]
            with torch.no_grad():
                logits = model(x)
                pred_idx = int(torch.argmax(logits, dim=1).item())
            pred_class = idx_to_class.get(pred_idx, "noop")
            action = action_mapping.get(pred_class, 0)  # fallback NOOP
            if jump_hold_steps > 0:
                action = jump_action
                jump_hold_steps -= 1
            elif pred_class in {"right_jump", "right_run_jump", "jump"} and args.min_jump_frames > 1:
                jump_hold_steps = args.min_jump_frames - 1
                action = jump_action
            if args.debug_steps > 0 and step < args.debug_steps:
                action_name = env.get_action_meanings()[action]
                if args.debug_probs:
                    probs = torch.softmax(logits, dim=1).squeeze(0)
                    topk = torch.topk(probs, k=min(3, probs.numel()))
                    top_items = []
                    for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                        cname = idx_to_class.get(int(idx), str(idx))
                        top_items.append(f"{cname}:{score:.3f}")
                    probs_text = ", ".join(top_items)
                else:
                    probs_text = "off"
                print(
                    f"step {step}: pred_idx={pred_idx}, pred_class={pred_class}, "
                    f"action={action} ({action_name}), top_probs=[{probs_text}]"
                )
                if pred_class not in action_mapping:
                    print(f"  [警告] 预测类别未映射到动作，已使用 fallback NOOP")

            obs, reward, done, info = env.step(action)
            step += 1
            # 判断是否通关第一关：当前在 1-1 且碰到旗杆（flag_get）
            if info.get("flag_get", False) and info.get("world", 1) == 1 and info.get("stage", 1) == 1:
                cleared_1_1 = True

        episode_results.append({"episode": ep + 1, "steps": step, "cleared_1_1": cleared_1_1})
        status = "通关第一关 ✓" if cleared_1_1 else "未通关第一关"
        print(f"episode {ep+1}/{args.episodes} finished, steps={step}, {status}")

    env.close()

    # 汇总：是否通关第一关
    cleared_count = sum(1 for r in episode_results if r["cleared_1_1"])
    print(f"\n===== 第一关通关记录 =====")
    print(f"总回合数: {args.episodes}, 通关第一关: {cleared_count} 次")
    if args.record_file:
        record_path = resolve_path(args.record_file)
        with open(record_path, "a", encoding="utf-8") as f:
            line = f"{datetime.now().isoformat()}\t{args.episodes}\t{cleared_count}\t{args.model}\n"
            f.write(line)
        print(f"结果已追加写入: {record_path}")


if __name__ == "__main__":
    main()
