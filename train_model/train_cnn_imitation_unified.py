"""
统一格式模仿学习训练：15 类，CNN backbone 与 CustomCNN 结构一致。
支持阶段 4 数据源别名映射（如 expert_both -> rl_expert_mario + rl_expert_coinrun）。
"""
import argparse
import os
import random
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split

from dataset.unified_imitation_dataset import UnifiedImitationDataset, DATA_SOURCES

NUM_CLASSES = 15
FEATURES_DIM = 512
DATA_ROOT = os.path.join(PROJECT_ROOT, "data_imitation_unified")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "CNN_models")

# 逻辑数据源 -> 真实目录名（UnifiedImitationDataset 使用真实目录名）
SOURCE_TO_DIRS = {
    "mario": ["mario"],
    "coinrun": ["coinrun"],
    "human_both": ["mario", "coinrun"],
    "expert_mario": ["rl_expert_mario"],
    "expert_coinrun": ["rl_expert_coinrun"],
    "expert_both": ["rl_expert_mario", "rl_expert_coinrun"],
}


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_imitation_model():
    """
    与 CustomCNN 结构一致的 backbone + 15 类分类头。
    输入 (N, 4, 84, 84) channels_first。
    """
    class ImitationCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                x = torch.zeros(1, 4, 84, 84)
                n_flatten = self.cnn(x).shape[1]
            self.linear = nn.Sequential(
                nn.Linear(n_flatten, FEATURES_DIM),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(FEATURES_DIM, NUM_CLASSES)

        def forward(self, x):
            feat = self.linear(self.cnn(x))
            return self.classifier(feat)

        def get_backbone_state_dict(self):
            """仅 backbone（cnn + linear），供 PPO 加载。"""
            return {k: v for k, v in self.state_dict().items() if not k.startswith("classifier")}

    return ImitationCNN()


def validate_dataset_preprocess(dataset, max_checks=64):
    """
    训练前快速抽样检查数据预处理一致性，避免脏样本进入训练。
    检查项：shape / dtype / 值域 / 有限值 / label 范围。
    """
    total = len(dataset)
    if total == 0:
        raise ValueError("数据集为空，无法执行预处理检查。")
    checks = max(1, min(int(max_checks), total))
    step = max(1, total // checks)
    checked = 0
    for idx in range(0, total, step):
        obs, label = dataset[idx]
        if not torch.is_tensor(obs):
            raise TypeError(f"样本 {idx} 的 obs 不是 torch.Tensor。")
        if obs.ndim != 3:
            raise ValueError(f"样本 {idx} 维度异常：obs.ndim={obs.ndim}，预期为 3。")
        if obs.dtype != torch.float32:
            raise ValueError(f"样本 {idx} dtype 异常：obs.dtype={obs.dtype}，预期为 torch.float32。")
        # 兼容 HWC(84,84,4) 与 CHW(4,84,84)，训练时会统一转为 CHW。
        valid_hwc = tuple(obs.shape) == (84, 84, 4)
        valid_chw = tuple(obs.shape) == (4, 84, 84)
        if not (valid_hwc or valid_chw):
            raise ValueError(f"样本 {idx} shape 异常：obs.shape={tuple(obs.shape)}。")
        if not torch.isfinite(obs).all():
            raise ValueError(f"样本 {idx} 含 NaN/Inf。")
        obs_min = float(obs.min().item())
        obs_max = float(obs.max().item())
        if obs_min < -1e-6 or obs_max > 1.0 + 1e-6:
            raise ValueError(
                f"样本 {idx} 值域异常：min={obs_min:.6f}, max={obs_max:.6f}，预期在 [0,1]。"
            )
        label_int = int(label)
        if not (0 <= label_int < NUM_CLASSES):
            raise ValueError(f"样本 {idx} label 越界：label={label_int}，预期 [0,{NUM_CLASSES - 1}]。")
        checked += 1
        if checked >= checks:
            break
    print(f"预处理检查通过：抽检 {checked}/{total} 条样本。")


def parse_args():
    parser = argparse.ArgumentParser(description="统一模仿学习训练")
    parser.add_argument(
        "--source",
        default="expert_both",
        choices=list(SOURCE_TO_DIRS.keys()),
        help="逻辑数据源：mario/coinrun/human_both/expert_mario/expert_coinrun/expert_both",
    )
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--max-samples-per-source",
        type=int,
        default=0,
        help="每个真实数据源最多保留多少条样本；0 表示不限制",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="模型保存目录，默认保存到 CNN_models 文件夹",
    )
    parser.add_argument(
        "--check-samples",
        type=int,
        default=64,
        help="训练前预处理抽检样本数；设为 0 可跳过检查",
    )
    parser.add_argument("--no-amp", action="store_true", help="禁用混合精度")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    os.chdir(PROJECT_ROOT)
    os.makedirs(args.out_dir, exist_ok=True)

    include_sources = SOURCE_TO_DIRS[args.source]

    dataset = UnifiedImitationDataset(args.data_root, include_sources=include_sources)
    if len(dataset) == 0:
        print(f"错误：在 {args.data_root} 下未找到 {include_sources} 的数据。请先运行采集脚本。")
        return

    if args.max_samples_per_source > 0:
        grouped_samples = defaultdict(list)
        for sample_dir, label in dataset.samples:
            source_name = os.path.basename(os.path.dirname(sample_dir))
            grouped_samples[source_name].append((sample_dir, label))

        limited_samples = []
        rng = random.Random(42)
        print("按 source 限制样本上限：")
        for source_name in sorted(grouped_samples.keys()):
            source_samples = grouped_samples[source_name]
            before = len(source_samples)
            if before > args.max_samples_per_source:
                source_samples = list(source_samples)
                rng.shuffle(source_samples)
                source_samples = source_samples[: args.max_samples_per_source]
            after = len(source_samples)
            print(f"  {source_name}: {before} -> {after}")
            limited_samples.extend(source_samples)

        dataset.samples = limited_samples
        print(f"限制后总样本数: {len(dataset.samples)}")

    if args.check_samples > 0:
        validate_dataset_preprocess(dataset, max_checks=args.check_samples)
    else:
        print("已跳过预处理抽检（--check-samples=0）。")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = build_imitation_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    use_amp = not args.no_amp and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for obs, labels in train_loader:
            obs = obs.to(device)
            if obs.shape[-1] == 4:
                obs = obs.permute(0, 3, 1, 2)
            labels = labels.to(device)
            optimizer.zero_grad()
            if use_amp:
                with autocast():
                    out = model(obs)
                    loss = criterion(out, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(obs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
            train_loss += loss.item() * obs.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for obs, labels in val_loader:
                obs = obs.to(device)
                if obs.shape[-1] == 4:
                    obs = obs.permute(0, 3, 1, 2)
                labels = labels.to(device)
                out = model(obs)
                correct += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)

        train_loss /= len(train_set)
        val_acc = correct / max(1, total)
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")
        scheduler.step()

    out_name = f"{args.source}_backbone.pt"
    backbone_path = os.path.join(args.out_dir, out_name)
    backbone_state = model.get_backbone_state_dict()
    torch.save(backbone_state, backbone_path)
    print(f"Backbone 已保存: {backbone_path}")

    full_name = f"{args.source}_cnn.pt"
    full_path = os.path.join(args.out_dir, full_name)
    torch.save(model.state_dict(), full_path)
    print(f"完整模型已保存: {full_path}")


if __name__ == "__main__":
    main()
