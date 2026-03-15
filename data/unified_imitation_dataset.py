"""
统一模仿学习数据集：支持 mario、jumper、rl_expert_mario、rl_expert_jumper。
每样本：4 张 PNG 堆叠为 (64, 64, 4)，label.txt 为 0–14。
"""
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# 支持的数据源
DATA_SOURCES = ["mario", "jumper", "rl_expert_mario", "rl_expert_jumper"]


class UnifiedImitationDataset(Dataset):
    """
    统一格式模仿学习数据集。
    每样本目录：frame_0.png ~ frame_3.png + label.txt
    """

    def __init__(
        self,
        root: str,
        include_sources: list = None,
        transform=None,
    ):
        """
        :param root: 根目录，如 data_imitation_unified/
        :param include_sources: 包含的数据源，如 ["mario", "jumper", "rl_expert_mario", "rl_expert_jumper"]
        :param transform: 可选，对堆叠后的 obs 做变换
        """
        self.root = Path(root)
        self.include_sources = include_sources or DATA_SOURCES
        self.transform = transform
        self.samples = []  # [(sample_dir, label), ...]

        for src in self.include_sources:
            src_dir = self.root / src
            if not src_dir.exists():
                continue
            for sample_dir in sorted(src_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                label_path = sample_dir / "label.txt"
                if not label_path.exists():
                    continue
                label = int(label_path.read_text().strip())
                if 0 <= label < 15:
                    self.samples.append((str(sample_dir), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir, label = self.samples[idx]
        frames = []
        for i in range(4):
            img_path = Path(sample_dir) / f"frame_{i}.png"
            img = np.array(Image.open(img_path).convert("L"))
            if img.shape != (64, 64):
                from PIL import Image as PILImage
                img = np.array(PILImage.fromarray(img).resize((64, 64)))
            frames.append(img)
        obs = np.stack(frames, axis=-1)  # (64, 64, 4)
        obs = obs.astype(np.float32) / 255.0

        if self.transform:
            obs = self.transform(obs)

        return torch.from_numpy(obs).float(), label
