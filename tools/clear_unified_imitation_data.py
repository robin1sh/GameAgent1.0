"""
清理 unified 模仿学习数据目录中的样本子目录。

示例：
  # 预览将删除哪些内容（不实际删除）
  python tools/clear_unified_imitation_data.py --source expert_both --dry-run

  # 实际删除 expert 两个目录的数据
  python tools/clear_unified_imitation_data.py --source expert_both --yes
"""
import argparse
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data_imitation_unified"

# 逻辑 source -> 真实目录
SOURCE_TO_DIRS = {
    "mario": ["mario"],
    "coinrun": ["coinrun"],
    "human_both": ["mario", "coinrun"],
    "expert_mario": ["rl_expert_mario"],
    "expert_coinrun": ["rl_expert_coinrun"],
    "expert_both": ["rl_expert_mario", "rl_expert_coinrun"],
    "all": ["mario", "coinrun", "rl_expert_mario", "rl_expert_coinrun"],
}


def _count_numeric_dirs(target_dir: Path) -> int:
    """统计形如 000123 的样本目录数量。"""
    if not target_dir.exists():
        return 0
    return sum(1 for p in target_dir.iterdir() if p.is_dir() and p.name.isdigit())


def _delete_numeric_dirs(target_dir: Path, dry_run: bool) -> int:
    """仅删除数字样本目录，避免误删其他文件。"""
    if not target_dir.exists():
        print(f"跳过：目录不存在 {target_dir}")
        return 0

    deleted = 0
    for p in sorted(target_dir.iterdir()):
        if not (p.is_dir() and p.name.isdigit()):
            continue
        if dry_run:
            print(f"[dry-run] 将删除: {p}")
        else:
            shutil.rmtree(p)
            print(f"已删除: {p}")
        deleted += 1
    return deleted


def main():
    parser = argparse.ArgumentParser(description="清理 unified 模仿学习数据")
    parser.add_argument(
        "--source",
        required=True,
        choices=list(SOURCE_TO_DIRS.keys()),
        help="清理目标：mario/coinrun/human_both/expert_mario/expert_coinrun/expert_both/all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览将删除的内容，不实际删除",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="确认执行删除（非 dry-run 时必须提供）",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.yes:
        raise SystemExit("安全保护：实际删除必须显式添加 --yes。")

    target_dirs = SOURCE_TO_DIRS[args.source]
    print(f"数据根目录: {DATA_ROOT}")
    print(f"逻辑 source: {args.source}")
    print(f"真实目录: {target_dirs}")

    total_before = 0
    for name in target_dirs:
        total_before += _count_numeric_dirs(DATA_ROOT / name)
    print(f"删除前样本数（数字目录统计）: {total_before}")

    total_deleted = 0
    for name in target_dirs:
        total_deleted += _delete_numeric_dirs(DATA_ROOT / name, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n[dry-run] 共将删除 {total_deleted} 个样本目录。")
    else:
        print(f"\n实际共删除 {total_deleted} 个样本目录。")


if __name__ == "__main__":
    main()
