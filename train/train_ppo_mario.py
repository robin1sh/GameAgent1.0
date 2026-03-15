"""
Mario 专用 PPO 训练入口。
作用：固定 --env mario，复用 train_ppo_model.py 的全部参数与逻辑。
"""
import os
import subprocess
import sys

# Mario 默认超参数（后续可直接在此微调）
MARIO_HPARAMS = [
    "--save-path", "./best_model/mario/",
    "--log-path", "./logs/mario/",
    "--callback-log-path", "./callback_logs/mario/",
    "--learning-rate", "1e-4",
    "--n-steps", "2048",
    "--batch-size", "4096",
    "--n-epochs", "5",
    "--ent-coef", "0.1",
    "--gamma", "0.97",
    "--target-kl", "0.01",
    "--clip-range", "0.15",
]


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_script = os.path.join(project_root, "train", "train_ppo_model.py")

    # 如果用户手动传了 --env，则去掉，避免与固定入口冲突
    passthrough = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--env":
            skip_next = True
            continue
        if arg.startswith("--env="):
            continue
        passthrough.append(arg)

    # 默认参数放前面，命令行显式传参放后面可覆盖默认值
    cmd = [sys.executable, target_script, "--env", "mario", *MARIO_HPARAMS, *passthrough]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
