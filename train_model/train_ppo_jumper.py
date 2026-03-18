"""
Jumper 专用 PPO 训练入口。
作用：固定 --env jumper，复用 train_ppo_model.py 的全部参数与逻辑。
"""
import os
import subprocess
import sys

# Jumper 默认超参数（后续可直接在此微调）
JUMPER_HPARAMS = [
    "--save-path", "./best_model/jumper/",
    "--log-path", "./logs/jumper/",
    "--callback-log-path", "./callback_logs/jumper/",
    "--learning-rate", "7e-5",
    "--n-steps", "1024",
    "--batch-size", "4096",
    "--n-epochs", "8",
    "--ent-coef", "0.05",
    "--gamma", "0.99",
]


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_script = os.path.join(project_root, "train_model", "train_ppo_model.py")

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
    cmd = [sys.executable, target_script, "--env", "jumper", *JUMPER_HPARAMS, *passthrough]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
