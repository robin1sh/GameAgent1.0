"""
CoinRun 专用 PPO 训练入口。
作用：固定 --env coinrun，复用 train_ppo_model.py 的全部参数与逻辑。
"""
import os
import subprocess
import sys

# CoinRun 默认超参数（先复用 Jumper 的稳定配置，后续可按实验结果微调）
COINRUN_HPARAMS = [
    "--save-path", "./best_model/coinrun/",
    "--log-path", "./logs/coinrun/",
    "--callback-log-path", "./callback_logs/coinrun/",
    "--learning-rate", "3e-4",
    "--n-steps", "2048",
    "--batch-size", "2048",
    "--n-epochs", "8",
    "--ent-coef", "0.1",
    "--gamma", "0.95",
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
    cmd = [sys.executable, target_script, "--env", "coinrun", *COINRUN_HPARAMS, *passthrough]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
