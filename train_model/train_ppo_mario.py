"""
Mario 专用 PPO 训练入口。
作用：固定 --env mario，复用 train_ppo_model.py 的全部参数与逻辑。
"""
import os
import subprocess
import sys #读 sys.argv（命令行参数）、用 sys.executable（当前 Python 解释器路径）。

# Mario 默认超参数（后续可直接在此微调）
MARIO_HPARAMS = [
    "--save-path", "./best_model/mario/",
    "--log-path", "./logs/mario/",
    "--callback-log-path", "./callback_logs/mario/",
    "--learning-rate", "3e-4",
    "--n-steps", "2048",#每次更新前每个环境要收集的步数（rollout 长度相关，和并行环境数一起影响一次更新的样本量）。
    "--batch-size", "2048",#优化时的 mini-batch 大小（需满足与 n_steps、环境数等的整除关系，具体以 train_ppo_model.py 校验为准）。
    "--n-epochs", "10",#每个样本要重复训练的次数（和 n_steps 一起决定一次更新的样本量）。
    "--ent-coef", "0.1",#熵系数，鼓励探索；越大策略越随机，可能减慢收敛或帮助跳出局部策略。
    "--gamma", "0.95",#回报折扣因子，越大越看重后期回报。
]
#以上为超参数列表

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_script = os.path.join(project_root, "train_model", "train_ppo_model.py")

    """target_script：先算出项目根目录，再拼出 train_model/train_ppo_model.py 的绝对路径。
    cmd：构造一条等价于下面的命令：
    cmd = [sys.executable, target_script, "--env", "mario", *MARIO_HPARAMS, *passthrough]
    subprocess.call(cmd)：真正启动子进程执行该脚本。
    sys.executable：保证用你当前这个 Python 环境来跑，避免环境不一致。
    """
    # 如果用户手动传了 --env，则去掉，避免与固定入口冲突
    passthrough = []    #用户传入的命令行参数（不包括 --env 和 --env= 开头的参数）。
    skip_next = False   #是否跳过下一个参数（处理 --env 和 --env= 特殊情况）。
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
#train_ppo_mario.py 只是一个“Mario 专用启动器”，真正训练逻辑都在 train_ppo_model.py