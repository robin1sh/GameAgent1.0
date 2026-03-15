"""
根据配置文件批量运行实验组 A/B/C。
用法: python scripts/run_experiments.py configs/exp_a1.yaml
      python scripts/run_experiments.py configs/exp_a1.yaml configs/exp_a2.yaml
"""
import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_config(path):
    with open(path) as f:
        content = f.read()
    # 简单解析：支持 # 注释和 key: value
    config = {}
    for line in content.splitlines():
        line = line.split("#")[0].strip()
        if ":" in line and not line.startswith("-"):
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if v.lower() in ("true", "false"):
                v = v.lower() == "true"
            elif v.isdigit():
                v = int(v)
            elif v.replace(".", "").isdigit():
                v = float(v)
            config[k] = v
    return config


def run_experiment(config_path):
    config = load_config(config_path)
    exp_id = config.get("exp_id", "exp")
    env = config.get("env", "mario")
    mode = config.get("mode", "")
    total = config.get("total_timesteps", 10_000_000)
    n_envs = config.get("n_envs", 10)
    save_path = config.get("save_path", "./best_model/")
    log_path = config.get("log_path", "./logs/")

    os.chdir(PROJECT_ROOT)

    if mode in ("alternating", "mixed"):
        cmd = [
            sys.executable, "train/train_unified.py",
            "--mode", mode,
            "--exp-id", exp_id,
            "--total-timesteps", str(total),
            "--n-envs", str(n_envs),
            "--save-path", save_path,
            "--log-path", log_path,
        ]
        if mode == "alternating":
            cmd.extend(["--alternate-rounds", str(config.get("alternate_rounds", 1))])
    else:
        cmd = [
            sys.executable, "train/train_ppo_model.py",
            "--env", env,
            "--exp-id", exp_id,
            "--total-timesteps", str(total),
            "--n-envs", str(n_envs),
            "--save-path", save_path,
            "--log-path", log_path,
        ]
        if config.get("fixed_level", False):
            cmd.append("--fixed-level")

    print(f"运行实验: {exp_id}")
    print("命令:", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="配置文件路径")
    args = parser.parse_args()

    for cfg in args.configs:
        cfg = os.path.join(PROJECT_ROOT, cfg) if not os.path.isabs(cfg) else cfg
        if not os.path.exists(cfg):
            print(f"配置文件不存在: {cfg}")
            sys.exit(1)
        ret = run_experiment(cfg)
        if ret != 0:
            sys.exit(ret)


if __name__ == "__main__":
    main()
