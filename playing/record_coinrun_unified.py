"""
CoinRun 统一格式采集：4 帧堆叠 + 统一动作 (0-14)。
CoinRun 与 Jumper 同属 Procgen，统一动作直接映射为 Procgen 原生 15 动作。
输出到 data_imitation_unified/coinrun/

用法：python playing/record_coinrun_unified.py
  默认固定关卡（第一关），可用 --no-fixed-level 切换为随机关卡
  默认自动续写样本编号，不覆盖旧数据

键盘操作：
- 方向键 ← → ↑ ↓：移动
- 空格 / W：跳跃
- A 键：左、D 键：右、S 键：下
- 关闭窗口：退出
"""
import argparse
import json
import os
import sys
from collections import deque

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import pygame
import numpy as np

from envs.coinrun_env import make_coinrun_vec_env, PROCGEN_AVAILABLE

OUT_DIR = os.path.join(PROJECT_ROOT, "data_imitation_unified", "coinrun")
LEVEL_CONFIG_PATH = os.path.join(OUT_DIR, "level_config.json")
SAVE_EVERY = 2
IDLE_SAVE_EVERY = 8


def _get_action_from_keys(keys):
    """根据当前按键组合返回统一动作。"""
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    up = keys[pygame.K_UP]
    jump = keys[pygame.K_SPACE] or keys[pygame.K_w]

    if left and down:
        return 0
    if right and down:
        return 6
    if left and (up or jump):
        return 2
    if right and (up or jump):
        return 8
    if up:
        return 5
    if jump:
        return 10
    if left:
        return 1
    if right:
        return 7
    if down:
        return 3
    return 4


def _get_start_sample_id(out_dir):
    """从现有目录中恢复样本编号，避免覆盖历史数据。"""
    max_id = 0
    for name in os.listdir(out_dir):
        if name.isdigit():
            max_id = max(max_id, int(name))
    return max_id


def _save_level_config(path, fixed_level, start_level, distribution_mode):
    """保存本次采集关卡配置，供训练脚本复用。"""
    payload = {
        "env": "coinrun",
        "fixed_level": bool(fixed_level),
        "start_level": int(start_level),
        "distribution_mode": str(distribution_mode),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="CoinRun 统一格式采集")
    parser.add_argument(
        "--fixed-level",
        action="store_true",
        default=True,
        help="使用固定关卡，不随机关卡（默认开启）",
    )
    parser.add_argument(
        "--no-fixed-level",
        action="store_false",
        dest="fixed_level",
        help="关闭固定关卡，使用随机关卡",
    )
    parser.add_argument("--start-level", type=int, default=0, help="固定关卡 ID（第一关为 0）")
    parser.add_argument("--distribution-mode", default="easy", help="Procgen 难度分布，建议 easy")
    parser.add_argument(
        "--save-every",
        type=int,
        default=SAVE_EVERY,
        help="非 idle 动作每 N 步保存一次（默认 2）",
    )
    parser.add_argument(
        "--idle-save-every",
        type=int,
        default=IDLE_SAVE_EVERY,
        help="idle(动作4) 每 N 步保存一次（默认 8）",
    )
    args = parser.parse_args()

    if not PROCGEN_AVAILABLE:
        print("请先安装 procgen: pip install procgen")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    _save_level_config(
        LEVEL_CONFIG_PATH,
        fixed_level=args.fixed_level,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode,
    )
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("CoinRun - 键盘操作采集")

    # 采集时保持正常帧率，不在环境层做跳帧；再手动按 4 帧间隔取样，与 Mario 采集保持一致。
    venv = make_coinrun_vec_env(
        n_envs=1,
        fixed_level=args.fixed_level,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode,
        skip_frames=1,
    )
    obs = venv.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = obs[0] if obs.ndim == 4 else obs
    gray_init = obs[:, :, 0] if obs.ndim == 3 else obs
    rgb_init = getattr(venv, "_last_rgb", None)

    frame_buffer = deque(maxlen=13)
    sample_id = _get_start_sample_id(OUT_DIR)
    step_count = 0
    clock = pygame.time.Clock()
    running = True

    def _render_obs(disp, use_rgb=True):
        """将 84x84 观测放大并渲染到 pygame 窗口。disp: RGB(H,W,3) 或灰度(H,W)。"""
        if use_rgb and disp.ndim == 3 and disp.shape[-1] == 3:
            rgb = disp
        else:
            gray = disp[:, :, 0] if disp.ndim == 3 else disp
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        scaled = cv2.resize(rgb, (400, 400), interpolation=cv2.INTER_NEAREST)
        try:
            arr = np.ascontiguousarray(np.transpose(scaled, (1, 0, 2)))
            surf = pygame.surfarray.make_surface(arr)
        except Exception:
            surf = pygame.image.frombuffer(scaled.tobytes(), (400, 400), "RGB")
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    _render_obs(rgb_init if rgb_init is not None else gray_init, use_rgb=(rgb_init is not None))
    print(f"采集数据保存到: {OUT_DIR}")
    print(f"关卡配置已保存: {LEVEL_CONFIG_PATH}")
    print(f"当前起始样本编号: {sample_id + 1:06d}")
    if args.fixed_level:
        print(f"当前模式：固定关卡（start_level={args.start_level}，每次 reset 回到同一关）")
    print(
        f"采样频率：非idle每 {args.save_every} 步保存，"
        f"idle每 {args.idle_save_every} 步保存"
    )
    print("方向键移动，空格/W 跳跃。关闭窗口退出。")

    waiting = True
    while waiting and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                waiting = False
                break
            if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                waiting = False
                break
        pygame.event.pump()
        clock.tick(30)

    if running:
        print("已开始，使用方向键 + 空格/W 控制角色。")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if not running:
            break

        pygame.event.pump()
        keys = pygame.key.get_pressed()
        action = _get_action_from_keys(keys)

        result = venv.step(np.array([action]))
        obs, rewards, dones, infos = result[:4]
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = obs[0]
        gray = obs[:, :, 0] if obs.ndim == 3 else obs
        frame_buffer.append(gray.copy())
        rgb = infos[0].get("rgb", None) if infos and len(infos) > 0 else None
        _render_obs(rgb if rgb is not None else gray, use_rgb=(rgb is not None))

        if dones[0]:
            obs = venv.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            obs = obs[0] if obs.ndim == 4 else obs
            gray = obs[:, :, 0] if obs.ndim == 3 else obs
            frame_buffer.clear()
            frame_buffer.append(gray.copy())
            rgb = getattr(venv, "_last_rgb", None)
            _render_obs(rgb if rgb is not None else gray, use_rgb=(rgb is not None))
            clock.tick(30)
            continue

        step_count += 1
        if len(frame_buffer) >= 13:
            if action == 4:
                should_save = (step_count % max(1, args.idle_save_every) == 0)
            else:
                should_save = (step_count % max(1, args.save_every) == 0)
        else:
            should_save = False

        if should_save:
            sample_id += 1
            sample_dir = os.path.join(OUT_DIR, f"{sample_id:06d}")
            os.makedirs(sample_dir, exist_ok=False)
            frames_4step = [frame_buffer[-13], frame_buffer[-9], frame_buffer[-5], frame_buffer[-1]]
            for i, frame in enumerate(frames_4step):
                cv2.imwrite(os.path.join(sample_dir, f"frame_{i}.png"), frame)
            with open(os.path.join(sample_dir, "label.txt"), "w") as f:
                f.write(str(action))

        clock.tick(15)

    venv.close()
    pygame.quit()


if __name__ == "__main__":
    main()
