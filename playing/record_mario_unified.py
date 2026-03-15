"""
Mario 统一格式采集：4 帧堆叠 + 统一动作 (0-14)。
输出到 data_imitation_unified/mario/，每样本一个目录：frame_0.png ~ frame_3.png + label.txt

键盘操作（由 nes_py / gym_super_mario_bros 提供）：
- 方向键 ← → ↑ ↓：移动
- A 键（跳跃）、B 键（加速）：具体按键因 nes_py 版本而异，运行后以游戏窗口为准
- ESC：退出
"""
import os
import sys
from collections import deque

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
from nes_py.app.play_human import play_human

from envs.mario_env import make_mario_env_for_recording
from envs.action_mapping import MARIO_TO_UNIFIED

OUT_DIR = os.path.join(PROJECT_ROOT, "data_imitation_unified", "mario")
SAVE_EVERY = 2  # 每 N 步保存一次，避免冗余
RIGHT_SAVE_EVERY = 4  # right 动作采样更稀疏


def _frame_to_gray_64(obs):
    """将观测转为 64x64 灰度，用于保存 PNG。"""
    if obs.ndim == 3:
        if obs.shape[-1] == 1:
            return obs[:, :, 0]
        return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    return obs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    env = make_mario_env_for_recording()

    frame_buffer = deque(maxlen=33)  # 存最近 33 帧，用于每 8 帧取一帧组成 4 帧
    sample_id = 0

    def callback(state, action, reward, done, next_state):
        nonlocal sample_id
        # next_state 为 step 后的观测，需转为 64x64 灰度
        gray = _frame_to_gray_64(next_state)
        if gray.shape != (64, 64):
            gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        frame_buffer.append(gray.copy())

        if done:
            frame_buffer.clear()
            return

        if len(frame_buffer) < 25:
            return

        # Mario 原生动作 0-11 -> 统一动作 0-14
        unified_action = int(MARIO_TO_UNIFIED[action])

        # 稀疏采样：right(1) 和 noop(0) 降低频率
        if action == 1 and sample_id % RIGHT_SAVE_EVERY != 0:
            return
        if action == 0 and sample_id % SAVE_EVERY != 0:
            return

        sample_id += 1
        sample_dir = os.path.join(OUT_DIR, f"{sample_id:06d}")
        os.makedirs(sample_dir, exist_ok=True)
        # 取 t-24, t-16, t-8, t 四帧（每 8 帧一帧，与 PPO 一致）
        frames_8step = [frame_buffer[-25], frame_buffer[-17], frame_buffer[-9], frame_buffer[-1]]
        for i, f in enumerate(frames_8step):
            cv2.imwrite(os.path.join(sample_dir, f"frame_{i}.png"), f)
        with open(os.path.join(sample_dir, "label.txt"), "w") as f:
            f.write(str(unified_action))

    print(f"采集数据保存到: {OUT_DIR}")
    print("使用键盘操作，关闭窗口退出。每样本保存 4 帧 + 统一动作标签。")
    play_human(env, callback=callback)


if __name__ == "__main__":
    main()
