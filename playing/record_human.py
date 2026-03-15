import os
import time
import cv2
import gym_super_mario_bros
from nes_py.app.play_human import play_human
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def main():
    env = gym_super_mario_bros.make("SuperMarioBros-v2")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    out_dir = "data_imitation"
    images_root = os.path.join(out_dir, "images")
    os.makedirs(images_root, exist_ok=True)

    action_names = env.get_action_meanings()
    name_map = {
        "NOOP": "noop",
        "A": "jump",
        "B": "run",
        "right": "right",
        "left": "left",
        "right A": "right_jump",
        "right B": "right_run",
        "right A B": "right_run_jump",
    }
    action_dirs = {}
    for name in action_names:
        safe_name = name_map.get(name, name.replace(" ", "_").lower())
        dir_path = os.path.join(images_root, safe_name)
        os.makedirs(dir_path, exist_ok=True)
        action_dirs[name] = dir_path

    save_every = 1
    right_save_every = 3  # 向右样本保存频率（值越大保存越少）
    noop_interval_sec = 30
    last_noop_time = 0.0

    frame_idx = 0

    def callback(state, action, reward, done, next_state):
        nonlocal frame_idx
        nonlocal last_noop_time
        if done:
            frame_idx += 1
            return
        if frame_idx % save_every != 0:
            frame_idx += 1
            return
        action_name = action_names[action]
        # 不保存向左走
        if action_name == "left":
            frame_idx += 1
            return
        # 降低“right”样本采样频率，避免数据过于偏向向右
        if action_name == "right" and frame_idx % right_save_every != 0:
            frame_idx += 1
            return
        if action_name == "NOOP":
            now = time.time()
            if now - last_noop_time < noop_interval_sec:
                frame_idx += 1
                return
            last_noop_time = now
        img_path = os.path.join(action_dirs[action_name], f"{frame_idx:06d}.png")
        gray = cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        gray = gray[:, :, None]  # 保留 1 通道
        cv2.imwrite(img_path, gray)
        frame_idx += 1

    play_human(env, callback=callback)


if __name__ == "__main__":
    main()
