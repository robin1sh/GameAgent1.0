"""
动作空间统一：将 15 维统一动作映射到各环境原生动作空间。
Procgen 使用 15 动作，Mario COMPLEX_MOVEMENT 使用 12 动作。
"""
import gym
from gym import spaces
import numpy as np

# Procgen 动作组合 (get_combos): 0-14
# 0:(LEFT,DOWN), 1:(LEFT), 2:(LEFT,UP), 3:(DOWN), 4:(), 5:(UP), 6:(RIGHT,DOWN),
# 7:(RIGHT), 8:(RIGHT,UP), 9:(D), 10:(A), 11:(W), 12:(S), 13:(Q), 14:(E)

# Mario COMPLEX_MOVEMENT 顺序: [NOOP, right, right+A, right+B, right+A+B, A, left,
#   left+A, left+B, left+A+B, down, up]
# 索引: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11

# 统一动作(0-14) -> Mario COMPLEX_MOVEMENT(0-11) 映射
"""
MARIO_ACTION_MAP: （统一 → Mario)
含义	统一动作 (0-14) → Mario COMPLEX_MOVEMENT (0-11)
用途	推理/执行:PPO 输出 0-14,需要转成 Mario 的 0-11 再 env.step()
方向	统一动作 → Mario 动作
示例	统一 7 (RIGHT) → Mario 1 (right)
MARIO_TO_UNIFIED:(Mario → 统一）
含义	Mario COMPLEX_MOVEMENT (0-11) → 统一动作 (0-14)
用途	采集：人类玩 Mario 时得到的是 Mario 的 0-11,需要转成统一 0-14 再保存
方向	Mario 动作 → 统一动作
示例	Mario 1 (right) → 统一 7 (RIGHT)

"""
MARIO_ACTION_MAP = np.array([
    8,   # 0: (LEFT,DOWN) -> left+B
    6,   # 1: (LEFT) -> left
    7,   # 2: (LEFT,UP) -> left+A
    10,  # 3: (DOWN) -> down
    0,   # 4: () -> noop
    11,  # 5: (UP) -> up（管道入口等）
    3,   # 6: (RIGHT,DOWN) -> right+B
    1,   # 7: (RIGHT) -> right
    2,   # 8: (RIGHT,UP) -> right+A
    0,   # 9: (D) -> noop
    5,   # 10: (A) -> jump
    5,   # 11: (W) -> jump
    10,  # 12: (S) -> down
    6,   # 13: (Q) -> left
    1,   # 14: (E) -> right
], dtype=np.int32)

# Mario COMPLEX_MOVEMENT 索引(0-11) -> 统一动作(0-14) 反向映射，用于采集时转换
MARIO_TO_UNIFIED = np.array([4, 7, 8, 6, 8, 10, 1, 2, 0, 2, 3, 5], dtype=np.int32)


class ActionMappingWrapper(gym.ActionWrapper):
    """
    将统一 15 维动作映射到各环境原生动作空间。
    策略网络输出 0-14，此 wrapper 负责映射到环境原生动作。
    env_type: "mario" | "jumper"
    """

    UNIFIED_ACTION_DIM = 15

    def __init__(self, env, env_type: str):
        super().__init__(env)
        self.env_type = env_type
        if env_type == "mario":
            self._action_map = MARIO_ACTION_MAP
            self.action_space = spaces.Discrete(self.UNIFIED_ACTION_DIM)
        elif env_type == "jumper":
            self._action_map = None
            self.action_space = spaces.Discrete(self.UNIFIED_ACTION_DIM)
        else:
            raise ValueError(f"未知 env_type: {env_type}")

    def action(self, action):
        if self._action_map is None:
            return int(np.clip(np.asarray(action).item(), 0, 14))
        action = int(np.asarray(action).item())
        action = np.clip(action, 0, 14)
        return int(self._action_map[action])
