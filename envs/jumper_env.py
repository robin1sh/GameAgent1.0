"""
Procgen Jumper 环境：84x84 RGB -> 84x84 灰度，与马里奥统一观测空间。
Procgen 返回 VecEnv，需用 VecEnv 包装器做观测预处理。
"""
import numpy as np
from PIL import Image

try:
    import gym
    from gym import spaces
except ImportError:
    import gymnasium as gym
    from gymnasium import spaces

# Procgen 可选：若未安装则 jumper 相关功能不可用
try:
    from procgen import ProcgenEnv
    PROCGEN_AVAILABLE = True
except ImportError:
    ProcgenEnv = None
    PROCGEN_AVAILABLE = False


OBS_SIZE = (84, 84)
SKIP_FRAMES = 4


def _extract_rgb(obs):
    """Procgen 可能返回 dict，提取 rgb 数组。"""
    if isinstance(obs, dict) and "rgb" in obs:
        return obs["rgb"]
    return obs


def _grayscale_resize(obs, target_size=(84, 84)):
    """将 RGB 观测转为灰度并缩放到 target_size。支持单帧或批量。"""
    obs = _extract_rgb(obs)
    if obs.ndim == 3:
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        img = Image.fromarray(gray.astype(np.uint8))
        out = np.array(img.resize((target_size[1], target_size[0])))
        return out[:, :, np.newaxis]
    else:
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        out = np.zeros((obs.shape[0], target_size[0], target_size[1], 1), dtype=np.uint8)
        for i in range(obs.shape[0]):
            img = Image.fromarray(gray[i].astype(np.uint8))
            out[i, :, :, 0] = np.array(img.resize((target_size[1], target_size[0])))
        return out


class ProcgenObsWrapper:
    """
    包装 Procgen VecEnv，将观测统一为 84x84 灰度，并执行与 Mario 一致的跳帧。
    需在 VecFrameStack 之前使用。
    """

    def __init__(self, venv, obs_size=OBS_SIZE, skip_frames=SKIP_FRAMES):
        self.venv = venv
        self.obs_size = obs_size
        self.skip_frames = max(1, int(skip_frames))
        self.num_envs = venv.num_envs
        self._last_actions = None
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(obs_size[0], obs_size[1], 1),
            dtype=np.uint8
        )
        self.action_space = venv.action_space

    def _process_obs_infos(self, obs, infos):
        """统一做灰度缩放，并把彩色帧放入 info，供可视化/采集复用。"""
        rgb_raw = _extract_rgb(obs)
        self._last_rgb = rgb_raw[0] if rgb_raw.ndim == 4 else rgb_raw
        obs = _grayscale_resize(obs, self.obs_size)
        processed_infos = []
        for i in range(len(infos)):
            info = infos[i] if infos[i] is not None else {}
            info = dict(info) if isinstance(info, dict) else {}
            frame = rgb_raw[i] if rgb_raw.ndim == 4 else rgb_raw
            info["rgb"] = np.ascontiguousarray(frame)
            processed_infos.append(info)
        return obs, processed_infos

    def _repeat_step(self, actions, first_result=None):
        """
        与 Mario 的 SkipFrameWrapper 对齐：
        - 同一动作重复执行 skip_frames 次
        - 奖励在跳帧窗口内累积
        - 某个子环境一旦 done，本窗口内不再继续累计该环境的后续奖励
        """
        total_rewards = np.zeros(self.num_envs, dtype=np.float32)
        done_mask = np.zeros(self.num_envs, dtype=bool)
        final_dones = np.zeros(self.num_envs, dtype=bool)
        final_infos = [None] * self.num_envs
        final_obs = None

        for step_idx in range(self.skip_frames):
            if step_idx == 0 and first_result is not None:
                result = first_result
            else:
                result = self.venv.step(actions)

            obs, rewards, dones, infos = result[:4]
            if isinstance(obs, tuple):
                obs = obs[0]

            rewards = np.asarray(rewards, dtype=np.float32)
            dones = np.asarray(dones, dtype=bool)
            active_mask = ~done_mask

            if final_obs is None:
                # Procgen 可能返回 {"rgb": ...}，不能直接 np.array(obs)，否则会变成 0 维 object 数组
                if isinstance(obs, dict):
                    final_obs = {key: np.array(value, copy=True) for key, value in obs.items()}
                else:
                    final_obs = np.array(obs, copy=True)
            else:
                if isinstance(obs, dict):
                    for key, value in obs.items():
                        final_obs[key][active_mask] = value[active_mask]
                else:
                    final_obs[active_mask] = obs[active_mask]

            total_rewards[active_mask] += rewards[active_mask]
            final_dones[active_mask] = dones[active_mask]

            for i in range(self.num_envs):
                if active_mask[i]:
                    info = infos[i] if infos[i] is not None else {}
                    final_infos[i] = dict(info) if isinstance(info, dict) else {}

            done_mask = np.logical_or(done_mask, dones)
            if np.all(done_mask):
                break

        processed_obs, processed_infos = self._process_obs_infos(final_obs, final_infos)
        return processed_obs, total_rewards, final_dones, processed_infos

    def reset(self):
        obs = self.venv.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        infos = [{} for _ in range(self.num_envs)]
        obs, _ = self._process_obs_infos(obs, infos)
        return obs

    def step(self, actions):
        return self._repeat_step(actions)

    def close(self):
        self.venv.close()

    def env_is_wrapped(self, *args, **kwargs):
        return self.venv.env_is_wrapped(*args, **kwargs)

    def get_attr(self, *args, **kwargs):
        return self.venv.get_attr(*args, **kwargs)

    def set_attr(self, *args, **kwargs):
        return self.venv.set_attr(*args, **kwargs)

    def step_async(self, actions):
        self._last_actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        first_result = self.venv.step_wait()
        return self._repeat_step(self._last_actions, first_result=first_result)


def make_jumper_vec_env(
    n_envs=10,
    fixed_level=False,
    start_level=0,
    num_levels=None,
    distribution_mode="easy",
    skip_frames=SKIP_FRAMES,
    **procgen_kwargs,
):
    """
    创建 Procgen Jumper 的 VecEnv，观测已统一为 84x84 灰度。
    默认每个动作重复执行 4 帧，与 Mario 训练节奏保持一致。
    返回的 VecEnv 可直接接 VecFrameStack。

    :param n_envs: 并行环境数
    :param fixed_level: 若为 True，使用固定关卡（num_levels=1），不随机关卡
    :param start_level: 固定关卡 ID（fixed_level=True 时生效）
    :param num_levels: 可选关卡数量（fixed_level=False 时可传）
    :param distribution_mode: Procgen 难度分布（默认 easy）
    :param skip_frames: 动作重复执行帧数，默认 4
    :param procgen_kwargs: 其他 ProcgenEnv 参数，如 rand_seed 等
    """
    if not PROCGEN_AVAILABLE:
        raise ImportError("procgen 未安装，无法使用 jumper 环境。请运行: pip install procgen")
    kwargs = dict(procgen_kwargs)
    kwargs.setdefault("distribution_mode", distribution_mode)
    if fixed_level:
        # 固定关卡核心参数：起始关卡 + 仅 1 个关卡
        kwargs["start_level"] = int(start_level)
        kwargs["num_levels"] = 1
    elif num_levels is not None:
        kwargs["num_levels"] = int(num_levels)
    venv = ProcgenEnv(num_envs=n_envs, env_name="jumper", **kwargs)
    venv = ProcgenObsWrapper(venv, obs_size=OBS_SIZE, skip_frames=skip_frames)
    return venv
