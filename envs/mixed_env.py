"""
联合训练环境：混合 Mario 与 CoinRun，用于实验组 B。
"""
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, VecFrameStack

from .mario_env import make_mario_env
from .coinrun_env import make_coinrun_vec_env, PROCGEN_AVAILABLE


class MixedVecEnv(VecEnv):
    """
    将 mario 与 coinrun 的 VecEnv 拼接为一个 VecEnv。
    前 n_mario 个 env 为 mario，后 n_coinrun 个 env 为 coinrun。
    """

    def __init__(
        self,
        mario_vec,
        coinrun_vec,
        coinrun_reward_scale=1.0,
        coinrun_progress_coef=0.05,
        coinrun_success_bonus=10.0,
        coinrun_fail_penalty=2.0,
        coinrun_step_penalty=0.002,
        use_aligned_reward=False,
        progress_coef=0.02,
        success_bonus=100.0,
        fail_penalty=20.0,
        step_penalty=0.002,
        max_episode_steps=None,
    ):
        n_mario = mario_vec.num_envs
        n_coinrun = coinrun_vec.num_envs
        n_total = n_mario + n_coinrun
        obs_space = mario_vec.observation_space
        act_space = mario_vec.action_space
        super().__init__(n_total, obs_space, act_space)
        self.mario_vec = mario_vec
        self.coinrun_vec = coinrun_vec
        self.n_mario = n_mario
        self.n_coinrun = n_coinrun
        # 统一奖励尺度：缩放 CoinRun 奖励，缓解多任务量级失衡
        self.coinrun_reward_scale = float(coinrun_reward_scale)
        # CoinRun 专用塑形参数（仅在 use_aligned_reward=False 时启用）
        self.coinrun_progress_coef = float(coinrun_progress_coef)
        self.coinrun_success_bonus = float(coinrun_success_bonus)
        self.coinrun_fail_penalty = float(coinrun_fail_penalty)
        self.coinrun_step_penalty = float(coinrun_step_penalty)
        # 记录每个 CoinRun 子环境上一时刻的横向位置，用于计算右移增量奖励
        self._coinrun_prev_x = np.full(self.n_coinrun, np.nan, dtype=np.float32)
        # 统一奖励语义：前进（可选）+ 通关 - 失败 - 时间
        self.use_aligned_reward = bool(use_aligned_reward)
        self.progress_coef = float(progress_coef)
        self.success_bonus = float(success_bonus)
        self.fail_penalty = float(fail_penalty)
        self.step_penalty = float(step_penalty)
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)
        self._mario_episode_steps = np.zeros(self.n_mario, dtype=np.int32)
        self._coinrun_episode_steps = np.zeros(self.n_coinrun, dtype=np.int32)
        self._preprocess_checked = False

        # 启动时先检查空间定义，尽早发现两环境预处理配置不一致问题。
        self._validate_preprocess_spaces()

    @staticmethod
    def _action_spaces_match(space_a, space_b):
        """尽量稳健地比较动作空间，避免仅比较对象地址。"""
        if type(space_a) is not type(space_b):
            return False
        if hasattr(space_a, "n") and hasattr(space_b, "n"):
            return int(space_a.n) == int(space_b.n)
        if hasattr(space_a, "shape") and hasattr(space_b, "shape"):
            return tuple(space_a.shape) == tuple(space_b.shape)
        return str(space_a) == str(space_b)

    def _validate_preprocess_spaces(self):
        """检查 Mario/CoinRun 的观测空间和动作空间是否可安全拼接。"""
        mario_obs_space = self.mario_vec.observation_space
        coinrun_obs_space = self.coinrun_vec.observation_space
        if mario_obs_space.shape != coinrun_obs_space.shape:
            raise ValueError(
                f"混合训练预处理不一致：observation shape 不匹配，"
                f"mario={mario_obs_space.shape}, coinrun={coinrun_obs_space.shape}"
            )
        if mario_obs_space.dtype != coinrun_obs_space.dtype:
            raise ValueError(
                f"混合训练预处理不一致：observation dtype 不匹配，"
                f"mario={mario_obs_space.dtype}, coinrun={coinrun_obs_space.dtype}"
            )
        if len(mario_obs_space.shape) != 3:
            raise ValueError(
                f"混合训练预处理不符合预期：观测应为 (H, W, C)，当前 mario shape={mario_obs_space.shape}"
            )
        if mario_obs_space.shape[-1] <= 0:
            raise ValueError(
                f"混合训练预处理不符合预期：通道数必须大于 0，当前 C={mario_obs_space.shape[-1]}"
            )
        mario_low = np.asarray(mario_obs_space.low)
        mario_high = np.asarray(mario_obs_space.high)
        coinrun_low = np.asarray(coinrun_obs_space.low)
        coinrun_high = np.asarray(coinrun_obs_space.high)
        if mario_low.shape != mario_obs_space.shape and mario_low.shape != ():
            raise ValueError(
                f"mario 观测空间 low 形状异常：low.shape={mario_low.shape}, obs.shape={mario_obs_space.shape}"
            )
        if mario_high.shape != mario_obs_space.shape and mario_high.shape != ():
            raise ValueError(
                f"mario 观测空间 high 形状异常：high.shape={mario_high.shape}, obs.shape={mario_obs_space.shape}"
            )
        if coinrun_low.shape != coinrun_obs_space.shape and coinrun_low.shape != ():
            raise ValueError(
                f"coinrun 观测空间 low 形状异常：low.shape={coinrun_low.shape}, obs.shape={coinrun_obs_space.shape}"
            )
        if coinrun_high.shape != coinrun_obs_space.shape and coinrun_high.shape != ():
            raise ValueError(
                f"coinrun 观测空间 high 形状异常：high.shape={coinrun_high.shape}, obs.shape={coinrun_obs_space.shape}"
            )
        if not self._action_spaces_match(self.action_space, self.coinrun_vec.action_space):
            raise ValueError(
                f"混合训练动作空间不一致：mario={self.action_space}, "
                f"coinrun={self.coinrun_vec.action_space}"
            )

    @staticmethod
    def _validate_runtime_obs(mario_obs, coinrun_obs):
        """首次 reset 时检查真实观测张量，确认预处理产物一致。"""
        if mario_obs.ndim != 4 or coinrun_obs.ndim != 4:
            raise ValueError(
                f"混合训练预处理不一致：runtime obs 维度应为 4，"
                f"mario.ndim={mario_obs.ndim}, coinrun.ndim={coinrun_obs.ndim}"
            )
        if mario_obs.shape[1:] != coinrun_obs.shape[1:]:
            raise ValueError(
                f"混合训练预处理不一致：runtime obs shape 不匹配，"
                f"mario={mario_obs.shape[1:]}, coinrun={coinrun_obs.shape[1:]}"
            )
        if mario_obs.dtype != coinrun_obs.dtype:
            raise ValueError(
                f"混合训练预处理不一致：runtime obs dtype 不匹配，"
                f"mario={mario_obs.dtype}, coinrun={coinrun_obs.dtype}"
            )
        mario_min, mario_max = float(np.min(mario_obs)), float(np.max(mario_obs))
        coinrun_min, coinrun_max = float(np.min(coinrun_obs)), float(np.max(coinrun_obs))
        if not np.isfinite(mario_min) or not np.isfinite(mario_max):
            raise ValueError("混合训练预处理异常：mario 观测包含 NaN/Inf。")
        if not np.isfinite(coinrun_min) or not np.isfinite(coinrun_max):
            raise ValueError("混合训练预处理异常：coinrun 观测包含 NaN/Inf。")
        # uint8 常见值域为 [0, 255]；若为浮点，常见值域为 [0, 1] 或 [0, 255]。
        if np.issubdtype(mario_obs.dtype, np.integer):
            if mario_min < 0.0 or mario_max > 255.0:
                raise ValueError(
                    f"混合训练预处理异常：mario uint 观测值域越界，min={mario_min}, max={mario_max}"
                )
        else:
            if mario_min < -1e-6 or mario_max > 255.0 + 1e-6:
                raise ValueError(
                    f"混合训练预处理异常：mario float 观测值域异常，min={mario_min}, max={mario_max}"
                )
        if np.issubdtype(coinrun_obs.dtype, np.integer):
            if coinrun_min < 0.0 or coinrun_max > 255.0:
                raise ValueError(
                    f"混合训练预处理异常：coinrun uint 观测值域越界，min={coinrun_min}, max={coinrun_max}"
                )
        else:
            if coinrun_min < -1e-6 or coinrun_max > 255.0 + 1e-6:
                raise ValueError(
                    f"混合训练预处理异常：coinrun float 观测值域异常，min={coinrun_min}, max={coinrun_max}"
                )

    @staticmethod
    def _coinrun_success(info):
        return bool(
            info.get("level_complete", False)
            or info.get("prev_level_complete", False)
            or info.get("carrot_get", False)
        )

    @staticmethod
    def _extract_coinrun_x(info):
        """
        从 CoinRun info 中尽量提取横向进度字段。
        不同 Procgen 版本字段名可能不同，故做多键兼容。
        """
        if not isinstance(info, dict):
            return None
        keys = (
            "x_pos",
            "x_position",
            "x",
            "agent_x",
            "player_x",
            "scroll_x",
        )
        for key in keys:
            value = info.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                return float(value)
        return None

    def _coinrun_shaped_reward(self, env_idx, raw_reward, done, info):
        """
        CoinRun 塑形奖励：
        1) 原始奖励缩放（兼容旧配置）
        2) 仅奖励向右位移（防止左右抖动刷分）
        3) 通关奖励 / 失败惩罚 / 每步时间惩罚
        """
        reward = float(raw_reward) * self.coinrun_reward_scale
        success = self._coinrun_success(info)
        x_now = self._extract_coinrun_x(info)
        if x_now is not None:
            x_prev = self._coinrun_prev_x[env_idx]
            if np.isfinite(x_prev):
                reward += max(0.0, x_now - x_prev) * self.coinrun_progress_coef
            self._coinrun_prev_x[env_idx] = float(x_now)
        reward -= self.coinrun_step_penalty
        if success:
            reward += self.coinrun_success_bonus
        elif bool(done):
            reward -= self.coinrun_fail_penalty
        if bool(done):
            self._coinrun_prev_x[env_idx] = np.nan
        return reward

    def _aligned_reward(self, raw_reward, done, success):
        """
        统一奖励模板（两环境同源）：
        - progress: max(raw_reward, 0) * progress_coef
        - success: +success_bonus
        - fail: -fail_penalty
        - step: -step_penalty
        """
        progress = max(0.0, float(raw_reward)) * self.progress_coef
        fail = bool(done and not success)
        shaped = progress - self.step_penalty
        if success:
            shaped += self.success_bonus
        if fail:
            shaped -= self.fail_penalty
        return shaped

    @staticmethod
    def _inject_game_id_channel(mario_obs, coinrun_obs):
        """
        在固定像素位注入 game_id，供双动作头策略路由使用：
        - Mario: 0
        - CoinRun: 255
        约定输入为 channels_last: (N, H, W, C)。
        """
        mario_obs[:, 0, 0, 0] = 0
        coinrun_obs[:, 0, 0, 0] = 255

    def reset(self):
        mario_obs = self.mario_vec.reset()
        coinrun_obs = self.coinrun_vec.reset()
        self._coinrun_prev_x.fill(np.nan)
        self._mario_episode_steps.fill(0)
        self._coinrun_episode_steps.fill(0)
        if isinstance(mario_obs, tuple):
            mario_obs = mario_obs[0]
        if isinstance(coinrun_obs, tuple):
            coinrun_obs = coinrun_obs[0]
        self._inject_game_id_channel(mario_obs, coinrun_obs)
        if not self._preprocess_checked:
            self._validate_runtime_obs(mario_obs, coinrun_obs)
            self._preprocess_checked = True
        return np.concatenate([mario_obs, coinrun_obs], axis=0)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        mario_actions = self._actions[: self.n_mario]
        coinrun_actions = self._actions[self.n_mario :]
        mario_result = self.mario_vec.step(mario_actions)
        coinrun_result = self.coinrun_vec.step(coinrun_actions)
        mario_obs, mario_rewards, mario_dones, mario_infos = mario_result[:4]
        coinrun_obs, coinrun_rewards, coinrun_dones, coinrun_infos = coinrun_result[:4]
        if isinstance(mario_obs, tuple):
            mario_obs = mario_obs[0]
        if isinstance(coinrun_obs, tuple):
            coinrun_obs = coinrun_obs[0]
        self._inject_game_id_channel(mario_obs, coinrun_obs)
        obs = np.concatenate([mario_obs, coinrun_obs], axis=0)
        if self.use_aligned_reward:
            aligned_mario_rewards = np.zeros_like(mario_rewards, dtype=np.float32)
            aligned_coinrun_rewards = np.zeros_like(coinrun_rewards, dtype=np.float32)
            for i in range(self.n_mario):
                info = mario_infos[i] if isinstance(mario_infos[i], dict) else {}
                success = bool(info.get("flag_get", False))
                # Mario 保持原始风格：仅前进项 + 通关奖励，不加失败/时间惩罚。
                progress = max(0.0, float(mario_rewards[i])) * self.progress_coef
                aligned_mario_rewards[i] = progress + (self.success_bonus if success else 0.0)
            for i in range(self.n_coinrun):
                info = coinrun_infos[i] if isinstance(coinrun_infos[i], dict) else {}
                success = self._coinrun_success(info)
                aligned_coinrun_rewards[i] = self._aligned_reward(
                    raw_reward=coinrun_rewards[i],
                    done=coinrun_dones[i],
                    success=success,
                )
            rewards = np.concatenate([aligned_mario_rewards, aligned_coinrun_rewards])
        else:
            shaped_coinrun_rewards = np.zeros_like(coinrun_rewards, dtype=np.float32)
            for i in range(self.n_coinrun):
                info = coinrun_infos[i] if isinstance(coinrun_infos[i], dict) else {}
                shaped_coinrun_rewards[i] = self._coinrun_shaped_reward(
                    env_idx=i,
                    raw_reward=coinrun_rewards[i],
                    done=coinrun_dones[i],
                    info=info,
                )
            rewards = np.concatenate([mario_rewards, shaped_coinrun_rewards])
        dones = np.concatenate([mario_dones, coinrun_dones])
        mario_infos = [dict(info) if isinstance(info, dict) else {} for info in mario_infos]
        coinrun_infos = [dict(info) if isinstance(info, dict) else {} for info in coinrun_infos]
        self._mario_episode_steps += 1
        self._coinrun_episode_steps += 1
        for info in mario_infos:
            info["game"] = "mario"
            info["use_aligned_reward"] = self.use_aligned_reward
            info["progress_coef"] = self.progress_coef
            info["success_bonus"] = self.success_bonus
            info["fail_penalty"] = self.fail_penalty
            info["step_penalty"] = self.step_penalty
        for idx, info in enumerate(mario_infos):
            if bool(mario_dones[idx]):
                success = bool(info.get("flag_get", False))
                timed_out = bool(
                    info.get("TimeLimit.truncated", False)
                    or (
                        self.max_episode_steps is not None
                        and self.max_episode_steps > 0
                        and self._mario_episode_steps[idx] >= self.max_episode_steps
                        and not success
                    )
                )
                info["done"] = True
                info["time_limit_truncated"] = timed_out
                info["episode_end_reason"] = "time_limit" if timed_out else ("success" if success else "fail")
                self._mario_episode_steps[idx] = 0
        for idx, info in enumerate(coinrun_infos):
            info["game"] = "coinrun"
            # 记录缩放信息，便于日志排查
            info["reward_scale"] = self.coinrun_reward_scale
            info["coinrun_progress_coef"] = self.coinrun_progress_coef
            info["coinrun_success_bonus"] = self.coinrun_success_bonus
            info["coinrun_fail_penalty"] = self.coinrun_fail_penalty
            info["coinrun_step_penalty"] = self.coinrun_step_penalty
            info["use_aligned_reward"] = self.use_aligned_reward
            info["progress_coef"] = self.progress_coef
            info["success_bonus"] = self.success_bonus
            info["fail_penalty"] = self.fail_penalty
            info["step_penalty"] = self.step_penalty
            if bool(coinrun_dones[idx]):
                success = self._coinrun_success(info)
                timed_out = bool(
                    info.get("TimeLimit.truncated", False)
                    or info.get("truncated", False)
                    or (
                        self.max_episode_steps is not None
                        and self.max_episode_steps > 0
                        and self._coinrun_episode_steps[idx] >= self.max_episode_steps
                        and not success
                    )
                )
                info["done"] = True
                info["time_limit_truncated"] = timed_out
                info["episode_end_reason"] = "time_limit" if timed_out else ("success" if success else "fail")
                self._coinrun_episode_steps[idx] = 0
        infos = list(mario_infos) + list(coinrun_infos)
        return obs, rewards, dones, infos

    def seed(self, seed=None):
        """为混合环境中的各子环境分别设置随机种子。"""
        mario_seeds = self.mario_vec.seed(seed)
        coinrun_seed = None if seed is None else seed + self.n_mario
        coinrun_seeds = self.coinrun_vec.seed(coinrun_seed)
        return list(mario_seeds) + list(coinrun_seeds)

    def close(self):
        self.mario_vec.close()
        self.coinrun_vec.close()

    def get_attr(self, attr_name, indices=None):
        if indices is None:
            indices = list(range(self.num_envs))
        results = []
        for i in indices:
            if i < self.n_mario:
                try:
                    r = self.mario_vec.get_attr(attr_name, [i])
                    results.append(r[0] if r else None)
                except Exception:
                    results.append(None)
            else:
                try:
                    r = self.coinrun_vec.get_attr(attr_name, [i - self.n_mario])
                    results.append(r[0] if r else None)
                except Exception:
                    results.append(None)
        return results

    def set_attr(self, attr_name, value, indices=None):
        if indices is None:
            indices = list(range(self.num_envs))
        for i in indices:
            if i < self.n_mario:
                self.mario_vec.set_attr(attr_name, value, [i])
            else:
                self.coinrun_vec.set_attr(attr_name, value, [i - self.n_mario])

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if indices is None:
            indices = list(range(self.num_envs))
        results = []
        for i in indices:
            if i < self.n_mario:
                try:
                    r = self.mario_vec.env_method(method_name, *args, indices=[i], **kwargs)
                    results.append(r[0] if r else None)
                except Exception:
                    results.append(None)
            else:
                try:
                    r = self.coinrun_vec.env_method(method_name, *args, indices=[i - self.n_mario], **kwargs)
                    results.append(r[0] if r else None)
                except Exception:
                    results.append(None)
        return results

    def env_is_wrapped(self, wrapper_class, indices=None):
        if indices is None:
            indices = list(range(self.num_envs))
        return [False] * len(indices)


def make_mixed_vec_env(
    n_envs=10,
    frame_stack=4,
    mario_ratio=0.5,
    coinrun_reward_scale=1.0,
    coinrun_progress_coef=0.05,
    coinrun_success_bonus=10.0,
    coinrun_fail_penalty=2.0,
    coinrun_step_penalty=0.002,
    use_aligned_reward=False,
    progress_coef=0.02,
    success_bonus=100.0,
    fail_penalty=20.0,
    step_penalty=0.002,
    fixed_level=False,
    start_level=0,
    distribution_mode="easy",
    max_episode_steps=3000,
):
    """
    创建混合 VecEnv：mario_ratio 比例为 mario，其余为 coinrun。
    """
    if not PROCGEN_AVAILABLE:
        raise ImportError("procgen 未安装，无法使用混合环境。")
    n_mario = max(1, int(n_envs * mario_ratio))
    n_coinrun = max(1, n_envs - n_mario)
    mario_vec = SubprocVecEnv(
        [lambda: make_mario_env(max_episode_steps=max_episode_steps) for _ in range(n_mario)]
    )
    coinrun_vec = make_coinrun_vec_env(
        n_envs=n_coinrun,
        fixed_level=fixed_level,
        start_level=start_level,
        distribution_mode=distribution_mode,
        max_episode_steps=max_episode_steps,
    )
    mario_vec = VecFrameStack(mario_vec, n_stack=frame_stack, channels_order="last")
    coinrun_vec = VecFrameStack(coinrun_vec, n_stack=frame_stack, channels_order="last")
    return MixedVecEnv(
        mario_vec,
        coinrun_vec,
        coinrun_reward_scale=coinrun_reward_scale,
        coinrun_progress_coef=coinrun_progress_coef,
        coinrun_success_bonus=coinrun_success_bonus,
        coinrun_fail_penalty=coinrun_fail_penalty,
        coinrun_step_penalty=coinrun_step_penalty,
        use_aligned_reward=use_aligned_reward,
        progress_coef=progress_coef,
        success_bonus=success_bonus,
        fail_penalty=fail_penalty,
        step_penalty=step_penalty,
        max_episode_steps=max_episode_steps,
    )
