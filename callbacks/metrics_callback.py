"""
扩展评估回调：记录通关率、平均得分、平均步数。
Mario: info["flag_get"] 表示通关；
Jumper: 优先读取 level_complete / prev_level_complete（吃到萝卜/过关信号），并兼容 carrot_get。
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MetricsEvalCallback(BaseCallback):
    """
    记录 rollout 中的平均奖励、通关率、平均步数。
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_completes = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                self.episode_rewards.append(ep.get("r", 0))
                self.episode_lengths.append(ep.get("l", 0))
                # Mario: flag_get；Jumper: level_complete / prev_level_complete / carrot_get
                complete = bool(
                    info.get("flag_get", False)
                    or info.get("level_complete", False)
                    or info.get("prev_level_complete", False)
                    or info.get("carrot_get", False)
                )
                self.episode_completes.append(float(complete))
        return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_rewards) > 0:
            n = min(100, len(self.episode_rewards))
            mean_reward = np.mean(self.episode_rewards[-n:])
            mean_length = np.mean(self.episode_lengths[-n:])
            complete_rate = np.mean(self.episode_completes[-n:]) if self.episode_completes else 0.0
            if self.verbose:
                print(f"  mean_reward={mean_reward:.1f} complete_rate={complete_rate:.2%} mean_steps={mean_length:.0f}")
            if self.logger:
                self.logger.record("rollout/mean_reward", mean_reward)
                self.logger.record("rollout/complete_rate", complete_rate)
                self.logger.record("rollout/mean_episode_length", mean_length)
        return None
