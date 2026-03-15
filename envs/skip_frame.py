"""
跳帧包装器：每 skip 帧执行一次动作，累积奖励，加速训练。
"""
import gym


class SkipFrameWrapper(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        obs, reward_total, done, info = None, 0.0, False, None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            reward_total += reward
            if done:
                break
        return obs, reward_total, done, info
