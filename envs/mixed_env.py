"""
联合训练环境：混合 Mario 与 CoinRun，用于实验组 B。
（小朋友版：就像把两个游戏机并排摆在一起，一起训练同一个“大脑”。）
"""
import numpy as np  # 用来做数字数组：拼画面、记分数、算位置
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, VecFrameStack  # 向量环境工具：并行、堆叠多帧

from .mario_env import make_mario_env  # 造一个马里奥小世界
from .coinrun_env import make_coinrun_vec_env, PROCGEN_AVAILABLE  # 造 CoinRun 世界；PROCGEN_AVAILABLE 表示 procgen 装没装好


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
        n_mario = mario_vec.num_envs  # 数一数：马里奥那边开了几条“并行跑道”
        n_coinrun = coinrun_vec.num_envs  # 数一数：CoinRun 那边开了几条“并行跑道”
        n_total = n_mario + n_coinrun  # 总共多少条跑道（训练时一批数据就这么长）
        obs_space = mario_vec.observation_space  # 画面长什么样（用马里奥这边的定义当“标准答案”）
        act_space = mario_vec.action_space  # 能按哪些键（也用马里奥这边的定义）
        super().__init__(n_total, obs_space, act_space)  # 告诉父类：我们是一个大向量环境
        self.mario_vec = mario_vec  # 存起来：左边这一坨是马里奥
        self.coinrun_vec = coinrun_vec  # 存起来：右边这一坨是 CoinRun
        self.n_mario = n_mario  # 记住马里奥有几条跑道，后面要切动作
        self.n_coinrun = n_coinrun  # 记住 CoinRun 有几条跑道
        # 统一奖励尺度：缩放 CoinRun 奖励，缓解多任务量级失衡
        self.coinrun_reward_scale = float(coinrun_reward_scale)  # CoinRun 原始分可能太大或太小，这里乘一个倍数“调音量”
        # CoinRun 专用塑形参数（仅在 use_aligned_reward=False 时启用）
        self.coinrun_progress_coef = float(coinrun_progress_coef)  # 往右走一点点，额外加多少分（鼓励前进）
        self.coinrun_success_bonus = float(coinrun_success_bonus)  # 通关了额外加一大笔分
        self.coinrun_fail_penalty = float(coinrun_fail_penalty)  # 输了扣一点分
        self.coinrun_step_penalty = float(coinrun_step_penalty)  # 每走一步扣一点点（鼓励别磨蹭）
        # 记录每个 CoinRun 子环境上一时刻的横向位置，用于计算右移增量奖励
        self._coinrun_prev_x = np.full(self.n_coinrun, np.nan, dtype=np.float32)  # 先全填“空记号”NaN，表示还没记过位置
        # 统一奖励语义：前进（可选）+ 通关 - 失败 - 时间
        self.use_aligned_reward = bool(use_aligned_reward)  # True：两边奖励按同一套“规矩”改写；False：主要只改写 CoinRun
        self.progress_coef = float(progress_coef)  # “前进进度”在总奖励里占多大比重
        self.success_bonus = float(success_bonus)  # 通关大奖（aligned 模式里用）
        self.fail_penalty = float(fail_penalty)  # 失败扣多少（aligned 里主要给 CoinRun 用）
        self.step_penalty = float(step_penalty)  # 每步小扣（aligned 里用）
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)  # 一局最多走多少步；None 表示不在这里掐表
        self._mario_episode_steps = np.zeros(self.n_mario, dtype=np.int32)  # 每条马里奥跑道本局走了几步（用来判断是否拖太久）
        self._coinrun_episode_steps = np.zeros(self.n_coinrun, dtype=np.int32)  # 每条 CoinRun 跑道本局走了几步
        self._preprocess_checked = False  # 第一次拿到真画面时要做一次“体检”，做完就变 True

        # 启动时先检查空间定义，尽早发现两环境预处理配置不一致问题。
        self._validate_preprocess_spaces()  # 立刻检查：两边画面大小、类型是不是真的能拼在一起

    @staticmethod
    def _action_spaces_match(space_a, space_b):
        """尽量稳健地比较动作空间，避免仅比较对象地址。"""
        if type(space_a) is not type(space_b):  # 动作空间种类都不一样就没法比
            return False  # 不算匹配
        if hasattr(space_a, "n") and hasattr(space_b, "n"):  # 离散动作：数一数有几个按钮
            return int(space_a.n) == int(space_b.n)  # 按钮数量要一样
        if hasattr(space_a, "shape") and hasattr(space_b, "shape"):  # 连续动作：看形状
            return tuple(space_a.shape) == tuple(space_b.shape)  # 形状要一样
        return str(space_a) == str(space_b)  # 实在不行就比字符串（兜底）

    def _validate_preprocess_spaces(self):
        """检查 Mario/CoinRun 的观测空间和动作空间是否可安全拼接。"""
        mario_obs_space = self.mario_vec.observation_space  # 马里奥声明：我的画面规格是……
        coinrun_obs_space = self.coinrun_vec.observation_space  # CoinRun 声明：我的画面规格是……
        if mario_obs_space.shape != coinrun_obs_space.shape:  # 高宽通道必须一模一样，不然拼不起来
            raise ValueError(
                f"混合训练预处理不一致：observation shape 不匹配，"
                f"mario={mario_obs_space.shape}, coinrun={coinrun_obs_space.shape}"
            )
        if mario_obs_space.dtype != coinrun_obs_space.dtype:  # 像素是整数还是小数也要一致
            raise ValueError(
                f"混合训练预处理不一致：observation dtype 不匹配，"
                f"mario={mario_obs_space.dtype}, coinrun={coinrun_obs_space.dtype}"
            )
        if len(mario_obs_space.shape) != 3:  # 我们约定画面是三维：高、宽、颜色通道
            raise ValueError(
                f"混合训练预处理不符合预期：观测应为 (H, W, C)，当前 mario shape={mario_obs_space.shape}"
            )
        if mario_obs_space.shape[-1] <= 0:  # 最后一维是“几层颜色”，不能是 0
            raise ValueError(
                f"混合训练预处理不符合预期：通道数必须大于 0，当前 C={mario_obs_space.shape[-1]}"
            )
        mario_low = np.asarray(mario_obs_space.low)  # 像素允许的最小值（可能是一张表，也可能是一个数）
        mario_high = np.asarray(mario_obs_space.high)  # 像素允许的最大值
        coinrun_low = np.asarray(coinrun_obs_space.low)  # CoinRun 最小值
        coinrun_high = np.asarray(coinrun_obs_space.high)  # CoinRun 最大值
        if mario_low.shape != mario_obs_space.shape and mario_low.shape != ():  # low 的形状要么跟画面一样，要么是单个数
            raise ValueError(
                f"mario 观测空间 low 形状异常：low.shape={mario_low.shape}, obs.shape={mario_obs_space.shape}"
            )
        if mario_high.shape != mario_obs_space.shape and mario_high.shape != ():  # high 同理
            raise ValueError(
                f"mario 观测空间 high 形状异常：high.shape={mario_high.shape}, obs.shape={mario_obs_space.shape}"
            )
        if coinrun_low.shape != coinrun_obs_space.shape and coinrun_low.shape != ():  # CoinRun 的 low 也要合理
            raise ValueError(
                f"coinrun 观测空间 low 形状异常：low.shape={coinrun_low.shape}, obs.shape={coinrun_obs_space.shape}"
            )
        if coinrun_high.shape != coinrun_obs_space.shape and coinrun_high.shape != ():  # CoinRun 的 high 也要合理
            raise ValueError(
                f"coinrun 观测空间 high 形状异常：high.shape={coinrun_high.shape}, obs.shape={coinrun_obs_space.shape}"
            )
        if not self._action_spaces_match(self.action_space, self.coinrun_vec.action_space):  # 两边能按的键必须一致
            raise ValueError(
                f"混合训练动作空间不一致：mario={self.action_space}, "
                f"coinrun={self.coinrun_vec.action_space}"
            )

    @staticmethod
    def _validate_runtime_obs(mario_obs, coinrun_obs):
        """首次 reset 时检查真实观测张量，确认预处理产物一致。"""
        if mario_obs.ndim != 4 or coinrun_obs.ndim != 4:  # 真实数据应是 (几条环境, 高, 宽, 通道) 共 4 维
            raise ValueError(
                f"混合训练预处理不一致：runtime obs 维度应为 4，"
                f"mario.ndim={mario_obs.ndim}, coinrun.ndim={coinrun_obs.ndim}"
            )
        if mario_obs.shape[1:] != coinrun_obs.shape[1:]:  # 去掉第 0 维后，高宽通道要一样
            raise ValueError(
                f"混合训练预处理不一致：runtime obs shape 不匹配，"
                f"mario={mario_obs.shape[1:]}, coinrun={coinrun_obs.shape[1:]}"
            )
        if mario_obs.dtype != coinrun_obs.dtype:  # 真实像素类型也要一样
            raise ValueError(
                f"混合训练预处理不一致：runtime obs dtype 不匹配，"
                f"mario={mario_obs.dtype}, coinrun={coinrun_obs.dtype}"
            )
        mario_min, mario_max = float(np.min(mario_obs)), float(np.max(mario_obs))  # 马里奥画面里最暗和最亮是多少
        coinrun_min, coinrun_max = float(np.min(coinrun_obs)), float(np.max(coinrun_obs))  # CoinRun 同理
        if not np.isfinite(mario_min) or not np.isfinite(mario_max):  # 不能出现“不是数字”的坏值
            raise ValueError("混合训练预处理异常：mario 观测包含 NaN/Inf。")
        if not np.isfinite(coinrun_min) or not np.isfinite(coinrun_max):  # CoinRun 也不能坏
            raise ValueError("混合训练预处理异常：coinrun 观测包含 NaN/Inf。")
        # uint8 常见值域为 [0, 255]；若为浮点，常见值域为 [0, 1] 或 [0, 255]。
        if np.issubdtype(mario_obs.dtype, np.integer):  # 如果是整数像素（常见 0~255）
            if mario_min < 0.0 or mario_max > 255.0:  # 超出常见屏幕亮度范围就不对劲
                raise ValueError(
                    f"混合训练预处理异常：mario uint 观测值域越界，min={mario_min}, max={mario_max}"
                )
        else:  # 如果是小数像素
            if mario_min < -1e-6 or mario_max > 255.0 + 1e-6:  # 也给一个宽松但合理的范围
                raise ValueError(
                    f"混合训练预处理异常：mario float 观测值域异常，min={mario_min}, max={mario_max}"
                )
        if np.issubdtype(coinrun_obs.dtype, np.integer):  # CoinRun 整数像素
            if coinrun_min < 0.0 or coinrun_max > 255.0:  # 同样检查亮度范围
                raise ValueError(
                    f"混合训练预处理异常：coinrun uint 观测值域越界，min={coinrun_min}, max={coinrun_max}"
                )
        else:  # CoinRun 小数像素
            if coinrun_min < -1e-6 or coinrun_max > 255.0 + 1e-6:  # 同样检查
                raise ValueError(
                    f"混合训练预处理异常：coinrun float 观测值域异常，min={coinrun_min}, max={coinrun_max}"
                )

    @staticmethod
    def _coinrun_success(info):
        return bool(
            info.get("level_complete", False)  # 这一局算不算过关：关卡完成
            or info.get("prev_level_complete", False)  # 或者上一刻已经算完成（兼容不同返回）
            or info.get("carrot_get", False)  # 或者吃到萝卜也算一种成功（兼容字段）
        )

    @staticmethod
    def _extract_coinrun_x(info):
        """
        从 CoinRun info 中尽量提取横向进度字段。
        不同 Procgen 版本字段名可能不同，故做多键兼容。
        """
        if not isinstance(info, dict):  # info 应该是小字典；不是就拿不到 x
            return None  # 表示“不知道横坐标”
        keys = (
            "x_pos",
            "x_position",
            "x",
            "agent_x",
            "player_x",
            "scroll_x",
        )
        for key in keys:  # 一个一个键名试，像猜谜语
            value = info.get(key)  # 试着取出这个键
            if isinstance(value, (int, float, np.integer, np.floating)):  # 只要是数字就行
                return float(value)  # 转成 float 返回：这就是大概的左右位置
        return None  # 所有键都没有，就放弃

    def _coinrun_shaped_reward(self, env_idx, raw_reward, done, info):
        """
        CoinRun 塑形奖励：
        1) 原始奖励缩放（兼容旧配置）
        2) 仅奖励向右位移（防止左右抖动刷分）
        3) 通关奖励 / 失败惩罚 / 每步时间惩罚
        """
        reward = float(raw_reward) * self.coinrun_reward_scale  # 先把游戏原始分乘一个倍数
        success = self._coinrun_success(info)  # 看看算不算赢
        x_now = self._extract_coinrun_x(info)  # 试着读出现在的横坐标
        if x_now is not None:  # 如果能读到横坐标
            x_prev = self._coinrun_prev_x[env_idx]  # 上一步的横坐标（可能还是 NaN）
            if np.isfinite(x_prev):  # 如果上一步有有效数字
                reward += max(0.0, x_now - x_prev) * self.coinrun_progress_coef  # 往右走才加分，往左走不加（防刷分）
            self._coinrun_prev_x[env_idx] = float(x_now)  # 把这一步的位置记下来，下一步用
        reward -= self.coinrun_step_penalty  # 每走一步扣一点点（鼓励快一点）
        if success:  # 如果赢了
            reward += self.coinrun_success_bonus  # 给一大笔奖励
        elif bool(done):  # 如果这局结束了但没赢
            reward -= self.coinrun_fail_penalty  # 扣一点分
        if bool(done):  # 只要这局结束
            self._coinrun_prev_x[env_idx] = np.nan  # 下一局重新开始，位置清空成“未记录”
        return reward  # 返回这一小步最后的分数

    def _aligned_reward(self, raw_reward, done, success):
        """
        统一奖励模板（两环境同源）：
        - progress: max(raw_reward, 0) * progress_coef
        - success: +success_bonus
        - fail: -fail_penalty
        - step: -step_penalty
        """
        progress = max(0.0, float(raw_reward)) * self.progress_coef  # 只把“正向原始分”当成前进（负数先变 0）
        fail = bool(done and not success)  # 结束了但不算成功，就算失败
        shaped = progress - self.step_penalty  # 先加前进分，再每步扣一点点时间
        if success:  # 如果成功
            shaped += self.success_bonus  # 加上通关大奖
        if fail:  # 如果失败
            shaped -= self.fail_penalty  # 再扣失败惩罚
        return shaped  # 返回这一小步对齐后的分数

    def reset(self):
        mario_obs = self.mario_vec.reset()  # 让所有马里奥跑道重新开始，拿到画面
        coinrun_obs = self.coinrun_vec.reset()  # 让所有 CoinRun 跑道重新开始，拿到画面
        self._coinrun_prev_x.fill(np.nan)  # 横坐标记忆全部清空
        self._mario_episode_steps.fill(0)  # 步数计数从零开始
        self._coinrun_episode_steps.fill(0)  # 步数计数从零开始
        if isinstance(mario_obs, tuple):  # 有些版本会返回 (画面, 额外信息)
            mario_obs = mario_obs[0]  # 我们只要画面
        if isinstance(coinrun_obs, tuple):  # CoinRun 也可能返回二元组
            coinrun_obs = coinrun_obs[0]  # 我们只要画面
        if not self._preprocess_checked:  # 如果还没做过真实画面体检
            self._validate_runtime_obs(mario_obs, coinrun_obs)  # 体检一次
            self._preprocess_checked = True  # 以后不再重复体检（省时间）
        return np.concatenate([mario_obs, coinrun_obs], axis=0)  # 把两条跑道的画面沿“第 0 维”拼成一长条

    def step_async(self, actions):
        self._actions = actions  # 先把一整包动作存起来，等 step_wait 真执行

    def step_wait(self):
        mario_actions = self._actions[: self.n_mario]  # 前一半动作给马里奥
        coinrun_actions = self._actions[self.n_mario :]  # 后一半动作给 CoinRun
        mario_result = self.mario_vec.step(mario_actions)  # 马里奥各跑一步
        coinrun_result = self.coinrun_vec.step(coinrun_actions)  # CoinRun 各跑一步
        mario_obs, mario_rewards, mario_dones, mario_infos = mario_result[:4]  # 取出：新画面、分数、是否结束、额外信息
        coinrun_obs, coinrun_rewards, coinrun_dones, coinrun_infos = coinrun_result[:4]  # CoinRun 同样取前四个
        if isinstance(mario_obs, tuple):  # 兼容返回 (obs, info) 的情况
            mario_obs = mario_obs[0]  # 只留画面
        if isinstance(coinrun_obs, tuple):  # CoinRun 同理
            coinrun_obs = coinrun_obs[0]  # 只留画面
        obs = np.concatenate([mario_obs, coinrun_obs], axis=0)  # 拼成一大张“所有跑道的新画面”
        if self.use_aligned_reward:  # 如果开了“奖励对齐模式”
            aligned_mario_rewards = np.zeros_like(mario_rewards, dtype=np.float32)  # 先准备装马里奥的新分数
            aligned_coinrun_rewards = np.zeros_like(coinrun_rewards, dtype=np.float32)  # 再准备装 CoinRun 的新分数
            for i in range(self.n_mario):  # 对每条马里奥跑道分别算
                info = mario_infos[i] if isinstance(mario_infos[i], dict) else {}  # 取出这一跑道的小字典信息
                success = bool(info.get("flag_get", False))  # 马里奥：吃到旗子算成功
                # Mario 保持原始风格：仅前进项 + 通关奖励，不加失败/时间惩罚。
                progress = max(0.0, float(mario_rewards[i])) * self.progress_coef  # 用原始分里“正向部分”当前进
                aligned_mario_rewards[i] = progress + (self.success_bonus if success else 0.0)  # 成功就加大奖，否则只加前进小奖
            for i in range(self.n_coinrun):  # 对每条 CoinRun 跑道
                info = coinrun_infos[i] if isinstance(coinrun_infos[i], dict) else {}  # 取信息
                success = self._coinrun_success(info)  # 用 CoinRun 的成功判定
                aligned_coinrun_rewards[i] = self._aligned_reward(
                    raw_reward=coinrun_rewards[i],
                    done=coinrun_dones[i],
                    success=success,
                )  # 用统一模板算分
            rewards = np.concatenate([aligned_mario_rewards, aligned_coinrun_rewards])  # 把两边新分数拼起来
        else:  # 默认模式：马里奥不改分，主要改 CoinRun
            shaped_coinrun_rewards = np.zeros_like(coinrun_rewards, dtype=np.float32)  # 准备装 CoinRun 塑形后的分
            for i in range(self.n_coinrun):  # 每条 CoinRun 单独塑形
                info = coinrun_infos[i] if isinstance(coinrun_infos[i], dict) else {}  # 取信息
                shaped_coinrun_rewards[i] = self._coinrun_shaped_reward(
                    env_idx=i,
                    raw_reward=coinrun_rewards[i],
                    done=coinrun_dones[i],
                    info=info,
                )  # 缩放 + 往右奖励 + 通关/失败/每步惩罚
            rewards = np.concatenate([mario_rewards, shaped_coinrun_rewards])  # 马里奥用原分，CoinRun 用塑形分
        dones = np.concatenate([mario_dones, coinrun_dones])  # 哪些跑道这局结束了
        mario_infos = [dict(info) if isinstance(info, dict) else {} for info in mario_infos]  # 每条 info 都复制成可改的字典
        coinrun_infos = [dict(info) if isinstance(info, dict) else {} for info in coinrun_infos]  # CoinRun 同理
        self._mario_episode_steps += 1  # 每条马里奥跑道步数 +1
        self._coinrun_episode_steps += 1  # 每条 CoinRun 跑道步数 +1
        for info in mario_infos:  # 给马里奥的 info 贴上标签，方便日志看
            info["game"] = "mario"  # 写明这是马里奥
            info["use_aligned_reward"] = self.use_aligned_reward  # 记录当时用没用对齐奖励
            info["progress_coef"] = self.progress_coef  # 记录参数，方便排查
            info["success_bonus"] = self.success_bonus  # 记录通关大奖系数
            info["fail_penalty"] = self.fail_penalty  # 记录失败惩罚系数
            info["step_penalty"] = self.step_penalty  # 记录每步惩罚系数
        for idx, info in enumerate(mario_infos):  # 再看每条马里奥是否这步结束
            if bool(mario_dones[idx]):  # 如果结束了
                success = bool(info.get("flag_get", False))  # 成功了吗（举旗）
                timed_out = bool(
                    info.get("TimeLimit.truncated", False)  # 环境自己说超时截断
                    or (
                        self.max_episode_steps is not None  # 或者我们设了最大步数
                        and self.max_episode_steps > 0  # 且是个正数
                        and self._mario_episode_steps[idx] >= self.max_episode_steps  # 且步数到了
                        and not success  # 且没成功
                    )
                )  # 判断是不是“时间到了还没过关”
                info["done"] = True  # 标记：这局真的结束
                info["time_limit_truncated"] = timed_out  # 标记：是不是拖到时间上限
                info["episode_end_reason"] = "time_limit" if timed_out else ("success" if success else "fail")  # 用一句话说明为啥结束
                self._mario_episode_steps[idx] = 0  # 下一把重新计数
        for idx, info in enumerate(coinrun_infos):  # CoinRun 的 info 也贴标签
            info["game"] = "coinrun"  # 写明这是 CoinRun
            # 记录缩放信息，便于日志排查
            info["reward_scale"] = self.coinrun_reward_scale  # 记录奖励放大了多少倍
            info["coinrun_progress_coef"] = self.coinrun_progress_coef  # 记录往右奖励系数
            info["coinrun_success_bonus"] = self.coinrun_success_bonus  # 记录通关加多少
            info["coinrun_fail_penalty"] = self.coinrun_fail_penalty  # 记录失败扣多少
            info["coinrun_step_penalty"] = self.coinrun_step_penalty  # 记录每步扣多少
            info["use_aligned_reward"] = self.use_aligned_reward  # 是否对齐奖励
            info["progress_coef"] = self.progress_coef  # 对齐模式相关参数也记一份
            info["success_bonus"] = self.success_bonus
            info["fail_penalty"] = self.fail_penalty
            info["step_penalty"] = self.step_penalty
            if bool(coinrun_dones[idx]):  # 如果这条 CoinRun 结束了
                success = self._coinrun_success(info)  # 算不算赢
                timed_out = bool(
                    info.get("TimeLimit.truncated", False)  # 环境说时间截断
                    or info.get("truncated", False)  # 或者另一种截断标记
                    or (
                        self.max_episode_steps is not None  # 或者我们自己掐表
                        and self.max_episode_steps > 0
                        and self._coinrun_episode_steps[idx] >= self.max_episode_steps
                        and not success
                    )
                )  # 综合判断是否拖太久失败
                info["done"] = True  # 这局结束
                info["time_limit_truncated"] = timed_out  # 是否时间原因
                info["episode_end_reason"] = "time_limit" if timed_out else ("success" if success else "fail")  # 结束原因
                self._coinrun_episode_steps[idx] = 0  # 下一把步数归零
        infos = list(mario_infos) + list(coinrun_infos)  # 把两边 info 列表拼成一个长列表（顺序要和画面拼接一致）
        return obs, rewards, dones, infos  # 把四大件交给训练算法

    def seed(self, seed=None):
        """为混合环境中的各子环境分别设置随机种子。"""
        mario_seeds = self.mario_vec.seed(seed)  # 先给马里奥这边撒种子（让随机可复现）
        coinrun_seed = None if seed is None else seed + self.n_mario  # CoinRun 用另一个起点，避免跟马里奥完全同随机序列
        coinrun_seeds = self.coinrun_vec.seed(coinrun_seed)  # 给 CoinRun 撒种子
        return list(mario_seeds) + list(coinrun_seeds)  # 把两边的种子列表拼起来返回

    def close(self):
        self.mario_vec.close()  # 关掉马里奥那边所有进程/环境
        self.coinrun_vec.close()  # 关掉 CoinRun 那边

    def get_attr(self, attr_name, indices=None):
        if indices is None:  # 如果没指定要哪几条跑道
            indices = list(range(self.num_envs))  # 默认全部跑道
        results = []  # 准备装查询结果
        for i in indices:  # 一条一条问
            if i < self.n_mario:  # 如果编号在马里奥区间
                try:
                    r = self.mario_vec.get_attr(attr_name, [i])  # 向马里奥向量环境要属性
                    results.append(r[0] if r else None)  # 取出第一个结果放进列表
                except Exception:  # 如果它不支持这个属性
                    results.append(None)  # 就记成空
            else:  # 否则编号在 CoinRun 区间
                try:
                    r = self.coinrun_vec.get_attr(attr_name, [i - self.n_mario])  # 编号要减掉马里奥数量
                    results.append(r[0] if r else None)  # 同样取第一个
                except Exception:  # 不支持就空
                    results.append(None)
        return results  # 返回一长串查询结果

    def set_attr(self, attr_name, value, indices=None):
        if indices is None:  # 默认所有跑道
            indices = list(range(self.num_envs))
        for i in indices:  # 对每个编号设置属性
            if i < self.n_mario:  # 马里奥区
                self.mario_vec.set_attr(attr_name, value, [i])  # 设置马里奥第 i 条
            else:  # CoinRun 区
                self.coinrun_vec.set_attr(attr_name, value, [i - self.n_mario])  # 设置 CoinRun 对应那条

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if indices is None:  # 默认所有跑道
            indices = list(range(self.num_envs))
        results = []  # 装每个方法的返回值
        for i in indices:  # 逐个调用
            if i < self.n_mario:  # 马里奥区
                try:
                    r = self.mario_vec.env_method(method_name, *args, indices=[i], **kwargs)  # 调底层环境的方法
                    results.append(r[0] if r else None)  # 收集返回值
                except Exception:  # 调用失败
                    results.append(None)  # 记空
            else:  # CoinRun 区
                try:
                    r = self.coinrun_vec.env_method(method_name, *args, indices=[i - self.n_mario], **kwargs)  # 注意编号平移
                    results.append(r[0] if r else None)
                except Exception:
                    results.append(None)
        return results  # 返回一长串结果

    def env_is_wrapped(self, wrapper_class, indices=None):
        if indices is None:  # 默认所有跑道
            indices = list(range(self.num_envs))
        return [False] * len(indices)  # 这里简单回答“没包过某层包装”（避免复杂实现；训练一般够用）


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
    if not PROCGEN_AVAILABLE:  # 如果 procgen 没装好
        raise ImportError("procgen 未安装，无法使用混合环境。")  # 直接报错提醒安装
    n_mario = max(1, int(n_envs * mario_ratio))  # 按比例算马里奥要几条跑道，至少 1 条
    n_coinrun = max(1, n_envs - n_mario)  # 剩下给 CoinRun，也至少 1 条（所以总数可能大于 n_envs）
    mario_vec = SubprocVecEnv(
        [lambda: make_mario_env(max_episode_steps=max_episode_steps) for _ in range(n_mario)]
    )  # 开多个进程跑马里奥（每条跑道一个工厂函数）
    coinrun_vec = make_coinrun_vec_env(
        n_envs=n_coinrun,
        fixed_level=fixed_level,
        start_level=start_level,
        distribution_mode=distribution_mode,
        max_episode_steps=max_episode_steps,
    )  # 创建 CoinRun 的向量环境
    mario_vec = VecFrameStack(mario_vec, n_stack=frame_stack, channels_order="last")  # 把最近几帧叠成“厚画面”，通道在最后
    coinrun_vec = VecFrameStack(coinrun_vec, n_stack=frame_stack, channels_order="last")  # CoinRun 同样叠帧
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
    )  # 最后包成 MixedVecEnv：对外就是一个大环境
