"""
为 SB3 EvalCallback 添加日志前缀，避免多路评估写入同名 eval/* 指标。
"""
from stable_baselines3.common.callbacks import EvalCallback


class PrefixedEvalCallback(EvalCallback):
    """
    将 EvalCallback 写入的 `eval/*` 指标重定向为 `<prefix>/*`。
    例如：eval/mean_reward -> eval_mario/mean_reward
    """

    def __init__(self, *args, metric_prefix="eval", **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_prefix = str(metric_prefix).strip("/") or "eval"

    def _on_step(self) -> bool:
        # 仅在触发评估时临时改写 logger.record，避免影响其他训练指标。
        should_eval = self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0)
        if not should_eval or self.logger is None:
            return super()._on_step()

        original_record = self.logger.record

        def _record_with_prefix(key, value, *args, **kwargs):
            if isinstance(key, str) and key.startswith("eval/"):
                key = f"{self.metric_prefix}/{key[len('eval/'):]}"
            return original_record(key, value, *args, **kwargs)

        self.logger.record = _record_with_prefix
        try:
            return super()._on_step()
        finally:
            self.logger.record = original_record
