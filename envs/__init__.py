# 统一环境封装层
from .unified_env import (
    make_mario_env,
    make_jumper_env,
    make_coinrun_env,
    make_unified_env,
    make_vec_env,
)

__all__ = [
    "make_mario_env",
    "make_jumper_env",
    "make_coinrun_env",
    "make_unified_env",
    "make_vec_env",
]

#以上环境作为公共接口，可以被其他模块调用