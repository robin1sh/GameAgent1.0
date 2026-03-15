"""
环境工厂：兼容旧接口 make_env，新接口使用 envs 模块。
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs import make_mario_env, make_vec_env

# 兼容旧代码：make_env 返回马里奥环境（64x64 灰度，15 维动作）
def make_env():
    return make_mario_env()
