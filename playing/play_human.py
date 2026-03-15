import gym_super_mario_bros
from nes_py.app.play_human import play_human
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def main():
    env = gym_super_mario_bros.make("SuperMarioBros-v2")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    play_human(env)


if __name__ == "__main__":
    main()

#操作：
#键盘操作：
#方向键：A键向左，D键向右，O键向上