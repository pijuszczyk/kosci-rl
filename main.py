import stable_baselines3 as sb
import stable_baselines3.common.env_checker as sb_env_checker
from stable_baselines3.common.envs import SimpleMultiObsEnv

import agent
import rl_env
import sim


def check_env():
    sb_env_checker.check_env(rl_env.KosciEnv(4))


def train(env):
    model = sb.PPO('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=10000000)
    return model


def test(env, model):
    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


def __main__():
    check_env()

    env = rl_env.KosciEnv(4)
    model = train(env)
    test(env, model)


if __name__ == "__main__":
    __main__()
