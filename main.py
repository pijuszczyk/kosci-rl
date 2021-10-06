from typing import Optional

import stable_baselines3 as sb
import stable_baselines3.common.env_checker as sb_env_checker
import stable_baselines3.common.vec_env as sb_vec_env

import rl_env


def create_simple_environment(seed: int = 0):
    env = rl_env.KosciEnv(4)
    env.seed(seed)
    return env


def create_vectorized_environment(n_cpu: int = 8, seed: int = 0):
    def make_env(rank: int, seed: int = 0):
        return create_simple_environment(seed + rank)
    return sb_vec_env.SubprocVecEnv([lambda: make_env(i, seed) for i in range(n_cpu)])


def check_env(env):
    sb_env_checker.check_env(env)


def train(env, saved_model_path: Optional[str] = None, seed: int = 0):
    model = sb.PPO('MultiInputPolicy', env, seed=seed, verbose=1)
    if saved_model_path is not None:
        model = model.load(saved_model_path, env)
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
    n_cpu = 8
    seed = 0

    # env = create_simple_environment(seed)
    env = create_vectorized_environment(n_cpu, seed)

    # check_env(env)

    path = 'model.zip'
    print('Training')
    model = train(env, path, seed)
    # print('Test')
    # test(env, model)
    print(f'Saving to {path}')
    model.save(path)


if __name__ == "__main__":
    __main__()
