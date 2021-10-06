import stable_baselines3 as sb
import stable_baselines3.common.env_checker as sb_env_checker
import stable_baselines3.common.vec_env as sb_vec_env

import rl_env


def create_simple_environment(seed: int = 0):
    env = rl_env.KosciEnv(4)
    env.seed(seed)
    return env


def create_vectorized_environment(num_cpu: int = 8):
    def make_env(rank: int, seed: int = 0):
        return create_simple_environment(seed + rank)
    return sb_vec_env.SubprocVecEnv([lambda: make_env(i) for i in range(num_cpu)])


def check_env(env):
    sb_env_checker.check_env(env)


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
    # env = create_simple_environment()
    env = create_vectorized_environment()

    # check_env(env)

    print('Training')
    model = train(env)
    # print('Test')
    # test(env, model)
    path = 'model.zip'
    print(f'Saving to {path}')
    model.save(path)


if __name__ == "__main__":
    __main__()
