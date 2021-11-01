import logging
import os
from typing import Optional

import numpy as np
import stable_baselines3 as sb
import stable_baselines3.common.env_checker as sb_env_checker
import stable_baselines3.common.vec_env as sb_vec_env
import stable_baselines3.common.callbacks as sb3_callbacks

import rl_env


def _create_simple_environment(seed: int = 0):
    env = rl_env.KosciEnv(5)
    env.seed(seed)
    return env


def _create_vectorized_environment(n_cpu: int = 8, seed: int = 0):
    def make_env(rank: int, seed: int = 0):
        return _create_simple_environment(seed + rank)
    return sb_vec_env.SubprocVecEnv([lambda: make_env(i, seed) for i in range(n_cpu)])


def check_env():
    env = _create_simple_environment()
    sb_env_checker.check_env(env)


def _create_model(env, tensorboard_log: Optional[str] = None, seed: int = 0):
    policy_kwargs = {'net_arch': [256, 128, 64, {'pi': [64, 64], 'vf': [64, 64]}]}
    return sb.PPO('MultiInputPolicy',
                  env,
                  seed=seed,
                  verbose=1,
                  learning_rate=0.1,
                  batch_size=256,
                  tensorboard_log=tensorboard_log,
                  policy_kwargs=policy_kwargs)


def _train(timesteps, env, saved_model_path: Optional[str] = None, seed: int = 0):
    model = _create_model(env, 'tensorboard', seed)
    if saved_model_path is not None:
        model = model.load(saved_model_path, env)
    if timesteps > 0:
        checkpoint_callback = sb3_callbacks.CheckpointCallback(save_freq=10000, save_path='./logs/', name_prefix='model')
        model.learn(total_timesteps=timesteps, callback=checkpoint_callback)
    return model


def _test(env, model):
    obs = env.reset()
    for i in range(10000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # env.render()
        if np.any(done):
            return
            # obs = env.reset()


def do_multiproc_training(model_path: Optional[str] = 'model.zip', timesteps: int = 1000000, n_cpu: int = 8, seed: int = 0):
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.WARN)

    env = _create_vectorized_environment(n_cpu, seed)

    print('Training')
    path_to_load = model_path if model_path is not None and os.path.exists(model_path) else None
    model = _train(timesteps, env, path_to_load, seed)

    if model_path is not None:
        print(f'Saving to {model_path}')
        model.save(model_path)

    return model


def do_testing(model=None, model_path: Optional[str] = None, seed: int = 0):
    assert (model is None) != (model_path is None)

    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG)

    env = _create_simple_environment(seed)
    print('Test')

    if model is not None:
        _test(env, model)
    elif model_path is not None:
        model = _create_model(env, None, seed).load(model_path, env)
        _test(env, model)


def check_gpu():
    import torch

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    else:
        print('GPU not detected, try adjusting version of CUDA toolkit, CudNN or PyTorch')


def enable_gpu(enabled: bool = True):
    # 0 - GPU, 1 - CPU (on the original machine)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' if enabled else '1'


def __main__():
    # check_env()
    # check_gpu()

    n_cpu = 8
    seed = 0
    model_path = 'model.zip'
    enable_gpu(False)

    # do_multiproc_training(n_cpu=n_cpu, seed=seed)
    do_testing(model_path=model_path, seed=seed)


if __name__ == "__main__":
    __main__()
