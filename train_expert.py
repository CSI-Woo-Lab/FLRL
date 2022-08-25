from gym.envs import register
import argparse
import pathlib
import gym
from env_wrapper import *
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from config import *

# python train_online.py --env_id {env_id}

learning_timesteps = 10_000

parser = argparse.ArgumentParser()
parser.add_argument("--env_id", type=int, default=argparse.SUPPRESS)
args = parser.parse_args()

mode = ['easy', 'normal', 'hard', 'very_hard']

for idx, obs_conf in enumerate(config_set):
    register(id='Custom-Navi-Vel-Full-Obs-Task{}_{}-v0'.format(idx%8, mode[idx//8]), entry_point="env_wrapper:CustomEnv2", max_episode_steps=200, kewargs=dict(task_args=obs_conf))

env_name = f"Navi-Vel-Full-Obs-Task{args.env_id}_easy-v0"
#for idx, obs_conf in enumerate(config_set):
#    register(id='Navi-Vel-Full-Obs-Task0_easy-v0', entry_point='env_wrapper:CustomEnv2', max_episode_steps=200, kwargs=dict(task_rags=obs_conf))

env = gym.make(env_name)
obs = env.reset()
print(obs)
exit()
checkpoint_callback = CheckpointCallback(
    save_freq=5_000,
    save_path=f"./models_env_id_{args.env_id}/",
    name_prefix=f"model-{env_name}",
)
agent = SAC("MlpPolicy", env, tensorboard_log="tensorboard", verbose=1)
agent.learn(
    learning_timesteps, tb_log_name=f"env-{args.env_id}", callback=checkpoint_callback
)

agent.save_replay_buffer(
    f"buffers/replay-buffer-{env_name}-{learning_timesteps}-steps.pkl"
)
