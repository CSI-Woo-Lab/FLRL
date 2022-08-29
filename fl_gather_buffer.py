from gym.envs import register

import torch
import argparse
import gym
from env_wrapper import *
from config import *
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import save_to_pkl
from tqdm import tqdm

# Before execution, change seed if needed
# python fl_gather_buffers.py --env_id {env_id} --expert_steps {expert_steps} --num_trajtecotires {num_trajectories} --num_clients {num_clients}

parser = argparse.ArgumentParser()
parser.add_argument("--env_id", type=int, default=argparse.SUPPRESS)
parser.add_argument("--expert_steps", type=int, default=argparse.SUPPRESS)
parser.add_argument("--num_trajectories", type=int, default=argparse.SUPPRESS)
parser.add_argument("--num_clients", type=int, default=argparse.SUPPRESS)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Change the seed if needed

# Set Environment
mode = ['easy', 'normal', 'hard', 'very_hard']

for idx, obs_conf in enumerate(config_set):
    register(id='Custom-Navi-Vel-Full-Obs-Task{}_{}-v0'.format(idx%8, mode[idx//8]), entry_point='env_wrapper:CustomEnv2', max_episode_steps=200, kwargs=dict(task_args=obs_conf))

env_name = f"Custom-Navi-Vel-Full-Obs-Task{args.env_id}_easy-v0"
env = gym.make(env_name)

# Load agent
expert_model = f"models_env_id_{args.env_id}/model-Custom-Navi-Vel-Full-Obs-Task{args.env_id}_easy-v0_{args.expert_steps}_steps.zip"
agent = SAC.load(expert_model)

# Set replay buffer
replay_buffer = ReplayBuffer(
    buffer_size=100000,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

# Make replay buffer
for i in range(args.num_clients):
    seed = 300 + i
    torch.manual_seed(seed)
    env.seed(seed)
    total_reward = 0
    for _ in tqdm(range(args.num_trajectories)):
        done = False
        obs = env.reset()
        while not done:
            action, _ = agent.predict(obs)
            next_obs, reward, done, info = env.step(action)
            replay_buffer.add(obs, next_obs, action, reward, done, [info])
            obs = next_obs
            total_reward += reward

    print(replay_buffer.pos)
    print(total_reward / args.num_trajectories)

    save_to_pkl(
        f"buffers_fl/replay-buffer-Navi-Vel-Full-Obs-Task{args.env_id}_easy-v0-ntraj-{args.num_trajectories}_{seed}.pkl",
        replay_buffer,
    )
