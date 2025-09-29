import gym
import d4rl
import numpy as np
import torch
from torchvision.io import write_video
import os

# Create the environment
env = gym.make('antmaze-medium-play-v2')
env.reset()
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()
print(dataset.keys()) # dict_keys(['actions', 'infos/goal', 'infos/qpos', 'infos/qvel', 'observations', 'rewards', 'terminals', 'timeouts'])
print(dataset['observations'].shape) # An N x dim_observation Numpy array of observations

dataset['seq_lengths'] = np.diff(np.where(dataset['terminals'] | dataset['timeouts'])[0])

states2compare = {}
num_video = 5
save_folder = "./data_gen_antmaze_videos"
os.makedirs(save_folder, exist_ok=True)

start_idx = 0
for i in range(len(dataset['seq_lengths'])):
    env.reset()
    traj_len = dataset['seq_lengths'][i]
    traj_acts = dataset['actions'][start_idx+1:start_idx+traj_len]
    qpos = dataset['infos/qpos'][start_idx:start_idx+traj_len]
    qvel = dataset['infos/qvel'][start_idx:start_idx+traj_len]
    frames = []
    for t in range(1, len(qpos)):
        qpos_t = qpos[t]
        qvel_t = qvel[t]
        env.unwrapped.set_state(qpos_t, qvel_t)
        obs = env.physics.render(224, 224).copy()
        frames.append(torch.from_numpy(obs))
    video_tensor = torch.stack(frames)
    write_video(f'{save_folder}/output_state_video_{i}.mp4', video_tensor, fps=30) # TODO: delete this after confirm
    
    start_idx += traj_len
    states2compare[i] = {
        "states": np.concatenate([qpos[1:], qvel[1:]], axis=1), # remove the first state to align with actions
    }
    if i >= num_video:
        break

env.close()

env = gym.make('antmaze-medium-play-v2')
env.reset()
dataset = env.get_dataset()
dataset['seq_lengths'] = np.diff(np.where(dataset['terminals'] | dataset['timeouts'])[0])

start_idx = 0
for i in range(len(dataset['seq_lengths'])):
    env.reset()
    traj_len = dataset['seq_lengths'][i]
    traj_acts = dataset['actions'][start_idx+1:start_idx+traj_len]
    qpos = dataset['infos/qpos'][start_idx:start_idx+traj_len]
    qvel = dataset['infos/qvel'][start_idx:start_idx+traj_len]
    frames = []

    qpos_0 = qpos[0]
    qvel_0 = qvel[0]
    env.unwrapped.set_state(qpos_0, qvel_0)

    states = []
    for t in range(len(traj_acts)):
        action = traj_acts[t]
        ob, reward, done, _ = env.step(action)
        states.append(ob)
        obs = env.physics.render(224, 224).copy()
        frames.append(torch.from_numpy(obs))
    video_tensor = torch.stack(frames)
    write_video(f'{save_folder}/output_action_video_{i}.mp4', video_tensor, fps=30) # TODO: delete this after confirm
    
    states = np.stack(states)
    states2compare[i].update({
        "actions": states,
    })
    start_idx += traj_len
    if i >= num_video:
        break

env.close()

import matplotlib.pyplot as plt
for i, d in states2compare.items():
    states = d.get("states")
    actions = d.get("actions")
    diffs = states - actions
    norms = np.linalg.norm(diffs, axis=1)

    fig = plt.figure(figsize=(8, 3))
    plt.plot(norms)
    plt.title(f"AntMaze state diff L2 norms (first 29 dims) — traj {i}")
    plt.xlabel("Frame")
    plt.ylabel("||Δstate||")
    plt.tight_layout()
    out_path = os.path.join(save_folder, f"norms_traj_{i}.png")
    plt.savefig(out_path, dpi=150)

    plt.close(fig)