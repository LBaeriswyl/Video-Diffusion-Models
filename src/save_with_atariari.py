from atariari.benchmark.episodes import get_episodes

from functools import partial 
from pathlib import Path

import hydra
from hydra.utils import instantiate
import torch
import omegaconf
import numpy as np

from models.agent import Agent
from envs import SingleProcessEnv, WorldModelEnv
# from game import AgentEnv, EpisodeReplayEnv, Save
from models.actor_critic import ActorCritic
from models.world_model import WorldModel
from utils import save_recording, make_video

import json
from einops import rearrange
from datetime import datetime

def save_mp4_recording(frames):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    make_video(f'{timestamp}.mp4', fps=15, frames=frames)
    print(f'Saved recording {timestamp}.')

def main(args):
    resolved_args = omegaconf.OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    print("{}".format(resolved_args).replace(', ', ',\n'))
    
    # Setup dataset
    tr_episodes, tr_actions, val_episodes, val_actions = get_episodes(env_name=args.env_name,
                 steps=args.steps,
                 seed=args.seed,
                 color=args.color,
                 collect_mode=args.collect_mode,
                 train_mode="train_encoder",
                 min_episode_length=args.min_episode_length) 

                                        
    print("Num train episodes: ", len(tr_episodes))
    print("Num train actions: ", len(tr_actions))
    print("Num val episodes: ", len(val_episodes))
    print("Num val actions: ", len(val_actions))
    print("Frame shape: ", tr_episodes[0][0].shape)
    print("Action shape: ", tr_actions[0][0].shape)

    device = torch.device(args.device)
    
    print("Saving Train data")
    for i, tr_one_episode in enumerate(tr_episodes):
        
        # change this. Need to use dataset and dataloader for identical batch sizes
        ep_length = len(tr_one_episode) # n times 1 x 4 x 3 x 64 x 64
        tr_one_episode = torch.stack(tr_one_episode)
        tr_one_actions = torch.stack(tr_actions[i])
        # print(tr_actions[i].shape)
        tr_one_episode = rearrange(tr_one_episode, 't (f c) h w -> t f c h w', t=ep_length, f=4, c=3 if args.color else 1) 

        tr_one_episode = tr_one_episode.to(device, non_blocking=True)
        tr_one_actions = tr_one_actions.to(device, non_blocking=True)

        episode_buffer = []
        action_buffer = []

        for stacked_frames in tr_one_episode:
            episode_buffer.append(np.array(stacked_frames.cpu()))

    
        for stacked_actions in tr_one_actions:
            action_buffer.append(np.array(stacked_actions.cpu()))
        
        # save_mp4_recording(rearrange(tr_one_episode, 't f c h w -> (t f) h w c').long())

        # print(np.stack(episode_buffer).shape)
        # print(np.stack(action_buffer).shape)
        save_recording(Path(args.save_dir) / args.env_name / "train", str(i+1), np.stack(episode_buffer), np.stack(action_buffer))

    print("Saving Val data")
    for i, val_one_episode in enumerate(val_episodes):
        
        # change this. Need to use dataset and dataloader for identical batch sizes
        ep_length = len(val_one_episode) # n times 1 x 4 x 3 x 64 x 64
        val_one_episode = torch.stack(val_one_episode)
        val_one_actions = torch.stack(val_actions[i])
        # print(tr_actions[i].shape)
        val_one_episode = rearrange(val_one_episode, 't (f c) h w -> t f c h w', t=ep_length, f=4, c=3 if args.color else 1) 

        val_one_episode = val_one_episode.to(device, non_blocking=True)
        val_one_actions = val_one_actions.to(device, non_blocking=True)


        episode_buffer = []
        action_buffer = []

        for stacked_frames in val_one_episode:
            episode_buffer.append(np.array(stacked_frames.cpu()))

        for stacked_actions in val_one_actions:
            action_buffer.append(np.array(stacked_actions.cpu()))

        # print(np.stack(episode_buffer).shape)
        # print(np.stack(action_buffer).shape)
        save_recording(Path(args.save_dir) / args.env_name / "val", str(i+1), np.stack(episode_buffer), np.stack(action_buffer))




