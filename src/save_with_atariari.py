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
from game import AgentEnv, EpisodeReplayEnv, Save
from models.actor_critic import ActorCritic
from models.world_model import WorldModel
from utils import save_recording

import json
from einops import rearrange

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
    
    for i, tr_one_episode in enumerate(tr_episodes):
        
        # change this. Need to use dataset and dataloader for identical batch sizes
        ep_length = len(tr_one_episode) # n times 1 x 4 x 3 x 64 x 64
        tr_one_episode = torch.cat(tr_one_episode)
        print(tr_one_episode.shape)
        print(tr_actions[i].shape)
        tr_one_episode = rearrange(tr_one_episode, '(t f c) h w -> t f c h w', t=ep_length, f=4, c=3 if args.color else 1) 

        tr_one_episode = tr_one_episode.to(device, non_blocking=True)

        episode_buffer = []
        action_buffer = np.array(tr_actions[i]) # array of length n

        for stacked_frames in tr_one_episode:

            episode_buffer.append(np.array(stacked_frames.cpu()))
        


        save_recording(Path(args.save_dir) / args.env_name / "train", str(i+1), np.stack(episode_buffer), action_buffer)





