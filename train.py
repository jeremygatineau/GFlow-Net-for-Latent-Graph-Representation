# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Imitation learning training script (behavioral cloning)."""
from datetime import datetime
from pathlib import Path
import pickle
import random
import json

import hydra
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from gfn import ConditionalFlowModel, ContrastiveScorer

from nocturne.envs.wrappers import create_env

DEBUG = True
def set_seed_everywhere(seed):
    """Ensure determinism."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)




@hydra.main(config_path="/home/jeremy/Documents/GitHub/GFlow-Net-for-Latent-Graph-Representation/", config_name="env_config")
def generate_contrastive_dataset(cfg):

    env = create_env(cfg)
    episodes = []

    for i in tqdm(range(cfg.number_of_episodes), desc="Generating observations"):
        env.reset()
        observations = []
        # render and display first frame
        env.render()
        env.make_all_vehicles_experts()
        for j in range(cfg.episode_length):
            action = env.action_space.sample()
            state, reward, done, info = env.step({})
            all_vhc_ids = list(state.keys())
            obs = env.get_observation(env.controlled_vehicles[0])
            
            observations.append(obs)
            if done["__all__"] == True:
                break
        episodes.append(observations)

    with open(f"../../../../data/contrastive_dataset_{cfg.number_of_episodes}_{cfg.episode_length}.pkl", "wb") as f:
        # create a dataset of pairs of observations with labels 0 if they are from the same episode and 1 if they are from different episodes
        observations_neg = []
        observations_pos = []

        # generate cfg.max_pairs_per_episode random pairs of observations from the same episodes
        for episode in episodes:
            for i in range(cfg.max_pairs_per_episode):
                obs1 = random.choice(episode)
                obs2 = random.choice(episode)
                observations_pos.append((obs1, obs2))
        # do the same for different episodes
        for ep_ix, episode1 in enumerate(episodes):
            # generate random other episode
            episode2_idx = random.randint(0, len(episodes)-1)
            while episode2_idx == ep_ix:
                episode2_idx = random.randint(0, len(episodes)-1)
            episode2 = episodes[episode2_idx]
            for i in range(cfg.max_pairs_per_episode):
                obs1 = random.choice(episode1)
                obs2 = random.choice(episode2)
                observations_neg.append((obs1, obs2))
        


        # make dataset
        dataset = []
        for obs1, obs2 in observations_pos:
            dataset.append((obs1, obs2, 0))
        for obs1, obs2 in observations_neg:
            dataset.append((obs1, obs2, 1))
        random.shuffle(dataset)
        pickle.dump(dataset, f)
        print(f"Saved dataset to {f.name}")


class Dataset(torch.utils.data.Dataset):
  'Contrastive dataset of pairs of observations with labels 0 if they are from the same episode and 1 if they are from different episodes'
  def __init__(self, file_name):
        'Initialization'
        with open(file_name, "rb") as f:
            self.dataset = pickle.load(f)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        obs1, obs2 = self.dataset[index][0]
        label = self.dataset[index][1]

        return obs1, obs2, label

@hydra.main(config_path="/home/jeremy/Documents/GitHub/GFlow-Net-for-Latent-Graph-Representation/", config_name="train_config")
def _main(args):
    """Train contrastive gflownet graph generation model."""

    set_seed_everywhere(args.seed)
    # create dataset and dataloader
    if args.actions_are_positions:
        expert_bounds = [[-0.5, 3], [-3, 3], [-0.07, 0.07]]
        actions_discretizations = [21, 21, 21]
        actions_bounds = [[-0.5, 3], [-3, 3], [-0.07, 0.07]]
        mean_scalings = [3, 3, 0.07]
        std_devs = [0.1, 0.1, 0.02]
    else:
        expert_bounds = [[-6, 6], [-0.7, 0.7]]
        actions_bounds = expert_bounds
        actions_discretizations = [15, 43]
        mean_scalings = [3, 0.7]
        std_devs = [0.1, 0.02]

    dataloader_cfg = {
        'tmin': 0,
        'tmax': 90,
        'view_dist': args.view_dist,
        'view_angle': args.view_angle,
        'dt': 0.1,
        'expert_action_bounds': expert_bounds,
        'expert_position': args.actions_are_positions,
        'state_normalization': 100,
        'n_stacked_states': args.n_stacked_states,
    }
    scenario_cfg = {
        'start_time': 0,
        'allow_non_vehicles': True,
        'spawn_invalid_objects': True,
        'max_visible_road_points': args.max_visible_road_points,
        'sample_every_n': 1,
        'road_edge_first': False,
    }
    dataset = WaymoDataset(
        data_path=args.path,
        file_limit=args.num_files,
        dataloader_config=dataloader_cfg,
        scenario_config=scenario_cfg,
    )
    data_loader = iter(
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.n_cpus,
            pin_memory=True,
        ))

    # create model
    sample_state, _ = next(data_loader)
    n_states = sample_state.shape[-1]

    net_config = {
        'num_node_features': 5,
        'hidden_dim': 7,
        'num_heads': 2,
        'dropout': 0.1,
        'num_gat_layers': 2,
        'graph_embedding_dim': 10,
        'num_contrastive_layers': 2,
    }
    net_config_gfn = {
        'obs_type': 'image',
        'obs_channels': 3,
        'num_conv_layers': 5,
        'obs_size': 200,
        'max_nodes': 4,
        'embedding_dim': 10,
        'dropout': 0.1,
        'num_heads': 2,
        'num_transformer_blocks': 4,
        'num_tb_stop_heads': 4,
        'num_tb_transition_heads': 4,
        'num_variables' : 10
    }

    scorer = ContrastiveScorer(net_config).to(args.device).train()
    gflownet = ConditionalFlowModel(net_config_gfn).to(args.device).train()
    
    # create optimizer
    optimizer_scorer = Adam(scorer.parameters(), lr=args.lr)
    optimizer_gflownet = Adam(gflownet.parameters(), lr=args.lr)

    # create exp dir
    time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    exp_dir = Path.cwd() / Path('train_logs') / time_str
    exp_dir.mkdir(parents=True, exist_ok=True)

    # save configs
    configs_path = exp_dir / 'configs.json'
    configs = {
        'scenario_cfg': scenario_cfg,
        'dataloader_cfg': dataloader_cfg,
        'gfn_config': net_config_gfn,
        'scorer_config': net_config,
    }
    with open(configs_path, 'w') as fp:
        json.dump(configs, fp, sort_keys=True, indent=4)
    print('Wrote configs at', configs_path)

    # tensorboard writer
    if args.write_to_tensorboard:
        writer = SummaryWriter(log_dir=str(exp_dir))
    # wandb logging
    if args.wandb:
        wandb_mode = "online"
        wandb.init(config=args,
                project=args.wandb_project,
                name=args.experiment,
                group=args.experiment,
                resume="allow",
                settings=wandb.Settings(start_method="fork"),
                mode=wandb_mode if not DEBUG else "disabled")

    # train loop
    print('Exp dir created at', exp_dir)
    print(f'`tensorboard --logdir={exp_dir}`\n')
    for epoch in range(args.epochs):
        print(f'\nepoch {epoch+1}/{args.epochs}')
        n_samples = epoch * args.batch_size * (args.samples_per_epoch //
                                               args.batch_size)
        
        for i in tqdm(range(args.samples_per_epoch // args.batch_size), 
                      unit='batch'): 
            
            # get states and expert actions
            states, expert_actions = next(data_loader)
            print(states.shape, states.dtype)
            print(np.sqrt(states.shape[1]))
            print(expert_actions.shape, expert_actions.dtype)
            print(states)
            
            states = states.to(args.device)
            expert_actions = expert_actions.to(args.device)
            1+" " # --------------------------------------------------- STOP 
            # compute loss
            if args.discrete:
                log_prob, expert_idxs = model.log_prob(states,
                                                       expert_actions,
                                                       return_indexes=True)
            else:
                dist = model.dist(states)
                log_prob = dist.log_prob(expert_actions.float())
            loss = -log_prob.mean()

            metrics_dict = {}

            # optim step
            optimizer.zero_grad()
            loss.backward()

            # grad clipping
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item()**2
            total_norm = total_norm**0.5
            metrics_dict['train/grad_norm'] = total_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item()**2
            total_norm = total_norm**0.5
            metrics_dict['train/post_clip_grad_norm'] = total_norm
            optimizer.step()

            # tensorboard logging
            metrics_dict['train/loss'] = loss.item()

            if args.actions_are_positions:
                metrics_dict['train/x_logprob'] = log_prob[0]
                metrics_dict['train/y_logprob'] = log_prob[1]
                metrics_dict['train/steer_logprob'] = log_prob[2]
            else:
                metrics_dict['train/accel_logprob'] = log_prob[0]
                metrics_dict['train/steer_logprob'] = log_prob[1]

            if not model_cfg['discrete']:
                diff_actions = torch.mean(torch.abs(dist.mean -
                                                    expert_actions),
                                          axis=0)
                metrics_dict['train/accel_diff'] = diff_actions[0]
                metrics_dict['train/steer_diff'] = diff_actions[1]
                metrics_dict['train/l2_dist'] = torch.norm(
                    dist.mean - expert_actions.float())

            if model_cfg['discrete']:
                with torch.no_grad():
                    model_actions, model_idxs = model(states,
                                                      deterministic=True,
                                                      return_indexes=True)
                accuracy = [
                    (model_idx == expert_idx).float().mean(axis=0)
                    for model_idx, expert_idx in zip(model_idxs, expert_idxs.T)
                ]
                if args.actions_are_positions:
                    metrics_dict['train/x_pos_acc'] = accuracy[0]
                    metrics_dict['train/y_pos_acc'] = accuracy[1]
                    metrics_dict['train/heading_acc'] = accuracy[2]
                else:
                    metrics_dict['train/accel_acc'] = accuracy[0]
                    metrics_dict['train/steer_acc'] = accuracy[1]

            for key, val in metrics_dict.items():
                if args.write_to_tensorboard:
                    writer.add_scalar(key, val, n_samples)
            if args.wandb:
                wandb.log(metrics_dict, step=n_samples)
        # save model checkpoint
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            model_path = exp_dir / f'model_{epoch+1}.pth'
            torch.save(model, str(model_path))
            pickle.dump(filter, open(exp_dir / f"filter_{epoch+1}.pth", "wb"))
            print(f'\nSaved model at {model_path}')
        if args.discrete:
            if args.actions_are_positions:
                print('xpos')
                print('model: ', model_idxs[0][0:10])
                print('expert: ', expert_idxs[0:10, 0])
                print('ypos')
                print('model: ', model_idxs[1][0:10])
                print('expert: ', expert_idxs[0:10, 1])
                print('steer')
                print('model: ', model_idxs[2][0:10])
                print('expert: ', expert_idxs[0:10, 2])
            else:
                print('accel')
                print('model: ', model_idxs[0][0:10])
                print('expert: ', expert_idxs[0:10, 0])
                print('steer')
                print('model: ', model_idxs[1][0:10])
                print('expert: ', expert_idxs[0:10, 1])

    print('Done, exp dir is', exp_dir)

    writer.flush()
    writer.close()


if __name__ == '__main__':
    #main()

    # generate dataset, parameters in env_config are episode_len = 80, n_episodes = 1000
    import os
    if not os.path.exists("data/contrastive_dataset_80_1000.pkl"):
        generate_contrastive_dataset()

    # test if dataset works, 
    dataset = Dataset("data/contrastive_dataset_10_80.pkl")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
    )
    for i, x in zip(range(100), data_loader):
        print(i, x[0].shape, x[1].shape, x[2].shape)

    
