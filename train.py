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
import os
import networkx as nx
import hydra
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from gfn import ContrastGFN

from nocturne.envs.wrappers import create_env
from util import get_mask, ContrastiveDataset
DEBUG = True
def set_seed_everywhere(seed):
    """Ensure determinism."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


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
            obs = env.scenario.getConeImage(
                env.controlled_vehicles[0],
                # how far the agent can see
                view_dist=cfg['subscriber']['view_dist'],
                # the angle formed by the view cone
                view_angle=cfg['subscriber']['view_angle'],
                # the agent's head angle
                head_angle=0.0,
                # whether to draw the goal position in the image
                img_height=cfg['subscriber']['img_height'],
                img_width=cfg['subscriber']['img_width'],
                draw_target_position=False)
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




@hydra.main(config_path="/home/jeremy/Documents/GitHub/GFlow-Net-for-Latent-Graph-Representation/", config_name="train_config")
def main(args):
    """Train contrastive gflownet graph generation model."""
    
    #force_cudnn_initialization()
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
    print("Loading dataset")
    dataset = ContrastiveDataset("../../../../../../data/" + args.dataset_name) # current dir for the dataset fct is deep in checkpoints dir
    data_loader = iter(torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    ))
    def gradient_clipping_and_metrics(model, metrics_dict_, prefix):
        # grad clipping
        total_norm = 0  
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item()**2
        total_norm = total_norm**0.5
        metrics_dict_['train/' + prefix + 'grad_norm'] = total_norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item()**2
        total_norm = total_norm**0.5
        metrics_dict_['train/' + prefix + 'grad_norm'] = total_norm
        return metrics_dict_
    

    net_config = {
        'num_node_features': 5,
        'embedding_dim': 10,
        'num_heads': 2,
        'dropout': 0.1,
        'num_transformer_blocks': 2,
        'graph_embedding_dim': 10,
        'num_contrastive_layers': 2,
        'batch_size': args.batch_size,
        'num_graph_per_obs': args.K,
        'node_embedding_dim': 10,
        'max_nodes': 20,
    }
    net_config_gfn = {
        'obs_type': 'image',
        'obs_channels': 3,
        'num_conv_layers': 5,
        'obs_size': 100,
        'max_nodes': 20,
        'embedding_dim': 10,
        'node_embedding_dim': 10,
        'dropout': 0.1,
        'num_heads': 4,
        'num_transformer_blocks': 4,
        'num_tb_stop_heads': 4,
        'num_tb_transition_heads': 4,
        'num_flow_layers':4,
        'num_variables' : 10,
        'batch_size': args.batch_size,
        'num_graph_per_obs': args.K,
        'height': 100,
        'width': 100,
        'log_flow': args.log_flow,
    }
    MAX_STEPS = 10
    contrastGFN = ContrastGFN(net_config, net_config_gfn, device=args.device)
    
    # create optimizer
    optimizer= Adam(contrastGFN.parameters(), lr=args.lr)

    # create exp dir
    time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    exp_dir = Path.cwd() / Path('train_logs') / time_str
    exp_dir.mkdir(parents=True, exist_ok=True)

    # save configs
    configs_path = exp_dir / 'configs.json'
    configs = {
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
        wandb_mode = "disabled" if DEBUG else "online"
        wandb.init(config=args,
                project=args.wandb_project,
                name=args.experiment,
                group=args.experiment,
                resume="allow",
                settings=wandb.Settings(start_method="fork"),
                mode=wandb_mode)

    # train loop
    print('Exp dir created at', exp_dir)
    print(f'`tensorboard --logdir={exp_dir}`\n')
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.epochs):
        print(f'\nepoch {epoch+1}/{args.epochs}')
        n_samples = epoch * args.batch_size * (args.samples_per_epoch //
                                               args.batch_size)
        
        for i in tqdm(range(args.samples_per_epoch // args.batch_size), 
                      unit='batch'): 
            
            # get states and expert actions
            obs1, obs2, labels = next(data_loader)

            obs1 = obs1.to(args.device).float()
            obs2 = obs2.to(args.device).float()
            labels = labels.to(args.device).float()
            # initialize adjacency matrices at 0
            adj = torch.zeros((2*args.batch_size, net_config_gfn['max_nodes'], net_config_gfn['max_nodes'])).float()
            
            # initialize noded embeddings at 0
            node_embeddings = torch.zeros(2*args.batch_size, net_config_gfn['max_nodes'], net_config_gfn['node_embedding_dim']).float()
            
            # reshape observations to be B x C x N x N
            obs1 = obs1.permute(0, 3, 1, 2)
            obs2 = obs2.permute(0, 3, 1, 2)

            # generate args.K graphs for each batch element by repeating batch elements K times and then reassambling their corresponding graphs
            
            obs1 = obs1.repeat(args.K, 1, 1, 1)
            obs2 = obs2.repeat(args.K, 1, 1, 1)
            
            adj = adj.repeat(args.K, 1, 1)
            node_embeddings = node_embeddings.repeat(args.K, 1, 1)
            # concatenate observations at the batch level to get 2*B*K x C x N x N to be faster
            obs = torch.cat((obs1, obs2), dim=0)
            
            # generate graphs until done flag is set to 1 for all batch elements
            done = torch.zeros(2*args.batch_size * args.K)
            step = 0
            # initialize last state's forward transition probability, flow and 
            last_fwd_transition_probs = torch.ones(2*args.batch_size * args.K, net_config_gfn['max_nodes'], net_config_gfn['max_nodes'])
            last_flow = torch.zeros(2*args.batch_size * args.K, net_config_gfn['max_nodes'], net_config_gfn['max_nodes'])
            step = 0
            while not done.all():

                step += 1

                metrics_dict = {}
                node_ebd1, node_ebd2 = node_embeddings[:args.batch_size*args.K], node_embeddings[args.batch_size*args.K:]
                adj1, adj2 = adj[:args.batch_size*args.K], adj[args.batch_size*args.K:]
                adj_new, nd_ebd_new, stop, comb_loss, cont_loss, gfn_loss, dE = contrastGFN.train_step(obs1, obs2, adj1, adj2, node_ebd1, node_ebd2, labels)
                
                # update the done flag 
                done = done + stop

                optimizer.zero_grad()
                comb_loss.mean().backward(retain_graph=False)
                gradient_clipping_and_metrics(contrastGFN, metrics_dict, 'ccGFN')
                metrics_dict['train/gflownet_loss'] = gfn_loss.item()
                metrics_dict['train/contrast_loss'] = cont_loss.item()
                metrics_dict['train/combined_loss'] = comb_loss.mean().item()
                optimizer.step()

                adj1, adj2 = adj_new[:args.batch_size*args.K], adj_new[args.batch_size*args.K:]
                node_ebd1, node_ebd2 = nd_ebd_new[:args.batch_size*args.K], nd_ebd_new[args.batch_size*args.K:]
                print("step: ", step, "number done: ", done.sum())
                print("losses ", comb_loss.mean().item(), cont_loss.item(), gfn_loss.item(), dE.item())
                if step == MAX_STEPS:
                    break
            for key, val in metrics_dict.items():
                if args.write_to_tensorboard:
                    writer.add_scalar(key, val, n_samples)
            if args.wandb:
                wandb.log(metrics_dict, step=n_samples)
                # every 10 epochs, log a vizualization of observations and their corresponding graphs
                if epoch % 10 == 0:
                    # log the first observation
                    obs = obs[0]
                    obs = obs.cpu().numpy().transpose(1, 2, 0)
                    # map to [0, 1]
                    obs = (obs - obs.min()) / (obs.max() - obs.min())
                    obs = obs * 255
                    obs = obs.astype(np.uint8)
                    wandb.log({'observation': wandb.Image(obs)}, step=n_samples)
                    # log the corresponding graph
                    adj = adj[0].cpu().numpy()
                    graph = nx.from_numpy_matrix(adj)
                    A = nx.nx_agraph.to_agraph(graph)
                    A.layout('dot')
                    A.draw('graph.png')
                    wandb.log({'graph': wandb.Image('graph.png')}, step=n_samples)
                    os.remove('graph.png')



        # save model checkpoint
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            model_path = exp_dir / f'model_{epoch+1}.pth'
            torch.save(scorer, str(model_path)+'.scorer')
            torch.save(gflownet, str(model_path) + '.gfn')
            pickle.dump(filter, open(exp_dir / f"filter_{epoch+1}.pth", "wb"))
            print(f'\nSaved model at {model_path}')

    print('Done, exp dir is', exp_dir)

    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
    """
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
    """

    
