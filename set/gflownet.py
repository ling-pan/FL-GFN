import argparse
import copy
import gzip
import heapq
import itertools
import os
import pickle
from collections import defaultdict
from itertools import count

import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

import wandb
import random
import sys
import tempfile
import datetime
import time

from itertools import chain
from itertools import combinations

parser = argparse.ArgumentParser()

parser.add_argument("--device", default='cuda', type=str)
parser.add_argument("--progress", action='store_true')
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--wdb", action='store_true')
parser.add_argument("--method", default='db_gfn', type=str)
parser.add_argument("--learning_rate", default=1e-4, help="Learning rate", type=float)
parser.add_argument("--tb_lr", default=0.001, help="Learning rate", type=float)
parser.add_argument("--tb_z_lr", default=0.1, help="Learning rate", type=float)
parser.add_argument("--mbsize", default=16, help="Minibatch size", type=int)
parser.add_argument("--n_hid", default=256, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--n_train_steps", default=10000, type=int)
parser.add_argument("--exp_weight", default=0., type=float)
parser.add_argument("--temp", default=1., type=float)
parser.add_argument("--rand_pb", default=0, type=int) 
parser.add_argument("--fl", default=0, type=int)
# Env
parser.add_argument("--size", default='small', type=str, choices=['small', 'medium', 'large'])
parser.add_argument("--action_dim", default=10, type=int)
parser.add_argument("--set_size", default=5, type=int)
parser.add_argument("--bufsize", default=16, type=int)

_dev = [torch.device('cuda')]
tf = lambda x: torch.FloatTensor(x).to(_dev[0])
tl = lambda x: torch.LongTensor(x).to(_dev[0])

def set_device(dev):
    _dev[0] = dev

class SetEnv:
    def __init__(self, action_dim, set_size, intermediate_energies):
        self.action_dim = action_dim
        self.set_size = set_size
        self.intermediate_energies = np.array(intermediate_energies)
        self.intermediate_rewards = np.exp(-self.intermediate_energies)

    def reward_func(self, a):
        reward = self.intermediate_rewards[a]
        return reward

    def obs(self, s=None):
        return self._state if s is None else s

    def reset(self):
        self._state = np.int32([0] * self.action_dim)
        self._step = 0
        return self.obs()

    def step(self, a):
        self._state[a] = 1
        self._step += 1

        done = self._step == self.set_size
        rew = self.reward_func(a)

        return self.obs(), rew, done

def make_mlp(l, act=nn.LeakyReLU(), tail=[]):
    return nn.Sequential(*(sum([[nn.Linear(i, o)] + ([act] if n < len(l)-2 else []) for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))

class TBFlowNetAgent:
    def __init__(self, args, envs):
        out_dim = (args.action_dim) + (args.action_dim) + (1)
        self.model = make_mlp([args.action_dim] + [args.n_hid] * args.n_layers + [out_dim])
        self.model.to(args.dev)
        print (self.model)

        self.Z = torch.zeros((1,)).to(args.dev)
        self.Z.requires_grad_()

        self.action_dim = args.action_dim

        self.envs = envs
        self.exp_weight = args.exp_weight
        self.temp = args.temp
        self.uniform_pb = args.rand_pb
        self.dev = args.dev

        self.all_unique_Rs = [] 
        self.visited_strs = []

    def parameters(self):
        return self.model.parameters()

    def sample_many(self, mbsize, evaluate=False):
        inf = 1000000000

        batch_s, batch_a = [[] for i in range(mbsize)], [[] for i in range(mbsize)]
        batch_ri = [[] for i in range(mbsize)]
        env_idx_done_map = {i: False for i in range(mbsize)}
        not_done_envs = [i for i in range(mbsize)]
        env_idx_return_map = {}

        s = tf([i.reset() for i in self.envs])
        done = [False] * mbsize

        while not all(done):
            with torch.no_grad():
                pred = self.model(s)
                edge_mask = s.float()
                logits = (pred[..., : self.action_dim] - inf * edge_mask).log_softmax(1)
                if evaluate:
                    sample_ins_probs = (logits / self.temp).softmax(1)
                else:
                    sample_ins_probs = (1 - self.exp_weight) * (logits / self.temp).softmax(1) + self.exp_weight * (1 - edge_mask) / (1 - edge_mask + 0.0000001).sum(1).unsqueeze(1)
                acts = sample_ins_probs.multinomial(1)
                acts = acts.squeeze(-1)

            step = [i.step(a) for i, a in zip([e for d, e in zip(done, self.envs) if not d], acts)]

            for dat_idx, (curr_s, curr_a, (_, curr_r, _)) in enumerate(zip(s, acts, step)):
                env_idx = not_done_envs[dat_idx]

                batch_s[env_idx].append(curr_s)
                batch_a[env_idx].append(curr_a.unsqueeze(-1))
                batch_ri[env_idx].append(curr_r)

            for dat_idx, (ns, r, d) in enumerate(step):
                env_idx = not_done_envs[dat_idx]
                env_idx_done_map[env_idx] = d

                if d:
                    env_idx_return_map[env_idx] = r
                    batch_s[env_idx].append(tf(ns))

            not_done_envs = []
            for env_idx in env_idx_done_map:
                if not env_idx_done_map[env_idx]:
                    not_done_envs.append(env_idx)

            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf([i[0] for i in step if not i[2]])

        batch_steps = [len(batch_s[i]) for i in range(len(batch_s))]

        for i in range(len(batch_s)):
            batch_s[i] = torch.stack(batch_s[i])
            batch_a[i] = torch.stack(batch_a[i])
            assert batch_s[i].shape[0] - batch_a[i].shape[0] == 1
            batch_ri[i] = torch.tensor(batch_ri[i]).unsqueeze(-1).float().to(self.dev)

        if evaluate:
            for i in range(len(batch_ri)):
                curr_R = torch.prod(batch_ri[i]).item()
                curr_formatted_s = np.where(batch_s[i][-1].cpu().data.numpy()  == 1)[0].tolist()
                if curr_formatted_s not in self.visited_strs:
                    self.all_unique_Rs.append(curr_R)
                    self.visited_strs.append(curr_formatted_s)
        
        return [batch_s, batch_a, batch_steps, batch_ri]

    def learn_from(self, it, batch):
        inf = 1000000000

        states, actions, episode_lens, intermediate_rewards = batch

        ll_diff = []
        for data_idx in range(len(states)):
            curr_episode_len = episode_lens[data_idx]

            curr_states = states[data_idx][:curr_episode_len, :]
            curr_actions = actions[data_idx][:curr_episode_len - 1, :]
            curr_intermediate_rewards = intermediate_rewards[data_idx].squeeze(-1)

            pred = self.model(curr_states)

            edge_mask = curr_states.float()
            logits = (pred[..., :self.action_dim] - inf * edge_mask).log_softmax(1) 

            init_edge_mask = (curr_states == 0).float() 
            back_logits = ((0 if self.uniform_pb else 1) * pred[..., self.action_dim:-1] - inf * init_edge_mask).log_softmax(1) 

            logits = logits[:-1, :].gather(1, curr_actions).squeeze(1) 
            back_logits = back_logits[1:, :].gather(1, curr_actions).squeeze(1) 

            sum_logits = torch.sum(logits)
            sum_back_logits = torch.sum(back_logits)

            curr_return = torch.prod(curr_intermediate_rewards)

            curr_ll_diff = self.Z + sum_logits - curr_return.log() - sum_back_logits
            ll_diff.append(curr_ll_diff ** 2)

        ll_diff = torch.cat(ll_diff)

        loss = ll_diff.sum() / len(states)

        return [loss]

class DBFlowNetAgent:
    def __init__(self, args, envs):
        out_dim = (args.action_dim) + (args.action_dim) + (1)
        self.model = make_mlp([args.action_dim] + [args.n_hid] * args.n_layers + [out_dim])
        self.model.to(args.dev)
        print (self.model)

        self.action_dim = args.action_dim

        self.envs = envs
        self.exp_weight = args.exp_weight
        self.temp = args.temp
        self.uniform_pb = args.rand_pb
        self.dev = args.dev

        self.all_unique_Rs = [] 
        self.visited_strs = []

    def parameters(self):
        return self.model.parameters()

    def sample_many(self, mbsize, evaluate=False):
        inf = 1000000000

        batch_s, batch_a = [[] for i in range(mbsize)], [[] for i in range(mbsize)]
        batch_ri = [[] for i in range(mbsize)] 
        env_idx_done_map = {i: False for i in range(mbsize)}
        not_done_envs = [i for i in range(mbsize)]
        env_idx_return_map = {}

        s = tf([i.reset() for i in self.envs])
        done = [False] * mbsize

        while not all(done):
            with torch.no_grad():
                pred = self.model(s)
                edge_mask = s.float()
                logits = (pred[..., : self.action_dim] - inf * edge_mask).log_softmax(1)
                if evaluate:
                    sample_ins_probs = logits.softmax(1)
                else:
                    sample_ins_probs = (1 - self.exp_weight) * (logits / self.temp).softmax(1) + self.exp_weight * (1 - edge_mask) / (1 - edge_mask + 0.0000001).sum(1).unsqueeze(1)
                acts = sample_ins_probs.multinomial(1)
                acts = acts.squeeze(-1)

            step = [i.step(a) for i, a in zip([e for d, e in zip(done, self.envs) if not d], acts)]
            
            for dat_idx, (curr_s, curr_a, (_, curr_r, _)) in enumerate(zip(s, acts, step)):
                env_idx = not_done_envs[dat_idx]

                batch_s[env_idx].append(curr_s)
                batch_a[env_idx].append(curr_a.unsqueeze(-1))
                batch_ri[env_idx].append(curr_r)

            for dat_idx, (ns, r, d) in enumerate(step):
                env_idx = not_done_envs[dat_idx]
                env_idx_done_map[env_idx] = d

                if d:
                    env_idx_return_map[env_idx] = r
                    batch_s[env_idx].append(tf(ns))

            not_done_envs = []
            for env_idx in env_idx_done_map:
                if not env_idx_done_map[env_idx]:
                    not_done_envs.append(env_idx)

            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf([i[0] for i in step if not i[2]])

        batch_steps = [len(batch_s[i]) for i in range(len(batch_s))]

        for i in range(len(batch_s)):
            batch_s[i] = torch.stack(batch_s[i])
            batch_a[i] = torch.stack(batch_a[i])
            assert batch_s[i].shape[0] - batch_a[i].shape[0] == 1
            batch_ri[i] = torch.tensor(batch_ri[i]).unsqueeze(-1).float().to(self.dev)

        if evaluate:
            for i in range(len(batch_ri)):
                curr_R = torch.prod(batch_ri[i]).item()
                curr_formatted_s = np.where(batch_s[i][-1].cpu().data.numpy() == 1)[0].tolist()
                if curr_formatted_s not in self.visited_strs:
                    self.all_unique_Rs.append(curr_R)
                    self.visited_strs.append(curr_formatted_s)

        return [batch_s, batch_a, batch_steps, batch_ri]

    def learn_from(self, it, batch):
        inf = 1000000000

        states, actions, episode_lens, intermediate_rewards = batch

        ll_diff = []
        for data_idx in range(len(states)):
            curr_episode_len = episode_lens[data_idx]

            curr_states = states[data_idx][:curr_episode_len, :] 
            curr_actions = actions[data_idx][:curr_episode_len - 1, :] 
            curr_intermediate_rewards = intermediate_rewards[data_idx].squeeze(-1)
            curr_return = torch.prod(curr_intermediate_rewards)

            pred = self.model(curr_states)

            edge_mask = curr_states.float()
            logits = (pred[..., :self.action_dim] - inf * edge_mask).log_softmax(1) 

            init_edge_mask = (curr_states == 0).float() 
            back_logits = ((0 if self.uniform_pb else 1) * pred[..., self.action_dim:-1] - inf * init_edge_mask).log_softmax(1) 

            logits = logits[:-1, :].gather(1, curr_actions).squeeze(1) 
            back_logits = back_logits[1:, :].gather(1, curr_actions).squeeze(1) 

            log_flow = pred[..., -1]
            log_flow = log_flow[:-1] 

            curr_ll_diff = torch.zeros(curr_states.shape[0] - 1).to(self.dev)
            curr_ll_diff += log_flow
            curr_ll_diff += logits
            curr_ll_diff[:-1] -= log_flow[1:]
            curr_ll_diff -= back_logits
            curr_ll_diff[-1] -= curr_return.log()

            ll_diff.append(curr_ll_diff ** 2)

        ll_diff = torch.cat(ll_diff)

        loss = ll_diff.sum() / len(ll_diff)

        return [loss]

    def learn_from_fl(self, it, batch):
        inf = 1000000000

        states, actions, episode_lens, intermediate_rewards = batch

        ll_diff = []
        for data_idx in range(len(states)):
            curr_episode_len = episode_lens[data_idx]

            curr_states = states[data_idx][:curr_episode_len, :] 
            curr_actions = actions[data_idx][:curr_episode_len - 1, :] 
            curr_intermediate_rewards = intermediate_rewards[data_idx].squeeze(-1)

            pred = self.model(curr_states)

            edge_mask = curr_states.float()
            logits = (pred[..., :self.action_dim] - inf * edge_mask).log_softmax(1) 

            init_edge_mask = (curr_states == 0).float() 
            back_logits = ((0 if self.uniform_pb else 1) * pred[..., self.action_dim:-1] - inf * init_edge_mask).log_softmax(1)

            logits = logits[:-1, :].gather(1, curr_actions).squeeze(1) 
            back_logits = back_logits[1:, :].gather(1, curr_actions).squeeze(1) 

            log_flow = pred[..., -1] 

            curr_ll_diff = torch.zeros(curr_states.shape[0] - 1).to(self.dev)
            curr_ll_diff += log_flow[:-1]
            curr_ll_diff += logits
            curr_ll_diff -= back_logits
            curr_ll_diff -= log_flow[1:]
            curr_ll_diff -= curr_intermediate_rewards.log()

            ll_diff.append(curr_ll_diff ** 2)

        ll_diff = torch.cat(ll_diff)

        loss = ll_diff.sum() / len(ll_diff)

        return [loss]

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    method_name = args.method
    if args.method == 'db_gfn' and args.fl:
        method_name = 'fl_' + args.method

    if args.wdb:
        wandb.init(project='gfn', name='{}_{}'.format(method_name, args.seed))

    args.dev = torch.device(args.device)
    set_device(args.dev)

    # exp_weight is fine-tuned for each method by grid search
    if method_name == 'fl_db_gfn':
        args.exp_weight = 0.5
    elif method_name == 'db_gfn':
        args.exp_weight = 1.0
    elif method_name == 'tb_gfn':
        args.exp_weight = 0.0625

    if args.size == 'small':
        args.action_dim = 30
        args.set_size = 20
        intermediate_energies = [0.6961474461702144, 0.883494921538938, -0.2745059751263419, 0.883494921538938, 0.7442282139370466, 0.7442282139370466, -0.40046449111789073, 0.6261749162235306, -0.4381397850674522, 0.4110535720923896, -0.4381397850674522, 0.8206350287761408, 0.6961474461702144, 0.013149744911117978, 0.6961474461702144, 0.4110535720923896, 0.013149744911117978, -0.40046449111789073, -0.4381397850674522, 0.883494921538938, 0.7442282139370466, -0.2745059751263419, 0.6261749162235306, 0.6261749162235306, 0.8206350287761408, -0.40046449111789073, 0.8206350287761408, -0.2745059751263419, 0.4110535720923896, 0.013149744911117978]
        if method_name == 'fl_db_gfn':
            args.exp_weight = 1.0
        elif method_name == 'tb_gfn':
            args.exp_weight = 0.
    elif args.size == 'medium':
        args.action_dim = 80
        args.set_size = 60
        intermediate_energies = [-0.5230497555411129, 0.5731802451199923, 0.6881812517688572, -0.7866830411595669, -0.41860806745880197, 0.7396350970666805, 0.5731802451199923, 0.7396350970666805, -0.7866830411595669, 0.6009291618806774, 0.6881812517688572, -0.5230497555411129, -0.7837872949891345, -0.7837872949891345, -0.41860806745880197, -0.7866830411595669, 0.6009291618806774, -0.41860806745880197, -0.3855205729595568, -0.5230497555411129, -0.5230497555411129, -0.41860806745880197, 0.2961218224370139, 0.6881812517688572, 0.6009291618806774, -0.7837872949891345, 0.2961218224370139, -0.7866830411595669, 0.5731802451199923, 0.2961218224370139, -0.3855205729595568, -0.41860806745880197, -0.5230497555411129, -0.7837872949891345, 0.7396350970666805, -0.41860806745880197, 0.6009291618806774, -0.41860806745880197, 0.2961218224370139, -0.5230497555411129, 0.2961218224370139, 0.6881812517688572, -0.5230497555411129, -0.7837872949891345, 0.6881812517688572, 0.2961218224370139, 0.6881812517688572, 0.5731802451199923, 0.5731802451199923, 0.6881812517688572, -0.3855205729595568, -0.3855205729595568, -0.5230497555411129, 0.5731802451199923, 0.2961218224370139, -0.7837872949891345, -0.41860806745880197, 0.7396350970666805, 0.6881812517688572, -0.3855205729595568, 0.7396350970666805, 0.6009291618806774, -0.7837872949891345, -0.7866830411595669, 0.6009291618806774, 0.5731802451199923, 0.7396350970666805, -0.7866830411595669, 0.2961218224370139, -0.7866830411595669, 0.6009291618806774, 0.6009291618806774, 0.5731802451199923, 0.7396350970666805, -0.3855205729595568, 0.7396350970666805, -0.7866830411595669, -0.3855205729595568, -0.3855205729595568, -0.7837872949891345]
    elif args.size == 'large':
        args.action_dim = 100
        args.set_size = 80
        intermediate_energies = [-0.15382957507887518, -0.15382957507887518, -0.8596854107736154, -0.8596854107736154, 0.13832182722858843, -0.4589997720511263, 0.9244538380742333, -0.4589997720511263, -0.5938789622812419, 0.7326989860019331, 0.3925029176153736, 0.7231982581431591, 0.7326989860019331, -0.8596854107736154, 0.9244538380742333, -0.5938789622812419, -0.5938789622812419, 0.13832182722858843, -0.5596177369051512, 0.7326989860019331, -0.5596177369051512, -0.8596854107736154, 0.13832182722858843, 0.9244538380742333, 0.7231982581431591, -0.15382957507887518, -0.15382957507887518, -0.5938789622812419, 0.9244538380742333, 0.13832182722858843, 0.9244538380742333, 0.3925029176153736, -0.15382957507887518, 0.3925029176153736, -0.4589997720511263, -0.4589997720511263, -0.8596854107736154, 0.13832182722858843, -0.8596854107736154, -0.5596177369051512, 0.13832182722858843, -0.8596854107736154, 0.13832182722858843, 0.7326989860019331, 0.9244538380742333, 0.7231982581431591, 0.13832182722858843, 0.7231982581431591, -0.4589997720511263, -0.15382957507887518, 0.7326989860019331, -0.5938789622812419, -0.5938789622812419, -0.5596177369051512, 0.9244538380742333, 0.7231982581431591, 0.7231982581431591, -0.8596854107736154, 0.7326989860019331, -0.4589997720511263, 0.3925029176153736, -0.5938789622812419, 0.7326989860019331, 0.3925029176153736, 0.9244538380742333, -0.4589997720511263, -0.5596177369051512, -0.15382957507887518, 0.3925029176153736, -0.5938789622812419, -0.15382957507887518, -0.5938789622812419, 0.13832182722858843, 0.7326989860019331, -0.5596177369051512, -0.8596854107736154, -0.15382957507887518, -0.5596177369051512, 0.9244538380742333, 0.3925029176153736, -0.4589997720511263, -0.8596854107736154, 0.7326989860019331, 0.7231982581431591, -0.4589997720511263, -0.5596177369051512, -0.5596177369051512, 0.3925029176153736, 0.3925029176153736, 0.7231982581431591, 0.13832182722858843, -0.15382957507887518, 0.7231982581431591, 0.7231982581431591, -0.5596177369051512, -0.5938789622812419, 0.7326989860019331, -0.4589997720511263, 0.3925029176153736, 0.9244538380742333]
        if method_name == 'tb_gfn':
            args.exp_weight = 0.

    envs = [SetEnv(args.action_dim, args.set_size, intermediate_energies) for i in range(args.bufsize)]

    if args.method == 'tb_gfn':
        agent = TBFlowNetAgent(args, envs)
    elif args.method == 'db_gfn':
        agent = DBFlowNetAgent(args, envs)

    if args.method == 'tb_gfn':
        opt = torch.optim.Adam([{'params': agent.parameters(), 'lr': args.tb_lr}, {'params':[agent.Z], 'lr': args.tb_z_lr} ])
    elif args.method == 'db_gfn':
        opt = torch.optim.Adam([{'params': agent.parameters(), 'lr': args.tb_lr}])

    for i in tqdm(range(args.n_train_steps + 1), disable=not args.progress):
        experiences = agent.sample_many(args.mbsize)
        agent.sample_many(args.mbsize, evaluate=True)

        if method_name == 'fl_db_gfn':
            losses = agent.learn_from_fl(i, experiences) 
        else:
            losses = agent.learn_from(i, experiences) 

        losses[0].backward()
        opt.step()
        opt.zero_grad()

        if i % 10 == 0:
            all_unique_Rs = sorted(agent.all_unique_Rs, reverse=True)
            all_unique_Rs = np.array(all_unique_Rs)

            top_k_Rs = all_unique_Rs[:100]
            mean_top_k_R = sum(top_k_Rs) / len(top_k_Rs)
            if args.wdb:
                wandb.log({'mean_top_k_R': mean_top_k_R})

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args)
