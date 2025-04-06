# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 20:30:36 2024

@author: fmotoyama
"""
import itertools, pickle
import numpy as np
import torch

from BooleanNetwork import BooleanNetwork
from attractor_classification import Net, Config

load_path = 'checkpoints/f_dict_30_2024-10-19-18-54-38/'

# アトラクターを取得
with open(load_path + 'Dataset.pkl', 'rb') as f:
    attractors,Dataset,basins_bdd = pickle.load(f)

# ネットワークを取得
with open(load_path + 'config.pkl', 'rb') as f:
    config = pickle.load(f)
config.output_size = len(attractors) #!!!
net = Net(config).to(device='cuda').float()
net.load_state_dict(torch.load(load_path + 'f_dict_30_100000.chkpt'))

# ピニング制御
idx_tgt = 0
l = len(Dataset['X'][0])
P = np.zeros(l, dtype=np.bool_)     # ピニングノード
S = set([idx_tgt])                  # 目標状態に接続されたアトラクターグループ
C = dict()                          # C[idx] = [ジャンプ元state,ジャンプ先idx,ジャンプ先state]

def search1(idx,attractor):
    # 決定したピニングノードを優先的に使用し、attractorをSへジャンプさせる
    # ピニングノードの少ないものから順に探索
    for l1 in range(l-sum(P)+1):
        print(f'\r ピニングノード数: {l1}')
        for p1_idx in itertools.combinations(np.where(~P)[0],l1):
            for l2 in range(sum(P)+1):
                if l1 + l2 == 0:
                    continue
                for p2_idx in itertools.combinations(np.where(P)[0],l2):
                    p = np.zeros(l, dtype=np.bool_)
                    p[list(p1_idx)] = True
                    p[list(p2_idx)] = True
                    if sum(p) == 0:
                        continue
                    for state in attractor:
                        state_p = state ^ p
                        idx_to = np.argmax(net(torch.tensor(state_p, device='cuda').float()).to('cpu').detach().numpy())
                        if idx_to in S:
                            return p, state, idx_to, state_p
def search2(idx,attractor):
    # 決定したピニングノードを優先的に使用し、attractorをSへジャンプさせる
    # ピニングノードの列挙方法が簡素
    for p1 in itertools.product([0,1], repeat=l-sum(P)):
        print(f'\r ピニングノード数: {len(p1)}')
        for p2 in itertools.product([0,1], repeat=sum(P)):
            p = np.zeros(l, dtype=np.bool_)
            p[~P] = p1
            p[P] = p2
            if sum(p) == 0:
                continue
            for state in attractor:
                state_p = state ^ p
                idx_to = np.argmax(net(torch.tensor(state_p, device='cuda').float()).to('cpu').detach().numpy())
                if idx_to in S:
                    return p, state, idx_to, state_p





for idx,attractor in enumerate(attractors):
    if idx == idx_tgt:
        continue
    print(f'attractor {idx}')
    p, state, idx_to, state_p = search2(idx,attractor)
    # 接続が成功したとき
    P |= p
    S.add(idx)
    C[idx] = [state, idx_to, state_p]
    print()

# 検証 Cのジャンプ先idx:ジャンプ先stateが一致していることを確認
B = BooleanNetwork()
check = [B.AssignConstAll(basins_bdd[idx],{i+1: int(v) for i,v in enumerate(state)}) for _, idx, state in C.values()]
assert len(check) == len(attractors)-1










