# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:25:50 2024

@author: fmotoyama
"""
import itertools, sys
from time import time
import numpy as np

from BooleanNetwork import BooleanNetwork
#from BooleanNetwork2 import BooleanNetwork
import drawset, def_f

rng = np.random.default_rng(0)


def search_jump(attractor_bdd, basin_bdd):
    pass


def solution1(attractors_bdd, basins_bdd, id_tgt: int, P: set = set()):
    """貪欲法"""
    Br = basins_bdd[id_tgt]
    P_state = np.zeros(l, dtype=np.bool_)
    P_state[[i-1 for i in P]] = True
    C = []  # [ジャンプ元アトラクタid, ジャンプ元状態, ジャンプ先状態]
    for id_attractor, attractor_bdd in enumerate(attractors_bdd):
        if id_attractor == id_tgt:
            continue
        # 状態に直す
        attractor,_ = B.EnumPath(attractor_bdd, V, shorten=False)
        basin,_ = B.EnumPath(Br, V, shorten=True)
        # ピニングノードでない部分だけを見る
        attractor2 = attractor[:,~P_state]
        basin2 = basin[:,~P_state]
        # 最短距離のジャンプを探す
        distance_min = sum(~P_state)
        for stateA1, stateA2 in zip(attractor, attractor2):
            for stateB1, stateB2 in zip(basin, basin2):
                # 距離を求める
                state_diff = (stateA2 != stateB2) & (stateB2 != -1)
                if sum(state_diff) < distance_min:
                    distance_min = sum(state_diff)
                    c = [id_attractor, stateA1, stateB1]
        
        # stateB1の-1にstateA1の値を割り当てる
        c[2][c[2]==-1] = c[1][c[2]==-1]
        C.append(c)
        P_state |= c[1] != c[2]
        Br |= basins_bdd[id_attractor]
        
    return P_state, C
                    
                
                

def solution2(attractors_bdd, basins_bdd, id_tgt, P):
    """
    最適制御を求める
    考えなければいけないパターンは、目標状態でないBasinの個数をAとして、
        (全てのジャンプパターン) - (目標状態に到達しないジャンプパターン)
        = A^A - (A-1)^A
    """

if __name__ == '__main__':
    B = BooleanNetwork()
    
    name = 'f_dict_30'
    name = 'f_dict_12_hakushi'
    #f_dict,_ = def_f.def_f('random','normal',n=3,prob=0.5)
    #f_dict,_ = def_f.def_f('import','f_dict_5')
    #f_dict,_ = def_f.def_f('import','f_dict_15')
    #f_dict,_ = def_f.def_f('import','f_dict_18')
    f_dict,_ = def_f.def_f('import',name)
    BN = B.GetBN(f_dict)
    V = list(f_dict)
    l = len(f_dict)
    #drawset.wiring_diagram(B.f_dict_to_parent(f_dict), f'wd_{name}')
    #drawset.transition_diagram(B.BN_to_TT(BN), f'td_{name}', format='pdf')
    #drawset.binary_tree(B.GetBDDdict(B.GetTransition(BN)), f'bt_{name}', node_name={1:'x_1^A',2:'x_2^A',3:'x_3^A',4:'x_1^B',5:'x_2^B',6:'x_3^B',}, format='pdf')
    #x_space = np.array(list(itertools.product([1,0], repeat=l)), dtype=np.bool_)
    B.save_f_dict(f_dict, name = name)
    
    t = time()
    attractors_bdd = B.GetAttractor(f_dict)
    print(time() - t)#; sys.exit()
    attractors = [B.EnumPath(attractor_bdd, V)[0] for attractor_bdd in attractors_bdd]
    #[drawset.binary_tree(B.GetBDDdict(attractor_bdd), f'bt_{i}', node_name={'1':'x1','2':'x2','3':'x3',}, format='svg') for i,attractor_bdd in enumerate(attractors_bdd)]
    basins_bdd = [B.GetBasin(BN, attractor_bdd) for attractor_bdd in attractors_bdd]
    #[drawset.binary_tree(B.GetBDDdict(basin_bdd), f'bt_basin{i}', node_name={'1':'x1','2':'x2','3':'x3',}, format='svg') for i,basin_bdd in enumerate(basins_bdd)]
    #drawset.binary_tree(B.GetBDDdict(basins_bdd[0] |basins_bdd[1] | basins_bdd[2]), 'bt_basin', node_name={'1':'x1','2':'x2','3':'x3',}, format='svg')
    
    # 目標状態basinのid
    id_tgt = 1
    P_state, C = solution1(attractors_bdd, basins_bdd, id_tgt)
    
    # ジャンプ元、ジャンプ先
    A_ = np.vstack([c[1] for c in C]).astype(np.bool_)
    B_ = np.vstack([c[2] for c in C]).astype(np.bool_)
    #"""
    # コントローラーを適用した状態遷移図の描画
    def state2idx(state: np.ndarray):
        return sum((2**i)*v for i,v in enumerate(np.flip(~state)))
    TT_controlled = B.BN_to_TT(BN)
    TT_controlled[1,[state2idx(state) for state in A_]] = B_
    drawset.transition_diagram(TT_controlled, f'td_{name}_controlled', format='pdf')
    #"""
    """
    # 各ピニングノードの計算に必要な、A_の前の状態を求める
    from QuineMcCluskey import QM
    controller = []
    for idx in np.where(P_state)[0]:
        A_sub = A_[A_[:,idx] != B_[:,idx]]
        node = B.leaf0
        for state in A_sub:
            node |= B.GetLineBDD({k+1:v for k,v in enumerate(state)})
        controller.append(B.BDD_to_f(B.Backward(BN, node)))
    #"""
    
    
    
    