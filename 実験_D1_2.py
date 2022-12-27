# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:33:43 2021

@author: f.motoyama
"""
import copy, random, csv
import numpy as np
from time import time

from BDD import Root
import BooleanNetwork, drawset, def_f
from QuineMcCluskey import QM2

#制御の実験

B = BooleanNetwork.BooleanNetwork()


def BN_concatenate(BN, cdict):
    """BNが目標状態に到達するための条件を表すBDDを得る"""
    root_cat = B.GetLeafNode(neg=False)
    for v,root in BN.items():
        temp = root if cdict[v] else Root(root.node, not root.neg)
        root_cat = B.AP('AND', root_cat, temp)
    return root_cat


def BCN_transition(f_dict:dict, controller:list, T:int):
    """
    コントローラである変数の変数名を毎時刻変更しながら状態遷移を行う
    """
    f_dict_sub = copy.deepcopy(f_dict)
    for n in controller:
        del f_dict_sub[n]
    BN = {v:B.GetUnitNode(v,neg=False) for v in f_dict_sub}
    v_last = max(f_dict_sub)    # 現在の最後尾の変数名
    
    for t in range(1, T+1):
        # コントローラである変数に、新たな変数名を与える
        V_converter = {v:v_last+1+i for i,v in enumerate(controller)}
        f_dict_temp = dict()
        for v,f in f_dict_sub.items():
            f_temp = []
            for term in f:
                term_temp = [V_converter[abs(v)] * [-1,1][0<v] if abs(v) in controller else v for v in term]
                f_temp.append(term_temp)
            f_dict_temp[v] = f_temp
        
        # 時間を1進める
        BN = B.nextf(BN, f_dict_temp, n=1)
        #BN_bdd = {v:B.GetBDD(root) for v,root in BN.items()}
        v_last += len(controller)
    
    return BN


# データ定義
f_dict, _ = def_f.def_f('import','f_dict_72')
controller = (1,2,3,4,5)    # コントローラに指定するノード　これらのノードの状態は見ない
#f_dict = {1:[[1,3,4],[2]],2:[[3],[4]],3:[[-1]],4:None}
#controller=(4,)


# 目標状態を設定
steady_state = B.GetSS(B.GetBN(f_dict))
target_state = steady_state[0]
#target_state = np.random.randint(0,2,len(f_dict),dtype='b')
target_state = np.array([
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,1,1,1,
    1,1,1,1,1,1,1,0,1,0,
    0,0,0,0,0,0,1,1,1,0,
    0,0,0,0,0,0,1,
    ])
#target_state = np.zeros(67)
cdict = {v:val for val,v in zip(target_state, sorted(set(f_dict) - set(controller)))}


# BDDの時間遷移・合成
B = BooleanNetwork.BooleanNetwork()
T = 10
#BN = BCN_transition(f_dict, controller, T)
#root_cat = BN_concatenate(BN, cdict)


# 得られた組み合わせに基づく時間遷移
"""
flag = 0
while not flag:
    path = B.PickPath(root_cat)
    for v,value in path.items():
        if 72<v and value==1:
            flag = 1
            break
"""
path = {122: 0, 121: 0, 120: 0, 119: 0, 117: 0, 116: 0, 115: 0, 114: 0, 112: 0, 111: 0, 110: 0, 109: 0, 107: 0, 106: 0, 105: 0, 104: 0, 102: 0, 101: 0, 100: 0, 99: 0, 97: 0, 96: 0, 95: 0, 94: 0, 92: 0, 91: 0, 90: 0, 89: 0, 87: 0, 86: 0, 85: 0, 84: 0, 82: 0, 81: 0, 80: 0, 79: 0, 76: 0, 75: 0, 74: 0, 69: 1, 68: 0, 67: 0, 66: 0, 64: 0, 63: 0, 62: 1, 61: 1, 59: 0, 55: 1, 49: 1, 48: 1, 47: 0, 46: 1, 45: 0, 44: 0, 43: 1, 41: 0, 40: 0, 39: 1, 38: 0, 37: 0, 36: 0, 35: 0, 34: 0, 33: 0, 32: 0, 31: 0, 30: 0, 29: 0, 28: 0, 26: 1, 25: 0, 24: 0, 23: 0, 22: 1, 20: 0, 19: 0, 18: 0, 17: 0, 16: 0, 14: 0, 13: 0, 12: 0, 10: 0, 9: 0, 8: 1, 7: 0, 6: 0}
#path = {122: 0, 121: 0, 120: 0, 119: 0, 117: 0, 116: 0, 115: 0, 114: 0, 112: 0, 111: 0, 110: 0, 109: 0, 107: 0, 106: 0, 105: 0, 104: 0, 102: 0, 101: 0, 100: 0, 99: 0, 97: 0, 96: 0, 95: 0, 94: 0, 92: 0, 91: 0, 90: 0, 89: 0, 87: 0, 86: 0, 85: 0, 84: 0, 82: 0, 81: 0, 80: 0, 79: 0, 76: 0, 75: 0, 74: 0, 70: 0, 69: 1, 68: 0, 67: 0, 66: 0, 65: 1, 64: 0, 63: 1, 62: 0, 61: 0, 60: 0, 56: 0, 55: 0, 49: 1, 48: 0, 47: 1, 46: 0, 45: 0, 44: 0, 43: 1, 41: 0, 40: 0, 39: 0, 38: 0, 37: 0, 36: 0, 35: 0, 34: 0, 33: 0, 32: 0, 31: 0, 30: 0, 29: 0, 28: 0, 26: 1, 25: 0, 24: 0, 23: 0, 22: 1, 20: 0, 19: 0, 18: 0, 17: 0, 16: 0, 15: 0, 14: 0, 13: 0, 12: 0, 10: 0, 9: 0, 8: 0, 7: 0, 6: 0}
#path = {122: 0, 121: 0, 120: 0, 119: 0, 117: 0, 116: 0, 115: 0, 114: 0, 111: 0, 110: 0, 109: 0, 107: 0, 106: 0, 105: 0, 104: 0, 102: 0, 101: 0, 100: 0, 99: 0, 97: 1, 96: 0, 95: 0, 94: 0, 91: 0, 90: 0, 89: 0, 87: 0, 86: 0, 85: 0, 84: 0, 81: 0, 80: 0, 79: 0, 77: 1, 76: 0, 75: 0, 74: 0, 70: 0, 69: 0, 68: 1, 67: 0, 66: 0, 65: 1, 63: 1, 62: 1, 61: 1, 60: 0, 56: 1, 49: 0, 48: 0, 47: 1, 46: 0, 45: 1, 44: 1, 43: 1, 41: 0, 40: 0, 39: 1, 38: 0, 37: 0, 36: 0, 35: 0, 34: 0, 33: 0, 32: 0, 31: 0, 30: 0, 29: 0, 28: 0, 26: 1, 25: 0, 24: 0, 23: 0, 22: 1, 20: 0, 19: 0, 18: 0, 17: 0, 16: 0, 15: 0, 14: 0, 13: 0, 12: 0, 10: 0, 9: 0, 8: 0, 7: 0, 6: 0}
#initial_state = {v:path[v] for v in path if v in f_dict}
initial_state = {v:path[v] if v in path else np.random.randint(0,2) for v in range(6,73)}
#controll_sequence = {v:path[v] for v in path if v not in f_dict}
controll_sequence = {v:path[v] if v in path else np.random.randint(0,2) for v in range(73,123)}

def BN_controll(f_dict, controller, T, initial_state, controll_sequence):
    """初期状態に対して制御入力列を適用して時間遷移させる"""
    initial_state = {v:B.GetLeafNode(not value) for v,value in initial_state.items()}
    controll_sequence = {v:B.GetLeafNode(not value) for v,value in controll_sequence.items()}
    BNs = {0:initial_state}
    combination = {**initial_state, **controll_sequence}
    
    f_dict_sub = copy.deepcopy(f_dict)
    for n in controller:
        del f_dict_sub[n]
    v_last = max(f_dict_sub)    # 現在の最後尾の変数名
    
    for t in range(1, T+1):
        # コントローラである変数に、新たな変数名を与える
        V_converter = {v:v_last+1+i for i,v in enumerate(controller)}
        f_dict_temp = dict()
        for v,f in f_dict_sub.items():
            f_temp = []
            for term in f:
                term_temp = [V_converter[abs(v)] * [-1,1][0<v] if abs(v) in controller else v for v in term]
                f_temp.append(term_temp)
            f_dict_temp[v] = f_temp
        
        # 時間を1進める
        BNs[t] = {v:B.calc_f(f,combination) for v,f in f_dict_temp.items()}
        combination = {**BNs[t], **controll_sequence}
        v_last += len(controller)
    return BNs

BNs_assigned = BN_controll(f_dict, controller, T, initial_state, controll_sequence)
BNs_assigned_bdd = {t:{v:B.GetBDD(root) for v,root in BN_assigned.items()} for t,BN_assigned in BNs_assigned.items()}


# ハミング距離を求める
V_sub = sorted(set(f_dict) - set(controller))
states = np.full((T+1,len(V_sub)), -1, dtype='i1')
#for i,BN_assigned in enumerate(BNs_assigned_bdd.values()):
for i,BN_assigned in enumerate(BNs_assigned.values()):
    for v,val in BN_assigned.items():
        if type(val) is not int:
            continue
        states[i,V_sub.index(v)] = val

hamming_distance = []
for state in states:
    idx = np.where(state!=-1)[0]
    hamming_distance.append(np.count_nonzero(state[idx] != target_state[idx]))



#"""
controll_sequence2 = []
for t in range(10):
    c2 = []
    for i in range(5):
        index = 72+1 + i + t*5
        if index not in controll_sequence:
            c2.append('*')
        else:
            c2.append(controll_sequence[index])
    controll_sequence2.append(c2)
initial_state2 = []
for v in range(6,73):
    index = 72+1 + i + t*5
    if v not in initial_state:
        initial_state2.append('*')
    else:
        initial_state2.append(initial_state[v])
#"""
#f = QM2(B.BDD_to_f(node_conv))

"""
# 描画
#drawset.wiring_diagram(B.BN_to_parent(BN_1))
v1 = list(f_dict_sub)[-1]
v2 = v_last
bdd = B.GetBDD2(abs(node_conv))
drawset.binary_tree(
    bdd,
    f'BDD_conv(t+{i})',
    {i:f'u_{(i-v1-1)%len(controller)+1}({(i-v1-1)//len(controller)})' for i in range(v1+1,v2+1)}
    )
#"""


"""
#node_convに初期値を代入する
node2 = B.AssignConst(node_conv,cdict)
drawset.binary_tree(
    B.GetBDD(abs(node2)),
    'BDD_conv+init',
    {v_last+i2:f'u{i2}' for i2 in range(1,i+1)}
    )
#"""


"""
#node_convで、制御入力への接続を1に接続する　このBDDで1に到達する状態は可制御
bdd = B.GetBDD(abs(node_conv))
node_input = [n for n,info in bdd.items() if info[0] not in f_dict_sub]

def scan(node, leaf_sign):
    '''node以下で、制御入力への接続をleaf_signに接続する'''
    key = abs(node)
    sign = (0<node)*2-1
    if key == 1:
        return node
    info = bdd[key]
    info = list(info)
    for i in [1,2]:
        if abs(info[i]) in node_input:
            info[i] = leaf_sign
        else:
            info[i] = scan(info[i], leaf_sign*((0<info[i])*2-1))
    return B.GetNode(*info, sign=sign)

node3 = scan(node_conv,(0<node_conv)*2-1)
bdd = B.GetBDD(abs(node3))
drawset.binary_tree(bdd, 'BDD_conv_stability')
f = B.BDD_to_f(node3)
#f2 = QM2(f)
#"""






#temp1 = {key:info for key,info in B.table.items() if (info[1]<0 and info[1]!=-1)}
#temp2 = {key:info for key,info in bdd.items() if (info[1]<0 and info[1]!=-1)}

#T = 2 : 1.0010182857513428, len(table)=15262, len(F)=
#T = 3 : 0.29781460762023926, len(table)=, len(F)=
#T = 4 : 20.41803741455078, len(table)=, len(F)=