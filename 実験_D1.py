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
target_state = [
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,1,1,1,
    1,1,1,1,1,1,1,0,1,0,
    0,0,0,0,0,0,1,1,1,0,
    0,0,0,0,0,0,1,
    ]
#target_state = np.zeros(67)
cdict = {v:val for val,v in zip(target_state, sorted(set(f_dict) - set(controller)))}


# BDD取得時間,経路数
N = 20
data = np.empty((N,3))

with open('data.csv', 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['BDD取得時間','経路数'])


for i in range(N):
    B = BooleanNetwork.BooleanNetwork()
    T = i+1
    t = time()
    BN = BCN_transition(f_dict, controller, T)
    #BN_bdd = {v:B.GetBDD(root) for v,root in BN.items()}
    root_cat = BN_concatenate(BN, cdict)
    #bdd = B.GetBDD(root_cat)
    t = time() - t
    
    #集計
    V_ordered = list(range(6,72+1+len(controller)*T))
    num_path1 = B.CountPath(root_cat)
    num_path2 = B.CountPath(root_cat, V_ordered)
    #num_path = 0
    data[i] = [t, num_path1, num_path2]
    #"""
    with open('data.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data[i])
    #"""
    print(i)



#　初期値を与える
initial_state = np.zeros(67)
root_cat_assign = B.AssignConst(root_cat,{v:val for v,val in zip(f_dict,initial_state)})


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