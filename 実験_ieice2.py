# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:26:55 2019

@author: motoyama
"""

import BooleanNetwork
from def_f import def_f
from QuineMcCluskey import QM2
import drawset


#関数を定義
f_dict = {
    'ex2':{
        1:[[-6]],
        2:[[-2,4,6],[2,4],[2,6]],
        3:[[-7]],
        4:[[4]],
        5:[[2],[-7]],
        6:[[3],[4]],
        7:[[-2],[7]],
        },
    'ex2_r':{
        2:[[2,-7],[4]],
        4:[[4]],
        7:[[-2],[7]],
        },
    'ex3':{
        1:[[1]],
        2:[[-1,2]],
        3:[[1,2,3]],
        },
    'ex4':{
        1:[[1],[-4]],
        2:[[2],[1,-5],[-3,-4]],
        3:[[3],[4]],
        4:[[1],[2],[4],[5]],
        5:[[4],[5]]
        }
    }

#Bインスタンスを作成して、BNをBDDの集合として扱う　（+　fvsを求める）
B = BooleanNetwork.BooleanNetwork()
BN = B.GetBN(f_dict['ex3'])
#[drawset.binary_tree(B.GetBDD(value),f'f{key}') for key,value in BN.items()]

#真理値表を得る
TT = B.MakeTT(BN)

#BNの定常状態を求める
V, SS = B.GetSS(BN, check=True)

#BNの式を得る
f_dict2 = B.BN_to_f_dict(BN, check=True)
f_dict2 = {key:QM2(f) for key,f in f_dict2.items()}

#BN,r_BNのインタラクショングラフを描画
#drawset.wiring_diagram(B.BN_to_parent(BN),"BN")




#B.save_f_dict(f_dict, fvs)









