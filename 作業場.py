# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 17:53:43 2021

@author: f.motoyama
"""
import sys
from copy import copy

import BooleanNetwork
import QuineMcCluskey as QM
import drawset


B = BooleanNetwork.BN()
"""
#問題の式
f_dict = {}
f_dict[1] = [[1,2,3]]
f_dict[2] = [[1],[2,3]]
f_dict[3] = [[-2],[3]]

f_dict2 = {}
f_dict2[1] = [[1,3]]
f_dict2[2] = [[2,3]]
f_dict2[3] = [[-2],[3]]


BN = B.get_BN(f_dict)
TT, SS = B.MakeTrueTable(BN)

BN2 = B.get_BN(f_dict2)
TT2, SS2 = B.MakeTrueTable(BN2)
"""

#"""
f_dict = {}
f_dict[1] = [[1]]
f_dict[2] = [[-1,2]]
f_dict[3] = [[1,2,3]]
fvs = [1,2,3]

"""
#式の簡単化
for v in f_dict:
    ttr, V = B.MakeTrueTable_node_f(B.get_BN(f_dict)[v], 'r')
    print(f'{v}:{QM.ttr_to_f(ttr, V)}')
#"""
TT,SS = B.MakeTrueTable(B.get_BN(f_dict))


#"""
f_dict = {}
f_dict[1] = [[1]]
f_dict[2] = [[2]]

#描画
BN = B.get_BN(f_dict)
drawset.wiring_diagram(B.BN_to_parent(BN), 'BN_')
#"""















