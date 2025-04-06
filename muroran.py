# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:11:54 2024

@author: fmotoyama
"""

from BooleanNetwork import BooleanNetwork
from QuineMcCluskey import QM
import numpy as np
rng = np.random.default_rng()

TTl = np.array([    
    [1,1,1],
    [1,1,0],
    [1,0,1],
    [1,0,0],
    [0,1,1],
    [0,1,0],
    [0,0,1],
    [0,0,0],
    ], dtype=np.bool_)
rng.shuffle(TTl)

TTr = TTl[[1,1,3,4,3,6,7,7]]

TT = np.stack([TTl,TTr])

B = BooleanNetwork()
f_dict = {k:QM(v) for k,v in B.TT_to_f_dict(TT).items()}
print(f'{f_dict[1]}\n{f_dict[2]}\n{f_dict[3]}')
B.save_f_dict(f_dict, name = 'f_dict_3_hakushi')
