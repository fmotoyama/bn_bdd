# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:25:53 2020

@author: motoyama
"""

import copy
import numpy as np

from BDD import BDD, Root, Leaf
#import drawset


class BooleanNetwork(BDD):
    def __init__(self):
        super().__init__()
        """
        f       : 論理式(list)
        f_dict  : BN式(dict)
        v       : 変数名
        BN      : BNのBDD(dict)
        node    : (nodeクラス)
        """
        
    def GetBN(self,f_dict):
        BN = {}
        for v in f_dict:
            if f_dict[v] == []:
                BN[v] = None
            else:
                BN[v] = self.calc_f(f_dict[v])
        return BN
    
    
    def GetSS(self, BN, check=False):
        """
        BNの定常状態を求める
        式が与えられていない変数の値は何でもよい
        """
        V_ordered = list(BN)
        l = len(V_ordered)
        
        def scan(SS, v_root, node, sign):
            """node以下の情報をSSに反映させる"""
            if type(node) is Leaf:
                # V.index(v_root)列でnot signの値をもつ行を除き、同列(の-1)をsignに書き換える
                SS = SS[np.where(SS[:,V_ordered.index(v_root)] != int(not sign))[0],:]
                SS[:,V_ordered.index(v_root)] = int(sign)
                return SS
            
            v = node.v
            v_id = V_ordered.index(v) if v in V_ordered else None
            # 0枝
            if v_id != None:
                # v列において値が1の行を除き、同列(の-1)を0に書き換える
                SS_temp = SS[np.where(SS[:,v_id] != 1)[0],:]
                SS_temp[:,v_id] = 0
                SS0 = scan(SS_temp, v_root, node.n0, sign == (not node.neg))
            else:
                SS0 = scan(SS, v_root, node.n0, sign == (not node.neg))
            # 1枝
            if v_id != None:
                # v列において値が0の行を除き、同列(の-1)を1に書き換える
                SS_temp = SS[np.where(SS[:,v_id] != 0)[0],:]
                SS_temp[:,v_id] = 1
                SS1 = scan(SS_temp, v_root, node.n1, sign)
            else:
                SS1 = scan(SS, v_root, node.n1, sign)
            
            return np.concatenate([SS0,SS1])
        
        SS = np.full((1,l), -1, dtype = 'i1')
        for v_root,root in BN.items():
            if type(root) is Root:
                SS = scan(SS, v_root, root.node, not root.neg)
        
        if check:
            assert not self.CheckSS(BN,SS,V_ordered), 'GetSS: SS wrong'
        return SS
    
    
    def CheckSS(self, BN, SS):
        """SSがBNで固定点となっているかチェックする"""
        V_ordered = list(BN)
        for state in SS:
            cdict = {v:value for v,value in zip(V_ordered,state)}
            for i,v in enumerate(V_ordered):
                output = self.AssignConst(BN[v],cdict)
                if type(output.node) is not Leaf:
                    if set(V_ordered) & self.GetV(output):
                        return state
                if state[i] != (not output.neg):
                    return state    
        return 0
    
    
    
    def BN_to_f_dict(self, BN, check = False):
        """
        BN(BDDの集合)からf_dictを求める
        check == Trueのとき、求めたf_dictからBNを作成し、元と変わらないか確認する
        """
        f_dict = {v:self.BDD_to_f(root) for v,root in BN.items()}
        if check:
            BN2 = self.GetBN(f_dict)
            for v in BN:
                assert (BN[v].node == BN2[v].node) and (BN[v].neg == BN2[v].neg), f'BN_to_f_dict: failed v={v}'
        return f_dict
    
    
    def BN_to_parent(self,BN):
        """BNからparentを返す"""
        return {v:self.GetV(root) for v,root in BN.items()}
        
    
    def nextf(self, BN, f_dict, n = 1):
        """BNに対してf_dictの演算をn回行う"""
        BN2 = copy.copy(BN)
        for i in range(n):
            BN_temp = {v:self.calc_f(f_dict[v], BN2) for v in BN2}
            BN2 = copy.copy(BN_temp)
        return BN2
    
    
    def SimplifyBN_const(self, BN):
        """定数となっている関数を代入する"""
        while True:
            #定数である関数を探し、clistを完成させる
            cdict = {}
            for v,root in BN.items():
                if type(root.node) is Leaf:
                    cdict[v] = not root.neg
            #clistが空のとき、終了
            if cdict == {}:
                return BN
            #clistを全関数に代入
            BN2 = {}
            for v in BN:
                if type(BN[v].node) is not Leaf:
                    BN2[v] = self.AssignConst(BN[v],cdict)
            BN = BN2
    
    
    def BDD_to_f(self, root):
        """
        BDDを見て、1に到達する状態の組み合わせから積和標準形のfを得る
        fは最小とは限らない
        """
        f = []
        states,V = self.EnumPath(root, shorten=True)
        for state in states:
            term = [v * [-1,1][bool(value)] for v,value in zip(V,state) if value != -1]
            f.append(term)
        return f
    
    
    def BN_to_TT(self, BN):
        V = list(BN)
        l = len(V)
        TT = np.empty((2, 2**l, l), dtype='i1')
        
        # 真理値表の左側
        for col in range(l):
            TT[0,:,col] = ([0]*2**col + [1]*2**col) * 2**(l-col-1)
        # 真理値表の右側
        for col,root in enumerate(BN.values()):
            states_small, V_small = self.EnumPath(root, shorten=False)
            cols = [V.index(v) for v in V_small]
            for state_small in states_small:
                rows = np.where(np.all(TT[0,:,cols].T == state_small, axis=1))[0]
                TT[1,rows,col] = 1
        
        return TT
    
    
    def TT_to_f_dict(self, TT, V_ordered=None):
        """TT : (2, 2**l, l)"""
        f_dict = {}
        l = TT.shape[2]
        if not V_ordered:
            V_ordered = np.arange(1,l+1)
        for col,v in enumerate(V_ordered):
            rows = np.where(TT[1,:,col])[0]
            f = []
            for row in rows:
                term = [V_ordered[col2]*(-1,1)[v2] for col2,v2 in enumerate(TT[0,row])]
                f.append(term)
            f_dict[v] = f
        return f_dict
    
    
    def f_dict_to_TT(self,f_dict):
        """fから直接計算する"""
        V = list(f_dict)
        l = len(V)
        TT = np.empty((2, 2**l, l), dtype='i1')
        # 真理値表の左側
        for col in range(l):
            TT[0,:,col] = ([0]*2**col + [1]*2**col) * 2**(l-col-1)
        # 真理値表の右側
        for col in range(l):
            TT[1,:,col] = self.calc_on_f(f_dict[V[col]],TT[0],V)
        return TT
    
    def calc_f(self, f:list, bdds:dict=dict()) -> Root:
        """BDDについて関数fの演算を行う"""
        # fが空のとき
        if f == []:
            return None
        # fが定数のとき
        if type(f) is int:
            return f
        
        def calc_product(term:list):
            # termがもつ変数を全て積算する
            assert len(term)
            root = Root(self.leaf, False)
            for v in term:
                root2 = bdds.get(abs(v))
                if not root2:
                    # BNにない変数を使おうとする場合、その変数を表すBDDを自動で用意する
                    root2 = self.GetUnitNode(abs(v),v<0)
                else:
                    root2 = Root(root2.node, root2.neg ^ (v<0))
                root = self.AP('AND', root, root2)
            return root
        
        #先頭のtermのBDDを生成
        root_sum = calc_product(f[0])
        #termごとにBDDを生成し、sum_BDDに和算する
        for term in f[1:]:
            root_sum = self.AP('OR', root_sum, calc_product(term))
        return root_sum
    
    
    @staticmethod
    def calc_on_f(f, states, V):
        """fをそのまま用いて計算する"""
        output = []
        for state in states:
            y = 0
            for term in f:
                y_term = 1
                for x in term:
                    val = state[V.index(abs(x))]
                    val = not val if x < 0 else val
                    y_term = y_term and val
                y = y or y_term
            output.append(y)
        return np.array(output, dtype='i1')
    
    
    @staticmethod
    def is_equal_BN(BN1,BN2):
        for (v1,root1),(v2,root2) in zip(BN1.items(),BN2.items()):
            assert v1 == v2
            assert root1.neg == root2.neg
            assert root1.node == root2.node
        return 0
    
    @staticmethod
    def save_f_dict(f_dict, fvs = None, name = 'f_dict'):
        """f_dictをtxtファイルとして出力する"""
        #shutil.rmtree('f_dict.txt')
        with open(f'{name}.txt', mode='w') as f:
            f.write(str(f_dict))
            if fvs:
                f.write('\n')
                f.write(str(fvs))
    

if __name__ == '__main__':
    f_dict = {
        1:[[2],[-3]],
        2:[[1],[2,3]],
        3:[[1,2],[1,-2,3],[-1,2,3]],
        #3:[[1,-2]],
        
        #1:[[1]],
        #2:[[2]],
        #3:[[3]]
        }
    
    
    B = BooleanNetwork()
    BN = B.GetBN(f_dict)
    
    a = type(BN[2]) == Root
    
    BN_BDD = {v:B.GetBDD(root) for v,root in BN.items()}
    f_dict2 = B.BN_to_f_dict(BN)    # f_dict → BDD → f_dict
    TT = B.BN_to_TT(BN)
    f_dict3 = B.TT_to_f_dict(TT)    # f_dict → BDD → TT → f_dict
    
    #BN_BDD = {v:B.GetBDD(node) for v,node in BN.items()}
    #table = B.table
    #drawset.binary_tree(table,'table')
    
    
    
    

