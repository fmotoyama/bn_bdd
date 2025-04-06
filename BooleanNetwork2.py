# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:25:53 2020

@author: motoyama
"""

import copy, itertools
import numpy as np

from BDD_not2 import BDD, Node, Leaf


class BooleanNetwork(BDD):
    def __init__(self):
        super().__init__()
        """
        f       : 論理式(list)
        f_dict  : 式をlistで表したBN(dict)
        v       : 変数名(int)
        BN      : 式をBDDで表したBN(dict)
        node    : (Nodeクラス)
        """
        
    def GetBN(self,f_dict):
        return {v:self.calc_f(f) for v,f in f_dict.items()}
    
    
    def GetFixedAttractor_simple(self, BN):
        """真理値表を書いて固定アトラクターを見つける"""
        TT = self.BN_to_TT(BN)
        return TT[0,np.all(TT[0] == TT[1], axis=1)]
        
    
    def GetFixedAttractor(self, BN, node_set=None):
        """node_setの表す状態のうち、BNの固定点となるものを探す"""
        if node_set is None:
            node_set = self.leaf1
        for v,node in BN.items():
            node_set = (node_set & node & self.GetUnitNode(v)) | (node_set & ~node & ~self.GetUnitNode(v))
        return node_set
    
    def GetAttractor(self, f_dict, BN=None):
        """
        固定/周期アトラクターを求める
        t時刻後の状態集合中の固定アトラクタ―は、周期tの周期アトラクターである
        """
        BN = BN if BN else self.GetBN(f_dict)
        V = list(BN)
        node_transition = self.GetTransition(BN)
        
        node_frontier = self.leaf1
        attractors_bdd = {}
        count = 1
        while node_frontier != self.leaf0:
            # 固定点を取得
            node_fixed = self.GetFixedAttractor(BN, node_frontier)
            if node_fixed is not self.leaf0:
                attractors_bdd[count] = node_fixed
            # 固定点に到達する状態をさかのぼり、それらをnode_frontierから削除
            node_frontier = node_frontier & ~self.GetBasin(BN, node_fixed)
            # 次時刻の状態を求める
            BN = self.nextf(BN, f_dict)
            count += 1
            
        # BDDの形のアトラクターを分解する
        attractors = []
        for size,node in attractors_bdd.items():
            while node != self.leaf0:
                # 状態を一つ取り出す
                node_start = self.GetLineBDD(self.PickPath(node,V))
                node_attractor = node_start
                node = node & ~node_start
                # ループするまで状態遷移
                node_frontier = node_start
                for _ in range(size):
                    node_frontier = self.Forward(node_frontier, V, node_transition)
                    node_attractor |= node_frontier
                assert node_start == node_frontier
                # 求めたアトラクターを記録
                attractors.append(node_attractor)
                node = node & ~node_attractor
        return attractors
        #  値に書き下す
        #return [self.EnumPath(node,V)[0] for node in attractors]
    
    
    def GetTransition(self, BN):
        """全状態が次時刻に何の状態になるかを表すBDDを得る"""
        V = list(BN)
        V_next = list(range(V[-1]+1, V[-1]+1+len(V)))   # 次時刻の変数名
        node_transition = self.leaf1
        for v,v_next in zip(V,V_next):
            node_transition &= self.EQ(BN[v], self.GetUnitNode(v_next))
        return node_transition
    
    def Forward(self, node, V, node_transition=None):
        """nodeの表す状態の次の状態を得る"""
        V_next = list(range(V[-1]+1, V[-1]+1+len(V)))   # 次時刻の変数名
        
        # 現在の状態に対して次時刻に何の状態になるかを表すBDDを得る
        node_transition = node_transition & node
        # V_nextの根ノードをすべて集めて結合
        nodes_next = set()
        def scan1(node):
            if node.type is Leaf:
                return
            if node.v in V_next:
                nodes_next.add(node)
                return
            scan1(node.n0); scan1(node.n1)
        scan1(node_transition)
        node_next = self.leaf0
        for n in nodes_next:
            node_next |= n
        # V_nextをVにリネーム
        def scan2(node):
            if node.type is Leaf:
                return node
            v = V[V_next.index(node.v)]
            return self.GetNode(v, scan2(node.n0), scan2(node.n1))
        return scan2(node_next)
    
    def Backward(self, BN, node):
        """nodeの表す状態の前の状態を得る"""
        if node.type is Leaf:
            return node
        node0_R = ~BN[node.v] & self.Backward(BN, node.n0)
        node1_R = BN[node.v] & self.Backward(BN, node.n1)
        return node0_R | node1_R
    
    def GetBasin(self, BN, node):
        """nodeの表す状態のBasinを求める"""
        node_basin = node
        node_frontier = node
        while node_frontier is not self.leaf0:
            node_frontier = self.Backward(BN, node_frontier) & ~node_basin
            node_basin |= node_frontier
        return node_basin

    
    def BN_to_f_dict(self, BN, check = False):
        """
        BN(BDDの集合)からf_dictを求める
        check == Trueのとき、求めたf_dictからBNを作成し、元と変わらないか確認する
        """
        f_dict = {v:self.BDD_to_f(node) for v,node in BN.items()}
        if check:
            BN2 = self.GetBN(f_dict)
            for v in BN:
                assert (BN[v] == BN2[v]), f'BN_to_f_dict: failed v={v}'
        return f_dict
    
    
    def BN_to_parent(self,BN):
        """BNからparentを返す"""
        return {v:self.GetV(node) for v,node in BN.items()}
        
    
    def f_dict_to_parent(self,f_dict):
        """f_dictからparentを返す"""
        return {
            v: list(set(map(abs,itertools.chain.from_iterable(f)))) if type(f) is list else []
            for v,f in f_dict.items()
            }
    
    
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
            for v,node in BN.items():
                if node.type is Leaf:
                    cdict[v] = node.value
            #clistが空のとき、終了
            if cdict == {}:
                return BN
            #clistを全関数に代入
            BN2 = {}
            for v in BN:
                if BN[v].type is Node:
                    BN2[v] = self.AssignConst(BN[v],cdict)
            BN = BN2
    
    
    def BDD_to_f(self, node):
        """
        BDDを見て、1に到達する状態の組み合わせから積和標準形のfを得る
        fは最小とは限らない
        """
        if node.type is Leaf:
            return node.value
        f = []
        states,V = self.EnumPath(node, shorten=True)
        for state in states:
            term = [v * [-1,1][bool(value)] for v,value in zip(V,state) if value != -1]
            f.append(term)
        return f
    
    
    def BN_to_TT(self, BN):
        V_ordered = sorted(BN)
        l = len(V_ordered)
        TT = np.empty((2, 2**l, l), dtype='i1')
        
        # 真理値表の左側
        TT[0] = list(itertools.product([1,0], repeat=l))
        # 真理値表の右側
        for row,state in enumerate(TT[0]):
            cdict = {var:val for var,val in zip(V_ordered,state)}
            TT[1,row] = [self.AssignConstAll(BN[v],cdict) for v in V_ordered]
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
        TT[0] = list(itertools.product([1,0], repeat=l))
        # 真理値表の右側
        for col in range(l):
            TT[1,:,col] = self.calc_on_f(f_dict[V[col]],TT[0],V)
        return TT
    
    
    def calc_f(self, f:[int,list], bdds:dict=dict()) -> Node:
        """BDDに対して関数fの演算を行う"""
        # fが空のとき
        if f == []:
            return None
        # fが定数のとき
        if f == 0 :
            return self.leaf0
        if f == 1 :
            return self.leaf1
        
        def calc_product(term:list):
            # termがもつ変数を全て積算する
            assert len(term)
            node = self.leaf1
            for v in term:
                node2 = bdds.get(abs(v))
                if not node2:
                    # BNにない変数を使おうとする場合、その変数を表すBDDを自動で用意する
                    node2 = self.GetUnitNode(abs(v))
                if v<0:
                    node2 = self.NOT(node2)
                node = self.AND(node, node2)
            return node
        
        #先頭のtermのBDDを生成
        node_sum = calc_product(f[0])
        #termごとにBDDを生成し、sum_BDDに和算する
        for term in f[1:]:
            node_sum = self.OR(node_sum, calc_product(term))
        return node_sum
    
    
    @staticmethod
    def calc_on_f(f, states, V):
        """値に対して関数fの演算を行う"""
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
        for (v1,node1),(v2,node2) in zip(BN1.items(),BN2.items()):
            assert v1 == v2
            assert node1 == node2
        return 0
    
    @staticmethod
    def save_f_dict(f_dict, name = 'f_dict', **kwargs):
        """f_dictをtxtファイルとして出力する"""
        #shutil.rmtree('f_dict.txt')
        with open(f'{name}.txt', mode='w') as f:
            f.write(str(f_dict))
            if kwargs:
                f.write('\n')
                f.write(str(kwargs))

if __name__ == '__main__':
    from time import time
    import drawset, def_f
    from QuineMcCluskey import QM
    
    #while True:
    B = BooleanNetwork()
    
    #f_dict,_ = def_f.def_f('import','f_dict')
    #f_dict,_ = def_f.def_f('import','f_dict_5')
    #f_dict,_ = def_f.def_f('import','f_dict_18')
    #f_dict,_ = def_f.def_f('import','f_dict_30')
    f_dict,_ = def_f.def_f('import','f_dict_muroran')
    #f_dict,_ = def_f.def_f('random','normal',n=4)
    #B.save_f_dict(f_dict, 'f_dict')
    V = list(f_dict)
    BN = B.GetBN(f_dict)
    
    #TT = B.BN_to_TT(BN)
    #drawset.wiring_diagram(B.f_dict_to_parent(f_dict), 'wd')
    #drawset.transition_diagram(B.BN_to_TT(BN),'td')
    #drawset.binary_tree(B.GetBDDdict(B.GetTransition(BN)),'bt',node_name={1:"xA1",2:"xA2",3:"xA3",4:"xB1",5:"xB2",6:"xB3"},format='pdf')
    
    #attractor1 = B.GetFixedAttractor(BN)
    t = time()
    attractors = B.GetAttractor(f_dict, BN)
    drawset.binary_tree(B.GetBDDdict(attractors[0]),'bt',format='pdf')
    attractors2 = [B.EnumPath(node,V)[0] for node in attractors]
    print(f'{time() - t} s')
    
    attractors_basin = [B.GetBasin(BN, node) for node in attractors]
    attractors_basin2 = [B.EnumPath(node,V)[0] for node in attractors_basin]
    print(B.table.__sizeof__())
    
    
    attractors_fixed_simple = B.GetFixedAttractor_simple(BN)
    assert len(attractors_fixed_simple) == sum([1 for attractor in attractors2 if len(attractor) == 1])
        
    

