# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:25:53 2020

@author: motoyama
"""
import itertools
import numpy as np
from collections import defaultdict
from typing import NamedTuple, Union

rng = np.random.default_rng()


class Node(NamedTuple):
    neg: bool       # 自身が否定されているときTrue
    hash: int
    # 以下をハッシュ値に変換
    v: int          # 自身の変数名
    n0_real: Union['Node','Leaf']   # n0_real.negは常にFalse
    n1_real: Union['Node','Leaf']
    @property
    def n0(self):
        return self.n0_real.inverse if self.neg else self.n0_real
    @property
    def n1(self):
        return self.n1_real.inverse if self.neg else self.n1_real
    @property
    def inverse(self):
        return Node(not self.neg, self.hash, self.v, self.n0_real, self.n1_real)

class Leaf(NamedTuple):
    neg: bool
    hash: int
    # 以下をハッシュ値に変換
    value_real: int = 0
    @property
    def value(self):
        return int(self.neg)
    @property
    def inverse(self):
        return Leaf(not self.neg, self.hash, self.value_real)

"""
class State:
    def __init__(self, state, V):
        self.state = state
        self.V = V
        assert self.state.ndim == 2
        assert self.state.shape[1] == len(self.V)
"""


class BDD:
    """
    n,node  : Nodeオブジェクト
    h,hash  : 各ノードに割り当てられたハッシュ
    ノードハッシュ元データ: (否定枝,v,0枝ノード,1枝ノード)
    演算ハッシュ元データ:
        orderd: (ノードA,ノードB)
        disordered: frozenset({ノードA,ノードB})
    否定枝ルール:
        ・葉ノードは0のみ
        ・1枝のみ否定枝になり得る
    """
    def __init__(self):
        self.table = dict()                 # 否定フラグを除いたノード内容を入れる
        self.table_AP = defaultdict(dict)   # table_AP[op][(nA,nB)]
        self.leaf0 = Leaf(False, hash((0,)))
        self.leaf1 = self.leaf0.inverse
        self.table[self.leaf0.hash] = self.leaf0
        
        # &
        def __and__(self_,other):
            return self.AND(self_,other)
        # |
        def __or__(self_,other):
            return self.OR(self_,other)
        # ^
        def __xor__(self_,other):
            return self.XOR(self_,other)
        # ~
        def __invert__(self_):
            return self_.inverse
        Node.__and__ = __and__; Leaf.__and__ = __and__
        Node.__or__ = __or__; Leaf.__or__ = __or__
        Node.__xor__ = __xor__; Leaf.__xor__ = __xor__
        Node.__invert__ = __invert__; Leaf.__invert__ = __invert__

    
    def GetNode(self, v, n0, n1):
        # 削除規則
        if n0 == n1:
            return n0
        # 0枝方向のノードが否定のとき、子ノードを否定し自身を否定する
        if n0.neg is True:
            n0 = n0.inverse
            n1 = n1.inverse
            neg = True
        else:
            neg = False
        # 共有規則
        h = hash((v, n1.neg, n0.hash, n1.hash))
        n_found = self.table.get(h)
        if n_found:
            return Node(neg, *n_found)
        else:
            n = Node(neg, h, v, n0, n1)
            self.table[h] = n[1:]
            return n
    
    
    def GetUnitNode(self, v):
        return self.GetNode(v, self.leaf0, self.leaf1)
    
    
    def OR(self,nA,nB):
        # 掘り進む必要がないケース
        if nA == self.leaf1 or nB == self.leaf1:
            return self.leaf1
        if nA == nB or nB == self.leaf0:
            return nA
        if nA == self.leaf0:
            return nB
        
        # 過去の計算結果を用いる
        h = hash(frozenset({(nA.neg, nA.hash), (nB.neg, nB.hash)})) # 順序を気にしない
        result_found = self.table_AP['OR'].get(h)
        if result_found:
            return result_found
        # ド・モルガンの法則
        h_temp = hash(frozenset({(not nA.neg, nA.hash), (not nB.neg, nB.hash)}))
        result_found = self.table_AP['AND'].get(h_temp)
        if result_found:
            return ~result_found
        
        # 再帰的にシャノン展開
        if nA.v < nB.v:
            r = self.GetNode(nA.v,self.OR(nA.n0,nB),self.OR(nA.n1,nB))
        elif nA.v > nB.v:
            r = self.GetNode(nB.v,self.OR(nA,nB.n0),self.OR(nA,nB.n1))
        elif nA.v == nB.v:
            r = self.GetNode(nA.v,self.OR(nA.n0,nB.n0),self.OR(nA.n1,nB.n1))
        
        self.table_AP['OR'][h] = r
        return r
        
        
    def AND(self,nA,nB):
        # 掘り進む必要がないケース
        if nA == self.leaf0 or nB == self.leaf0:
            return self.leaf0
        if nA == nB or nB == self.leaf1:
            return nA
        if nA == self.leaf1:
            return nB
        
        # 過去の計算結果を用いる
        h = hash(frozenset({(nA.neg, nA.hash), (nB.neg, nB.hash)})) # 順序を気にしない
        result_found = self.table_AP['AND'].get(h)
        if result_found:
            return result_found
        # ド・モルガンの法則
        h_temp = hash(frozenset({(not nA.neg, nA.hash), (not nB.neg, nB.hash)}))
        result_found = self.table_AP['OR'].get(h_temp)
        if result_found:
            return ~result_found
        
        # 再帰的にシャノン展開
        if nA.v < nB.v:
            r = self.GetNode(nA.v,self.AND(nA.n0,nB),self.AND(nA.n1,nB))
        elif nA.v > nB.v:
            r = self.GetNode(nB.v,self.AND(nA,nB.n0),self.AND(nA,nB.n1))
        elif nA.v == nB.v:
            r = self.GetNode(nA.v,self.AND(nA.n0,nB.n0),self.AND(nA.n1,nB.n1))
        
        self.table_AP['AND'][h] = r
        return r
    
        
    def XOR(self,nA,nB):
        # 掘り進む必要がないケース
        if nA == nB:
            return self.leaf0
        if nA == self.leaf0:
            return nB
        if nA == self.leaf1:
            return nB.inverse
        if nB == self.leaf0:
            return nA
        if nB == self.leaf1:
            return nA.inverse
        
        # 過去の計算結果を用いる
        h = hash(frozenset({(nA.neg, nA.hash), (nB.neg, nB.hash)})) # 順序を気にしない
        result_found = self.table_AP['XOR'].get(h)
        if result_found:
            return result_found
        
        # 再帰的にシャノン展開
        if nA.v < nB.v:
            r = self.GetNode(nA.v,self.XOR(nA.n0,nB),self.XOR(nA.n1,nB))
        elif nA.v > nB.v:
            r = self.GetNode(nB.v,self.XOR(nA,nB.n0),self.XOR(nA,nB.n1))
        elif nA.v == nB.v:
            r = self.GetNode(nA.v,self.XOR(nA.n0,nB.n0),self.XOR(nA.n1,nB.n1))
        
        self.table_AP['XOR'][h] = r
        return r
    
        
    def EQ(self,nA,nB):
        # 掘り進む必要がないケース
        if nA == nB:
            return self.leaf1
        if nA == self.leaf1:
            return nB
        if nA == self.leaf0:
            return nB.inverse
        if nB == self.leaf1:
            return nA
        if nB == self.leaf0:
            return nA.inverse
        
        # 過去の計算結果を用いる
        h = hash(frozenset({(nA.neg, nA.hash), (nB.neg, nB.hash)}))  # 順序を気にしない
        result_found = self.table_AP['EQ'].get(h)
        if result_found:
            return result_found
        
        # 再帰的にシャノン展開
        if nA.v < nB.v:
            r = self.GetNode(nA.v,self.EQ(nA.n0,nB),self.EQ(nA.n1,nB))
        elif nA.v > nB.v:
            r = self.GetNode(nB.v,self.EQ(nA,nB.n0),self.EQ(nA,nB.n1))
        elif nA.v == nB.v:
            r = self.GetNode(nA.v,self.EQ(nA.n0,nB.n0),self.EQ(nA.n1,nB.n1))
        
        self.table_AP['EQ'][h] = r
        return r
    
    @staticmethod
    def NOT(n):
        return n.inverse
    
    
    ####################
    @staticmethod
    def GetBDDdict(node):
        # bddをdict型で返す 葉ノードは省略 ハッシュ値はtableと対応しない
        bdd_dict = dict()   # hash: [v,hash0,hash1]
        def scan(node):
            if isinstance(node,Leaf):
                return node.value
            data = (node.v, scan(node.n0), scan(node.n1))
            n_hash = hash(data)
            bdd_dict[n_hash] = list(data)
            return n_hash
        scan(node)
        return bdd_dict
    
    
    @staticmethod
    def GetBDDdict_neg(node):
        # 否定枝のあるBDDを返す　根ノードの否定は表現されない
        bdd_dict = dict()   # hash: [v,n1_neg,hash0,hash1]
        def scan(node):
            if isinstance(node,Leaf):
                return node.value_real
            bdd_dict[node.hash] = [node.v, node.n1_real.neg, scan(node.n0_real), scan(node.n1_real)]
            return node.hash
        scan(node)
        return bdd_dict
    
    
    def AssignConst(self,node,cdict):  # cdict[v] = True/False
        """各変数に定数を代入した結果のbddを返す"""
        if isinstance(node,Leaf):
            return node
        const = cdict.get(node.v)
        if const is None:
            r = self.GetNode(
                node.v,
                self.AssignConst(node.n0,cdict),
                self.AssignConst(node.n1,cdict)
                )
        else:
            r = self.AssignConst(node.n1,cdict) if const else self.AssignConst(node.n0,cdict)
        return r
    
    
    @staticmethod
    def AssignConstAll(node,cdict):
        """BDDのすべての変数の値が与えられているとき、BDDの出力を求める"""
        if isinstance(node,Leaf):
            return node.value
        while isinstance(node,Node):
            node = node.n1 if cdict[node.v] else node.n0
        return node.value


    @staticmethod
    def GetV(node):
        """bddで使われている変数を列挙する"""
        V = set()
        def scan(node):
            if isinstance(node,Leaf):
                return
            V.add(node.v)
            scan(node.n0)
            scan(node.n1)
        scan(node)
        return V


    def EnumPath(self, node, V=None, shorten=False):
        """
        1へ到達するパスを全て求める
        Vが与えられていればBDDに無い変数の01の分岐を含む　与えられてなければBDD自体のパスを返す
        """
        if V is None:
            V = sorted(self.GetV(node))
        l = len(V)
        
        if node == self.leaf0:
            return np.empty((0, l), dtype = 'i1'), V
        if node == self.leaf1:
            return np.array(list(itertools.product([1,0], repeat=l)), dtype='i1'), V
        
        def scan(node):
            # nodeから1へ到達するための入力の組み合わせを求める
            id_v = V.index(node.v)
            states = []
            for branch,node_c in enumerate([node.n0,node.n1]):
                if node_c == self.leaf1:
                    states_sub = np.full((1, l), -1, dtype = 'i1')
                    states_sub[:,id_v] = branch
                    states.append(states_sub)
                elif isinstance(node_c,Node):
                    states_sub = scan(node_c)
                    states_sub[:,id_v] = branch
                    states.append(states_sub)
            return np.concatenate(states)
        
        states = scan(node)
        
        if not shorten:
            # -1の部分を書き下す
            states2 = []
            for state in states:
                cols = np.where(state==-1)[0]
                states2_sub = np.tile(state, (2**len(cols),1))   # stateを縦に並べる
                states2_sub[:,cols] = list(itertools.product([1,0], repeat=len(cols)))
                states2.append(states2_sub)
            states2 = np.concatenate(states2)
            states = states2
        
        return states,V
    
    
    def CountPath(self, node, V=None):
        """
        1へ到達するパスの本数を数える
        Vが与えられていればBDDに無い変数の01の分岐を含む　与えられてなければBDD自体のパス数を返す
        """
        if V == None:
            def scan(node):
                if isinstance(node,Leaf):
                    return node.value
                return scan(node.n0) + scan(node.n1)
            return scan(node)
            
        else:
            l = len(V)
            def scan(node):
                if isinstance(node,Leaf):
                    return node.value, l
                # 子ノードとの間で消えているノード数に応じてcountを増やす
                id_v = V.index(node.v)
                count0, id_0 = scan(node.n0)
                count0 *= 2 ** (id_0 - id_v - 1)
                count1, id_1 = scan(node.n1)
                count1 *= 2 ** (id_1 - id_v - 1)
                return count0+count1, id_v
            
            count, id_ = scan(node)
            count *= 2 ** (id_)
            return count
    
    def PickPath(self,node,V=None):
        """ランダムにパスを1本示す"""
        if isinstance(node,Leaf):
            if node == self.leaf1 and V is not None:
                return {v:rng.integers(2) for v in V}
            return None
        path = dict()
        while node != self.leaf1:
            edge = rng.integers(2)
            node_c = eval(f'node.n{edge}')
            if node_c == self.leaf0:
                edge = 1 - edge
                node_c = eval(f'node.n{edge}')
            path[node.v] = edge
            node = node_c
        # 登場していない変数に適当な値を割り振る
        if V is not None:
            path.update([(v, rng.integers(2)) for v in set(V)-set(list(path))])
        return path
    
    
    def GetLineBDD(self, cdict):
        """パスが1本のBDDを生成する"""
        node = self.leaf1
        for var, val in cdict.items():
            node &= self.GetUnitNode(var) if val else ~self.GetUnitNode(var)
        return node


    
if __name__ == '__main__':
    from drawset import binary_tree
    B = BDD()
    
    bn = {v:B.GetUnitNode(v) for v in range(1,6)}
    bdd1 = B.OR(B.AND(bn[1],bn[3]),bn[2])
    bdd2 = (bn[1] & bn[3]) | bn[2]
    assert bdd1.hash == bdd2.hash
    
    bdd_dict = B.GetBDDdict(~bdd1)
    bdd_dict_neg = B.GetBDDdict_neg(~bdd1)
    binary_tree(bdd_dict,node_name={2:'uu'})












