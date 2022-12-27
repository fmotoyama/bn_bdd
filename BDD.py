# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:25:53 2020

@author: motoyama
"""
import numpy as np
from collections import defaultdict
from typing import NamedTuple, Union



class Node(NamedTuple):
    v: int
    n0: Union['Node', 'Leaf']
    n1: Union['Node', 'Leaf']
    neg: bool               # 0枝が否定枝のときTrue
    hash: int

class Leaf(NamedTuple):
    # 終端1を表す葉ノード
    hash = hash((1,))

# 本来の根ノードの上に否定枝をつけるために使用
class Root(NamedTuple):
    node: Union[Node, Leaf] # Nodeオブジェクト
    neg: bool               # nodeに否定枝がつくときTrue



class BDD:
    """
    n,node  : Nodeオブジェクト 同じ意味のオブジェクトは1つのみ
    h,hash  : 各ノードに割り当てられたハッシュ
    r,root  : Rootオブジェクト
    neg     : 否定枝の有無
    sign    : 符号 (= not neg)
    """
    def __init__(self):
        self.table = {}                     # table[node(hash)] = (v,n0,v1)
        self.table_AP = defaultdict(dict)   # table_AP[op][hash] = ({n0,n1},演算結果) 
        self.leaf = Leaf()
        self.table[self.leaf.hash] = self.leaf

    
    def _RegisterNode(self, v, n0, n1, neg, diff=0):
        """
        ノード要素からハッシュ値計算　共有判定　ハッシュ登録　ノード決定
        この関数はGetNode()でのみ使用される
        ノードは否定枝BDDの規則を満たしていることが前提
        """
        h = hash((v, n0.hash, n1.hash, neg))
        if diff:
            h = h ^ diff
        
        # 重複検知 衝突回避
        node_found = self.table.get(h)
        node_new = Node(v,n0,n1,neg,h)
        if node_found:
            if node_new != node_found:
                print(f'_GetNodeHash: new:{node_new}, exist:{node_found} collided')
                node_found = self._RegisterNode(v, n0, n1, diff+1)
            return node_found
        self.table[h] = node_new
        return node_new
    
    
    def _SearchAP(self, op, rA, rB, order:bool, diff=0):
        """
        apply演算のハッシュ値計算　演算結果の検索
        この関数はAP()でのみ使用される
        order : rA,rBの順序が固定か否かのbool
        """
        if order:
            h = hash((rA.node.hash,rA.neg,rB.node.hash,rB.neg))
            # 8byte,正数にする
            h = h & 0xffff_ffff_ffff_ffff
        else:
            h1 = hash((rA.node.hash,rA.neg)) & 0xffff_ffff_ffff_ffff
            h2 = hash((rB.node.hash,rB.neg)) & 0xffff_ffff_ffff_ffff
            h = h1  ^ h2
        
        if diff:
            h = h ^ diff
        
        # 重複検知 衝突回避
        results_found = self.table_AP[op].get(h)
        pattern = (rA.node.hash,rA.neg,rB.node.hash,rB.neg) if order else {(rA.node,rA.neg),(rB.node,rB.neg)}
        if results_found:
            rA_found, rB_found = results_found[0]
            pattern_found = (rA_found.node.hash,rA_found.neg,rB_found.node.hash,rB_found.neg) if order else {(rA_found.node,rA_found.neg),(rB_found.node,rB_found.neg)}
            if pattern != pattern_found:
                print(f'_SearchAP: "{op}" collided')
                return self._SearchAP(op, rA, rB, diff+1)
            return results_found[1], h
        return None, h
    
    
    
    def GetNode(self, v, r0:Root, r1:Root):
        """
        0枝のみ否定枝となりうる　葉0は葉1へ向かう否定枝で表現
        """
        # 削除規則
        if r0.node.hash == r1.node.hash and r0.neg == r1.neg:
            return r0
        # 共有規則
        if r1.neg:
            # r1に否定枝がついているとき、子ノードの符号が反転した解を求め、最後に親ノードの符号を反転する
            node = self._RegisterNode(v, r0.node, r1.node, not r0.neg)
            return Root(node, True)
        else:
            node = self._RegisterNode(v, r0.node, r1.node, r0.neg)
            return Root(node, False)
    
    
    def GetUnitNode(self, v, neg):
        r0 = Root(self.leaf, not neg)
        r1 = Root(self.leaf, neg)
        return self.GetNode(v, r0, r1)
        
    def GetLeafNode(self, neg):
        return Root(self.leaf, neg)
    
    @staticmethod
    def FlipNode(root):
        return Root(root.node, not root.neg)
    
    
    def AP(self,op,rA,rB):
        def OR(rA,rB):
            order = False
            # 掘り進む必要がないケース
            if rA.node.hash == rB.node.hash:
                if rA.neg == rB.neg:
                    return rA                       # (fA,fA), (-fA,-fA)
                else:
                    return Root(self.leaf,False)    # (fA,-fA), (-fA,fA)
            if type(rA.node) is Leaf:
                if rA.neg:
                    return rB                       # (0,*)
                else:
                    return Root(self.leaf,False)    # (1,*)
            if type(rB.node) is Leaf:
                if rB.neg:
                    return rA                       # (*,0)
                else:
                    return Root(self.leaf,False)    # (*,1)
            
            # 過去の計算結果を用いる
            result_found,h = self._SearchAP('OR',rA,rB,order)
            if result_found:
                return result_found
            # ド・モルガンの法則 fA+fB = ~(~fA*~fB)
            rA_fliped = Root(rA.node, not rA.neg)
            rB_fliped = Root(rB.node, not rB.neg)
            result_found,_ = self._SearchAP('AND',rA_fliped,rB_fliped,order)
            if result_found:
                return Root(result_found.node, not result_found.neg)
            
            # 再帰的にシャノン展開
            vA,nA0,nA1 = rA.node[:3]
            vB,nB0,nB1 = rB.node[:3]
            rA0 = Root(nA0, rA.neg ^ rA.node.neg)
            rA1 = Root(nA1, rA.neg)
            rB0 = Root(nB0, rB.neg ^ rB.node.neg)
            rB1 = Root(nB1, rB.neg)
            
            if vA<vB:
                r=self.GetNode(vA,OR(rA0,rB),OR(rA1,rB))
            elif vA>vB:
                r=self.GetNode(vB,OR(rA,rB0),OR(rA,rB1))
            elif vA==vB:
                r=self.GetNode(vA,OR(rA0,rB0),OR(rA1,rB1))
            
            self.table_AP[op][h] = ([rA,rB], Root(r.node,r.neg))
            return r
        
        
        def AND(rA,rB):
            order = False
            # 掘り進む必要がないケース
            if rA.node.hash == rB.node.hash:
                if rA.neg == rB.neg:
                    return rA                       # (fA,fA), (-fA,-fA)
                else:
                    return Root(self.leaf,True)     # (fA,-fA), (-fA,fA)
            if type(rA.node) is Leaf:
                if rA.neg:
                    return Root(self.leaf,True)     # (0,*)
                else:
                    return rB                       # (1,*)
            if type(rB.node) is Leaf:
                if rB.neg:
                    return Root(self.leaf,True)     # (*,0)
                else:
                    return rA                       # (*,1)
            
            # 過去の計算結果を用いる
            result_found,h = self._SearchAP('AND',rA,rB,order)
            if result_found:
                return result_found
            # ド・モルガンの法則
            rA_fliped = Root(rA.node, not rA.neg)
            rB_fliped = Root(rB.node, not rB.neg)
            result_found,_ = self._SearchAP('OR',rA_fliped,rB_fliped,order)
            if result_found:
                return Root(result_found.node, not result_found.neg)
            
            # 再帰的にシャノン展開
            vA,nA0,nA1 = rA.node[:3]
            vB,nB0,nB1 = rB.node[:3]
            rA0 = Root(nA0, not rA.node.neg if rA.neg else rA.node.neg)
            rA1 = Root(nA1, rA.neg)
            rB0 = Root(nB0, not rB.node.neg if rB.neg else rB.node.neg)
            rB1 = Root(nB1, rB.neg)
            
            if vA<vB:
                r=self.GetNode(vA,AND(rA0,rB),AND(rA1,rB))
            elif vA>vB:
                r=self.GetNode(vB,AND(rA,rB0),AND(rA,rB1))
            elif vA==vB:
                r=self.GetNode(vA,AND(rA0,rB0),AND(rA1,rB1))
            
            self.table_AP[op][h] = ([rA,rB], Root(r.node,r.neg))
            return r
        
        
        #----------AP()の処理----------
        if op=='OR': return OR(rA,rB)
        elif op=='AND': return AND(rA,rB)
        
    
            
    ####################
    @staticmethod
    def GetBDD(root):
        def scan(n,sign):
            # ノードnのbddを返す
            if type(n) is Leaf:
                return int(sign)
            v,n0,n1 = n[:3]
            return [v, scan(n0, not sign if n.neg else sign), scan(n1, sign)]
        return scan(root.node, not root.neg)
    
    
    
    def AssignConst(self,root,cdict):  #cdict[v]=True/False
        """各変数に定数を代入した結果のbddを返す"""
        node = root.node
        if type(node) is Leaf:
            return root
        const = cdict.get(node.v)
        if const == None:
            root2 = self.GetNode(
                node.v,
                self.AssignConst(node.n0,cdict),
                self.AssignConst(node.n1,cdict)
                )
            root2.neg = root.neg ^ root2.neg
        elif const == False:
            # 0枝
            root_temp = Root(node.n0, root.neg ^ root.node.neg)
            root2 = self.AssignConst(root_temp,cdict)
        elif const == True:
            # 1枝
            root_temp = Root(node.n1, root.neg)
            root2 = self.AssignConst(root_temp,cdict)
        return root2
    
    
    @staticmethod
    def AssignConstAll(root,cdict):
        """BDDのすべての変数の値が与えられているとき、BDDの出力を求める"""
        while type(root.node) != Leaf:
            const = cdict[root.node.v]
            if const == False:
                neg = root.neg ^ root.node.neg
            else:
                neg = root.neg
            n = root.node.n0 if const==0 else root.node.n1
            root = Root(n, neg)
        return root


    @staticmethod
    def GetV(root):
        """bddで使われている変数を列挙する"""
        V = []
        
        def scan(node):
            if type(node) is Leaf:
                return
            V.append(node.v)
            scan(node.n0)
            scan(node.n1)
        
        scan(root.node)
        return set(V)


    def EnumPath(self, root, shorten=False, check=False):
        """1へ到達するパスを全て求める"""
        assert not (shorten==True and check==True)
        V_ordered = sorted(self.GetV(root))
        l = len(V_ordered)
        
        def scan(node,sign):
            # nodeからsignに到達するための入力の組み合わせを求める
            v,node0,node1 = node[:3]
            id_v = V_ordered.index(v)
            
            # 0枝
            if type(node0) is Leaf:
                # 枝先が終端のとき
                if sign == (not node.neg):
                    state_0 = np.full((1, l), -1, dtype = 'i1')
                    state_0[:,id_v] = 0
                else:
                    state_0 = np.empty((0,l), dtype = 'i1')
            else:
                state_0 = scan(node0, sign==(not node.neg))
                state_0[:,id_v] = 0
            # 1枝
            if type(node1) is Leaf:
                # 枝先が終端のとき
                if sign:
                    state_1 = np.full((1, l), -1, dtype = 'i1')
                    state_1[:,id_v] = 1
                else:
                    state_1 = np.empty((0,l), dtype = 'i1')
            else:
                state_1 = scan(node1, sign)
                state_1[:,id_v] = 1
            
            return np.concatenate([state_0, state_1])
            
        
        states = scan(root.node, not root.neg)
        
        if not shorten or check:
            # -1の部分を書き下す
            num_clos = 0
            for state in states:
                num_clos += 2** np.count_nonzero(state==-1)
            states2 = np.empty((num_clos,l), dtype = 'i1')
            row = 0
            for state in states:
                cols = np.where(state==-1)[0]
                num_clos = 2**len(cols)
                state2 = np.tile(state, (num_clos,1))
                for i,col in enumerate(cols):
                    state2[:,col] = ([0]*2**i + [1]*2**i) * 2**(len(cols)-i-1)
                states2[row:row+num_clos] = state2
                row += num_clos
            states = states2
                
        if check:
            for state in states:
                cdict = {v:value for v,value in zip(V_ordered,state)}
                assert self.AssignConstAll(root,cdict)
        
        return states,V_ordered
    
    
    def CountPath(self, root, V_ordered=None):
        """
        1へ到達するパスの本数を数える
        V_orderedのうちbddに登場していない変数のパターンも数える
        V_ordered=Noneのときは単純にbddのパス数を数える
        """
        if V_ordered == None:
            def scan(node,sign):
                if type(node) is Leaf:
                    return int(sign)
                return scan(node.n0,sign==(not node.neg)) + scan(node.n1,sign)
            
            return scan(root.node, not root.neg)
            
        else:
            l = len(V_ordered)
            def scan(node,sign):
                if type(node) is Leaf:
                    return int(sign), l
                id_v = V_ordered.index(node.v)
                count0, id_0 = scan(node.n0, sign==(not node.neg))
                count0 *= 2 ** (id_0 - id_v - 1)
                count1, id_1 = scan(node.n1, sign)
                count1 *= 2 ** (id_1 - id_v - 1)
                return count0+count1, id_v
            
            count, id_ = scan(root.node, not root.neg)
            count *= 2 ** (id_)
            return count
    
    def PickPath(self,root):
        """パスを1本示す"""
        path = {}
        def scan(node,sign):
            # 終端1に到達するパスを1本見つける
            if type(node) is Leaf:
                return sign
            
            edge = np.random.randint(0,2)
            if edge == 0:
                if scan(node.n0, sign==(not node.neg)):
                    path[node.v] = 0
                    return 1
                else:
                    scan(node.n1, sign)
                    path[node.v] = 1
                    return 1
            elif edge == 1:
                if scan(node.n1, sign):
                    path[node.v] = 1
                    return 1
                else:
                    scan(node.n0, sign==(not node.neg))
                    path[node.v] = 0
                    return 1
        
        scan(root.node, not root.neg)
        return path
            
    
    
    




    
if __name__ == '__main__':
    B = BDD()
    f = [[1,2],[1,-2,3],[-1,2,3]]
    
    #r1 = B.calc_f([[1,2]])
    #r2 = B.calc_f([[1,-2,3]])
    #r3 = B.calc_f([[-1,2,3]])
    #r12 = B.AP('OR',r1,r2)
    
    
    r = B.calc_f(f,dict())
    bdd = B.GetBDD(r)
    f = B.BDD_to_f(r)
    
    a = type(r) == Root












