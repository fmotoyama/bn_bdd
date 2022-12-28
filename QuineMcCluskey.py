# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 15:06:51 2020

@author: f.motoyama
"""
from time import time
import numpy as np
from copy import copy
import itertools


def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]

#!!!!!!!!!!マイナスをつけて否定を表現するので変数名に「0」があってはならない!!!!!!!!!!
def QM1(f):
    """
    f           :簡単化する対象である、積和標準形を表す2次元list
    term        :fの項を変数名で表現するlist　否定は変数名に「-」がつく
    V           :変数の集合
    l           :変数の個数
    f3          :3値に変換したf
    term3       :f3の項を3値で表現した長さlのndarray 値はVにおける同idxの変数の状態を表す　1:そのまま　-1:否定　0:存在しない
    terms_principal   :主項の集合　term3のlist
    table       :terms_principalを得るための圧縮表　3次元list
    """
    if f in [0,1]:
        return f
    #"""
    for v in itertools.chain.from_iterable(f):
        a = v * [1,-1][v<0]
    #fを平坦化、絶対値化、ユニーク化、整列
    V = sorted(list(set([v * [1,-1][v<0] for v in itertools.chain.from_iterable(f)])))
    l = len(V)          #変数の個数
    
    
    #fを主加法標準展開
    terms_minimum = []
    for i, term in enumerate(f):
        #termを3値((0,1,-1),ternary)に変換
        term_ternary = np.full((1,l), -1, dtype = 'i1')
        for v in term:
            term_ternary[0][V.index(abs(v))] = [0,1][0 < v]
        #term_ternaryの-1を展開
        idxs = np.where(term_ternary[0] == -1)[0]
        for idx in idxs:
            temp = len(term_ternary)  #行数
            term_ternary = np.concatenate([term_ternary,term_ternary])
            term_ternary[:temp,idx] = 0
            term_ternary[temp:,idx] = 1
        terms_minimum.extend(term_ternary)
    
    terms_minimum = np.array(terms_minimum)
    
    #圧縮
    terms_principal = []      #主項
    table = terms_minimum
    for _ in range(l):
        #圧縮回数は最大l-1回だが、l=1の場合を処理するためl回の繰り返し
        table_next = []
        compressed = np.zeros(len(table), dtype = 'bool')   #圧縮できたらTrue
        for i in range(len(table)-1):
            for j in range(i+1,len(table)):
                if np.sum(table[i] != table[j]) == 1:
                    compressed[[i,j]] = True
                    idx_del = np.where(table[i] != table[j])[0]
                    temp = copy(table[i])
                    temp[idx_del] = -1
                    table_next.append(temp)
        
        #圧縮されなかった項を主項に追加
        for i in np.where(~compressed)[0]:
            terms_principal.append(table[i])
        
        if table_next == []:
            break
        table_next = np.unique(table_next, axis=0)  #この処理でtable_nextがnp.arrayとなる 順序は混ざる
        table = copy(table_next)
    terms_principal = np.array(terms_principal)
        
    
    
    #主項の変数と値が一致する最小項を探し、主項図に印をつける
    table_principal = np.zeros((len(terms_principal), len(terms_minimum)), dtype = 'bool')    #主項図
    for row,term_p in enumerate(terms_principal):
        idx = np.where(term_p != -1)[0]
        for col,term_m in enumerate(terms_minimum):
            if np.all(term_p[idx] == term_m[idx]):
                table_principal[row,col] = True
    
    
    #必須項を求める
    idx_m = np.where(np.count_nonzero(table_principal, axis=0) == 1)[0]     #単独の主項がカバーする最小項のidx
    idx_p = list(set(np.where(table_principal[:,idx_m])[0]))                #idx_mの最小項をカバーする主項のidx
    terms_essential = terms_principal[idx_p]
    #使われていない主項とカバーされていない主項図の領域を求める
    terms_principal2 = np.delete(terms_principal, idx_p, axis=0)
    idx_m_noncover = np.where(np.sum(table_principal[idx_p], axis=0) != 1)[0]
    table_principal2 = table_principal[:,idx_m_noncover]        #カバーされていない最小項の列を抽出
    table_principal2 = np.delete(table_principal2, idx_p, 0)    #必須項の行を削除
    
    if len(table_principal2):
        #ぺトリック法 table_principal2の最小項をカバーするterms_principal2の主項を求める
        f_petric_ORAND = [np.where(column)[0].tolist() for column in table_principal2.T]  #ぺトリック方程式（和積系）
        #和積系を積和系に変換
        f_petric_ANDOR = [set()]
        for term_OR in f_petric_ORAND:
            f_petric_ANDOR2 = []
            for v in term_OR:
                f_petric_ANDOR2 += [term_AND | {v} for term_AND in f_petric_ANDOR]
            f_petric_ANDOR = get_unique_list(f_petric_ANDOR2)
        #積和系で最短の項を求める
        temp = len(terms_principal2)
        for term_AND in f_petric_ANDOR:
            if len(term_AND) < temp:
                idx_principal2 = term_AND
                temp = len(term_AND)
        
        idx_principal2 = list(idx_principal2)
        terms_essential = np.concatenate([terms_essential,terms_principal2[idx_principal2]])
    
    f_r = []
    for term_ternary in terms_essential:
        term = []
        for i,ternary in enumerate(term_ternary):
            if ternary == 0:
                term.append(-V[i])
            elif ternary == 1:
                term.append(V[i])
        f_r.append(term)
    return f_r
    
    
    
def QM2(f):
    """
    set()を用いたQM法
    QM1より速い
    """
    if f in [0,1]:
        return f
    
    #fで用いられている変数を抽出
    V = set([v * [1,-1][v<0] for v in itertools.chain.from_iterable(f)])
    l = len(V)          #変数の個数
    
    #fの各項をset型に変換
    f = [set(term) for term in f]
    
    #fを主加法標準展開
    terms_minimum = []
    for term in f:
        v_add = V - {abs(v) for v in term}
        if v_add:
            term_list = [term]
            for v in v_add:
                term_list0 = [term | {-v} for term in term_list]
                term_list1 = [term | {v} for term in term_list]
                term_list = term_list0 + term_list1
            terms_minimum.extend(term_list)
        else:
            terms_minimum.append(term)
    
    #圧縮
    terms_principal = []      #主項
    table = terms_minimum
    for _ in range(1,l+1):
        #圧縮回数は最大l-1回だが、l=1の場合を処理するためl回の繰り返し
        table_next = []
        compressed = np.zeros(len(table), dtype = 'bool')   #圧縮できたらTrue
        for i in range(len(table)-1):
            for j in range(i+1,len(table)):
                symmetric_difference = list(table[i] ^ table[j])
                if len(symmetric_difference) == 2:
                    if abs(symmetric_difference[0]) == abs(symmetric_difference[1]):
                        compressed[[i,j]] = True
                        table_next.append(table[i] - (table[i] - table[j]))
        
        #圧縮されなかった項を主項に追加
        for i in np.where(~compressed)[0]:
            terms_principal.append(table[i])
        
        if table_next == []:
            break
        table_next = get_unique_list(table_next)
        table = copy(table_next)
    
    
    #主項の変数と値が一致する最小項を探し、主項図に印をつける
    table_principal = np.zeros((len(terms_principal), len(terms_minimum)), dtype = 'bool')    #主項図
    for row,term_p in enumerate(terms_principal):
        for col,term_m in enumerate(terms_minimum):
            if term_m >= term_p:    #term_mがterm_pの部分集合のとき
                table_principal[row,col] = True
    
    
    #必須項を求める
    idx_m = np.where(np.count_nonzero(table_principal, axis=0) == 1)[0]     #単独の主項がカバーする最小項のidx
    idx_p = list(set(np.where(table_principal[:,idx_m])[0]))                #idx_mの最小項をカバーする主項のidx
    terms_essential = [terms_principal[i] for i in idx_p]
    #使われていない主項とカバーされていない主項図の領域を求める
    terms_principal2 = [terms_principal[i] for i in range(len(terms_principal)) if i not in idx_p]
    idx_m_noncover = np.where(np.sum(table_principal[idx_p], axis=0) != 1)[0]
    table_principal2 = table_principal[:,idx_m_noncover]        #カバーされていない最小項の列を抽出
    table_principal2 = np.delete(table_principal2, idx_p, 0)    #必須項の行を削除
    
    if len(table_principal2):
        #ぺトリック法 table_principal2の最小項をカバーするterms_principal2の主項を求める
        f_petric_ORAND = [np.where(column)[0].tolist() for column in table_principal2.T]  #ぺトリック方程式（和積系）
        #和積系を積和系に変換
        f_petric_ANDOR = [set()]
        for term_OR in f_petric_ORAND:
            f_petric_ANDOR2 = []
            for v in term_OR:
                f_petric_ANDOR2 += [term_AND | {v} for term_AND in f_petric_ANDOR]
            f_petric_ANDOR = get_unique_list(f_petric_ANDOR2)
        #積和系で最短の項を求める
        temp = len(terms_principal2)
        for term_AND in f_petric_ANDOR:
            if len(term_AND) < temp:
                idx_principal2 = term_AND
                temp = len(term_AND)
        
        terms_essential += [terms_principal2[i] for i in idx_principal2]
    
    terms_essential = [list(term) for term in terms_essential]
    return terms_essential


def QM_test(N = 10):
    from BooleanNetwork import BooleanNetwork
    from def_f import def_f
    
    for i in range(N):
        B = BooleanNetwork()
        f_dict,_ = def_f('random','normal')
        #f_dict = {1: [[7]], 2: [[1, -3, 10]], 3: [[-1, 2]], 4: [[-1, -2, -4, -5, 7], [-1, -2, -4, 5], [-1, -2, 4, -5, -6, -7], [-1, -2, 4, -5, -6, 7], [-1, -2, 4, -5, 6, 7], [-1, -2, 4, 5], [-1, 2], [1]], 5: [[-7]], 6: [[5, 10]], 7: [[-8]], 8: [[2, 4, -5]], 10: [[-1, 8], [1]]}
        BN = B.GetBN(f_dict)
        
        #BDDからf_dict2を生成
        f_dict2 = B.BN_to_f_dict(BN)
        try:
            t1 = time()
            f_dict21 = {v:QM1(f) for v,f in f_dict2.items()}
            t2 = time()
            f_dict22 = {v:QM2(f) for v,f in f_dict2.items()}
            t3 = time()
        except Exception as e:
            print(e)
            return(f_dict2)
        
        #f_dict2からBDDを生成
        BN21 = B.GetBN(f_dict21)
        BN22 = B.GetBN(f_dict22)
        #f_dict,f_dict21,f_dict22の一致判定
        for key in BN:
            if BN[key] != BN21[key]:
                print('QM1 failed')
                return((f_dict[key], f_dict21[key]))
        for key in BN:
            if BN[key] != BN22[key]:
                print('QM2 failed')
                return((f_dict[key], f_dict22[key]))
        
        print(f'QM1:{t2-t1}, QM2:{t3-t2}')
        return(0)



if __name__ == '__main__':
    
    f = [[-1,-2,3,4],[2,3,4],[1,2,-3],[1,-2,3,4]]
    #f = [[-1,2,-3,-4],[1,-2,-3,-4],[1,-2,3,-4],[1,-2,3,4],[1,2,-3,-4],[1,2,3,4]]
    #f = [[7]]
    f21 = QM1(f)
    f22 = QM2(f)
    
    e = QM_test()

