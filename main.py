import streamlit as st 
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import networkx as nx


def main():
    st.title("巡回セールスマン問題の近似解法と可視化")
    #st.subheader("局所探索法では2-opt近傍に対する局所最適解を求める")
    n = st.number_input(label="都市の数を入力", value=10, max_value=120, min_value=1)
    #selected_solution = st.radio("解法を選択", ["局所探索法(高速)", "GRASP(1分)"])
    seed = st.number_input(label="seed値を入力(ランダムに生成された都市を記憶するため)", value=1, min_value=1)
    np.random.seed(seed)
    C = {}
    
    for i in range(n):
        x = np.random.randint(0, 10000)
        y = np.random.randint(0, 10000)  # 0to10000
        C[i] = (x, y)

    if st.checkbox("近似解を生成"):
        Gt=init_graph(C)
        minimum_tree = nx.minimum_spanning_tree(Gt)
        tour = traverse_tree(minimum_tree, np.random.randint(0, n))
        visualize(tour,C)
    
    if st.checkbox("局所探索法で解を求める(2-opt近傍)"):
        Gt = init_graph(C)
        minimum_tree = nx.minimum_spanning_tree(Gt)
        tour = traverse_tree(minimum_tree, np.random.randint(0, n))
        tour = local_search(tour,C)
        visualize(tour, C)
    
    if st.checkbox("GRASPで解を求める(1分程度)"):
        Gt = init_graph(C)
        start = time.time()
        best_len = 10 ** 9
        ans = []
        
        while time.time() - start < 60:
            tour = NN(C, Gt)
            btour = local_search(tour, C)
            if graph_of_tour(btour,C).size(weight="weight") < best_len:
                best_len = graph_of_tour(btour,C).size(weight="weight")
                ans = btour
        
        visualize(ans,C)



def visualize(tour, C):
    Gt = graph_of_tour(tour,C)
    fig=plt.figure(figsize=(12, 12))
    nx.draw_networkx(Gt, pos=C)
    st.pyplot(fig)
    st.write("総移動距離:{}".format(Gt.size(weight="weight")))


def distance(pa, pb):
    return ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5


def init_graph(C):
    #無向グラフ
    G = nx.Graph()

    for v in C:
        G.add_node(v)  # 点を追加

    #辺を追加
    #完全グラフ(すべての点が他の点とつながっている)
    for i, ci in enumerate(C):
        for j, cj in enumerate(C):
            if i < j:
                G.add_edge(ci, cj, weight=distance(C[ci], C[cj]))
    return G


def traverse_tree(t, v, parent=None, tour=[]):  # tをvからたどる
    for u in t[v]:
        if u != parent:
            tour.append(u)
            traverse_tree(t, u, v, tour)
    return tour


def graph_of_tour(tour,C):
    Gt = nx.Graph()
    for i in range(len(tour)):
        ci = tour[i]
        cj = tour[0] if i == len(tour) - 1 else tour[i + 1]
        Gt.add_edge(ci, cj, weight=distance(C[ci], C[cj]))
    return Gt


def better_solution(sol,C):
    n = len(sol)
    for i in range(n):
        for length in range(2, n):
            if i + length - 1 > n - 1:
                break
            path = sol[i: i + length]
            path.reverse()
            diff = (
                -distance(C[sol[i - 1]], C[sol[i]])
                - distance(C[sol[i + length - 1]], C[sol[(i + length) % n]])
                + distance(C[sol[i - 1]], C[sol[i + length - 1]])
                + distance(C[sol[i]], C[sol[(i + length) % n]])
            )
            if diff < -0.000001:
                bsol = sol[:i] + path + sol[i + length:]
                return bsol

    return None


def local_search(init_sol,C):
    sol = init_sol
    while True:
        bsol = better_solution(sol,C)
        if bsol == None:
            return sol
        else:
            sol = bsol

    return None


def NN(C, G):
    tour = [0]
    visited = {tour[0]: True}
    while len(tour) < len(C):
        v = tour[-1]
        min_d, min_u = 10 ** 8, None
        dic = {}

        for u in G[v]:  # vから近い点20個をランダムに取得
            if min_d > G[v][u]["weight"] and not u in visited:
                min_d = G[v][u]["weight"]
                min_u = u
                dic[u] = min_d
        dic = sorted(dic.items(), key=lambda x: x[1])
        key, val = random.choice(dic[:20])
        visited[key] = True
        tour.append(key)
    return tour


if __name__ == "__main__":
    main()
