import random
import networkx as nx
import matplotlib.pyplot as plt
import math
import itertools

def create_graph_matrix(v_num, e_num):
    INF = math.inf
    graph = [[0 for _ in range(v_num)] for _ in range(v_num)]

    # まず必ず各頂点に1本のエッジを作る
    # 例: 0-1, 1-2, 2-3, ... のように連鎖
    edges_added = 0
    for i in range(1, v_num):
        u = i - 1
        v = i
        w = random.randint(1, 100)
        graph[u][v] = graph[v][u] = w
        edges_added += 1

    # v_num - 1 本は確実に追加された
    if e_num < v_num - 1:
        raise ValueError("孤立頂点をなくすには e_num >= v_num - 1 が必要です")

    # 残りのエッジをランダム追加
    while edges_added < e_num:
        u = random.randint(0, v_num - 1)
        v = random.randint(0, v_num - 1)

        if u != v and graph[u][v] == 0:
            weight = random.randint(1, 100)
            graph[u][v] = graph[v][u] = weight
            edges_added += 1

    # 0 の箇所を INF に変換（自己ループ除く）
    for i in range(v_num):
        for j in range(v_num):
            if i != j and graph[i][j] == 0:
                graph[i][j] = INF

    return graph


def visualize_graph_from_adjmatrix(adj_matrix, seed=0):
    """
    adj_matrix: 隣接行列 (2次元リスト)
    """
    n = len(adj_matrix)
    G = nx.Graph()

    # ノード追加
    G.add_nodes_from(range(n))

    INF = math.inf
    # 隣接行列から辺を追加
    for i in range(n):
        for j in range(i + 1, n):  # 無向グラフなので片側だけ見る
            w = adj_matrix[i][j]
            if (w != 0) and (w!=INF):  # 0 は辺なし
                G.add_edge(i, j, weight=w)

    # レイアウト
    pos = nx.spring_layout(G, seed=seed)

    # ノード描画
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue")

    # エッジ描画
    nx.draw_networkx_edges(G, pos, width=2)

    # ノードラベル
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    # 重みラベル
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.axis('off')
    plt.show()

def count_vertices_degree(adj_matrix):
    """
    adj_matrix: 隣接行列 (2次元リスト)
    戻り値: 各頂点の次数を格納したリスト
    """
    v_num = len(adj_matrix)
    degree_counts = [0]*v_num

    for i in range(v_num):
        degree = 0
        for j in range(v_num):
            if adj_matrix[i][j] != 0 and adj_matrix[i][j] != math.inf:
                degree += 1
        degree_counts[i] = degree
    return degree_counts

# 完全マッチングの最小経路を求める
def compute_minimum_weight_perfect_matching(adj_matrix: list, odd_vertices):
    num_odd = len(odd_vertices)
    
    # 各ペアリングについて重み合計を計算
    weight = [[0] * num_odd for _ in range(num_odd)]
    for i in range(num_odd):
        for j in range(i + 1, num_odd):
            u = odd_vertices[i]
            v = odd_vertices[j]
            weight[i][j] = adj_matrix[u][v]
            weight[j][i] = adj_matrix[u][v]
    
    # 重みが最小となる完全マッチングを計算
    # 最小コストの記録
    best_cost = float('inf')
    best_pairs = None

    first = 0
    others = list(range(1, num_odd))  # index のリストとして扱う

    # 全ての並べ替えを試す
    for perm in itertools.permutations(others):
        # ペアリングの作成
        pairs = [(first, perm[0])]
        for i in range(1, num_odd // 2):
            pairs.append((perm[2 * i - 1], perm[2 * i]))

        # コスト計算
        cost = sum(weight[u][v] for u, v in pairs)

        # 最小コストの更新
        if cost < best_cost:
            best_cost = cost
            best_pairs = pairs
        
    # index → 実際の odd_vertices の番号へ変換して返す
    result_pairs = [(odd_vertices[i], odd_vertices[j]) for i, j in best_pairs]

    return result_pairs, best_cost   

def sum_all_edges_undirected(adj_matrix):
    """
    無向グラフの全エッジの重み合計を計算
    adj_matrix: 隣接行列 (2次元リスト)
    """
    n = len(adj_matrix)
    total = 0
    for i in range(n):
        for j in range(i + 1, n):   # 上三角部分だけ見る
            if adj_matrix[i][j] != 0 and adj_matrix[i][j] != float('inf'):
                total += adj_matrix[i][j]
    return total

def main():
    # 無向グラフの作成
    v_num = random.randint(2, 5)
    e_num = random.randint(1, v_num * (v_num - 1) // 2)
    graph_matrix = create_graph_matrix(v_num, e_num)

    print(f"頂点数: {v_num}, エッジ数: {e_num}")
    graph_matrix = create_graph_matrix(v_num, e_num)
    for row in graph_matrix:
        print(' '.join(f"{x:>3}" for x in row))

    # 次数が奇数の頂点をリスト化
    degree_count = count_vertices_degree(graph_matrix)
    odd_vertices = [i for i, deg in enumerate(degree_count) if deg % 2 == 1]
    print("次数が奇数の頂点:", odd_vertices)

    result_pairs, best_cost = compute_minimum_weight_perfect_matching(graph_matrix, odd_vertices)
    print("最小完全マッチングのペア:", result_pairs)
    print("最小完全マッチングの重み合計:", best_cost)

    # 最小距離和を計算
    total_edge_weight = sum_all_edges_undirected(graph_matrix) + best_cost
    print("全エッジの重み合計 + 最小完全マッチングの重み合計 =", total_edge_weight)

    visualize_graph_from_adjmatrix(graph_matrix)

if __name__ == "__main__":
    main()