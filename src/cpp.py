import random
import networkx as nx
import matplotlib.pyplot as plt
import math
import itertools

class CPP:
    def count_vertices_degree(self, adj_matrix: list[list]) -> list:
        """
        adj_matrix: 隣接行列 (2次元リスト)
        戻り値: 各頂点の次数を格納したリスト
        """
        v_num = len(adj_matrix)
        degree_counts = [0] * v_num

        for i in range(v_num):
            degree = 0
            for j in range(v_num):
                if adj_matrix[i][j] != 0 and adj_matrix[i][j] != math.inf:
                    degree += 1
            degree_counts[i] = degree
        return degree_counts


    def get_odd_degree_vertices(self, degree_count: list[int]) -> list:
        return [i for i, deg in enumerate(degree_count) if deg % 2 == 1]


    def calculate_shortest_path_matrix(self, adj_matrix: list[list]) -> list:
        """
        隣接行列から全頂点間の最短距離行列を計算（Floyd-Warshallアルゴリズム）
        adj_matrix: 隣接行列 (2次元リスト)
        戻り値: 最短距離行列 (2次元リスト)
        """
        v_num = len(adj_matrix)
        # 初期化
        graph = [[adj_matrix[i][j] for j in range(v_num)] for i in range(v_num)]

        # Floyd-Warshall アルゴリズム
        for k in range(v_num):
            for i in range(v_num):
                for j in range(v_num):
                    if graph[i][k] + graph[k][j] < graph[i][j]:
                        graph[i][j] = graph[i][k] + graph[k][j]
        return graph


    # 完全マッチングの最小経路を求める
    def compute_minimum_weight_perfect_matching_bruteforce(self, adj_matrix: list[list], odd_vertices: list[int]):
        """
        Brute-force version: 全ての並べ替えを試して最小化する（小規模向け）。
        返り値は (pairs, cost) で、pairs は元の頂点インデックスのタプルのリスト。
        """
        num_odd = len(odd_vertices)

        # 完全マッチングが存在しない場合の処理
        if num_odd == 0:
            return [], 0

        if num_odd % 2 == 1:
            raise ValueError("完全マッチングが存在しません（頂点が奇数個）")

        # 各ペアリングについて重み合計を計算
        weight = [[0] * num_odd for _ in range(num_odd)]
        for i in range(num_odd):
            for j in range(i + 1, num_odd):
                u = odd_vertices[i]
                v = odd_vertices[j]
                weight[i][j] = adj_matrix[u][v]
                weight[j][i] = adj_matrix[u][v]

        # 最小コスト探索（全探索）
        best_cost = float("inf")
        best_pairs = None

        first = 0
        others = list(range(1, num_odd))

        for perm in itertools.permutations(others):
            pairs = [(first, perm[0])]
            for i in range(1, num_odd // 2):
                pairs.append((perm[2 * i - 1], perm[2 * i]))

            cost = sum(weight[u][v] for u, v in pairs)
            if cost < best_cost:
                best_cost = cost
                best_pairs = pairs

        result_pairs = [(odd_vertices[i], odd_vertices[j]) for i, j in best_pairs]
        return result_pairs, best_cost

    def sum_all_edges_undirected(self, adj_matrix: list[list]) -> int:
        """
        無向グラフの全エッジの重み合計を計算
        adj_matrix: 隣接行列 (2次元リスト)
        """
        n = len(adj_matrix)
        total = 0
        for i in range(n):
            for j in range(i + 1, n):  # 上三角部分だけ見る
                if adj_matrix[i][j] != 0 and adj_matrix[i][j] != float("inf"):
                    total += adj_matrix[i][j]
        return total
    
    def cpp_pipeline(self, adj_matrix: list[list]):
        v_num = len(adj_matrix)
        e_num = sum(
            1
            for i in range(v_num)
            for j in range(i + 1, v_num)
            if adj_matrix[i][j] != 0 and adj_matrix[i][j] != math.inf
        )

        # 最短距離行列を作成
        shortest_path_matrix = self.calculate_shortest_path_matrix(adj_matrix)

        # 次数が奇数の頂点をリスト化
        degree_count = self.count_vertices_degree(adj_matrix)
        odd_vertices = self.get_odd_degree_vertices(degree_count)

        result_pairs, best_cost = self.compute_minimum_weight_perfect_matching_bruteforce(
            shortest_path_matrix, odd_vertices
        )

        # 最小距離和を計算
        total_edge_weight = self.sum_all_edges_undirected(adj_matrix) + best_cost

        return shortest_path_matrix, result_pairs, total_edge_weight        
        

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
            if (w != 0) and (w != INF):  # 0 は辺なし
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

    plt.axis("off")
    plt.show()




def compute_minimum_weight_perfect_matching_fast(adj_matrix: list, odd_vertices):
    """
    高速版: NetworkX の min_weight_matching を使う（大きな頂点数向け）。
    """
    num_odd = len(odd_vertices)
    if num_odd == 0:
        return [], 0
    if num_odd % 2 == 1:
        raise ValueError("完全マッチングが存在しません（頂点が奇数個）")
    if num_odd == 2:
        u, v = odd_vertices
        return [(u, v)], adj_matrix[u][v]

    # グラフを作成（ノードは元の頂点番号）
    G = nx.Graph()
    for v in odd_vertices:
        G.add_node(v)

    # 大きな重みで unreachable を表す
    BIG = 10**12
    for i in range(num_odd):
        for j in range(i + 1, num_odd):
            u = odd_vertices[i]
            v = odd_vertices[j]
            w = adj_matrix[u][v]
            if w == math.inf or (isinstance(w, float) and math.isnan(w)):
                w = BIG
            G.add_edge(u, v, weight=w)

    matching = nx.algorithms.matching.min_weight_matching(
        G, maxcardinality=True, weight="weight"
    )
    result_pairs = [tuple(sorted(edge)) for edge in matching]
    total_cost = 0
    for u, v in result_pairs:
        w = adj_matrix[u][v]
        if w == math.inf or (isinstance(w, float) and math.isnan(w)):
            total_cost += BIG
        else:
            total_cost += w

    return result_pairs, total_cost


def compute_minimum_weight_perfect_matching(
    adj_matrix: list, odd_vertices, method: str = "auto"
):
    """
    統一インターフェース: method を 'auto'/'fast'/'bruteforce' で選択。
    - 'auto' : 頂点数に応じて自動選択（小規模は bruteforce、大規模は fast）
    - 'fast' : NetworkX ベースの高速実装
    - 'bruteforce' : 全探索（既存の実装）
    """
    num_odd = len(odd_vertices)
    if method == "auto":
        # 閾値は経験則。num_odd <= 10 程度なら全探索も可能
        if num_odd <= 10:
            return compute_minimum_weight_perfect_matching_bruteforce(
                adj_matrix, odd_vertices
            )
        else:
            return compute_minimum_weight_perfect_matching_fast(
                adj_matrix, odd_vertices
            )
    elif method == "fast":
        return compute_minimum_weight_perfect_matching_fast(adj_matrix, odd_vertices)
    elif method == "bruteforce":
        return compute_minimum_weight_perfect_matching_bruteforce(
            adj_matrix, odd_vertices
        )
    else:
        raise ValueError(
            "Unknown method for matching: choose 'auto', 'fast', or 'bruteforce'"
        )


def main():
    # 無向グラフの作成
    v_num = random.randint(2, 5)
    e_num = random.randint(v_num - 1, v_num * (v_num - 1) // 2)
    graph_matrix = create_graph_matrix(v_num, e_num)

    print(f"頂点数: {v_num}, エッジ数: {e_num}")
    for row in graph_matrix:
        print(" ".join(f"{x:>3}" for x in row))

    cpp = CPP()

    shortest_paths, pairs, total_edge_weight = cpp.cpp_pipeline(graph_matrix)
    print("最短距離行列:")
    for row in shortest_paths:
        print(" ".join(f"{x:>3}" for x in row))
    print("最小完全マッチングのペア:", pairs)
    print("全エッジの重み合計 + 最小完全マッチングの重み合計 =", total_edge_weight)

    visualize_graph_from_adjmatrix(graph_matrix)

if __name__ == "__main__":
    main()
