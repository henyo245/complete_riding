import pytest
from cpp import CPP
from cpp import (
    visualize_graph_from_adjmatrix,
)


def test_count_vertices_degree():
    adj_matrix = [
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [0, 1, 0, 1], 
        [1, 0, 1, 0]
    ]
    expected_degrees = [2, 2, 2, 2]
    cpp = CPP()
    assert cpp.count_vertices_degree(adj_matrix) == expected_degrees

    adj_matrix = [
        [0, 1, 1], 
        [1, 0, 0], 
        [1, 0, 0]
    ]
    expected_degrees = [2, 1, 1]
    cpp = CPP()
    assert cpp.count_vertices_degree(adj_matrix) == expected_degrees


def test_get_odd_degree_vertices():
    degree_count = [2, 3, 4, 1, 0]
    expected_odd_vertices = [1, 3]
    cpp = CPP()
    assert cpp.get_odd_degree_vertices(degree_count) == expected_odd_vertices


def test_compute_minimum_weight_perfect_matching_bruteforce():
    adj_matrix = [
        [0, 10, 15, 20], 
        [10, 0, 35, 25], 
        [15, 35, 0, 30], 
        [20, 25, 30, 0]
    ]
    odd_vertices = [0, 1, 2, 3]
    expected_pairs = [(0, 1), (2, 3)]
    expected_cost = 10 + 30

    cpp = CPP()
    result_pairs, best_cost = cpp.compute_minimum_weight_perfect_matching_bruteforce(
        adj_matrix, odd_vertices
    )
    assert set(result_pairs) == set(expected_pairs)
    assert best_cost == expected_cost


def test_returns_empty_on_no_odd_vertices():
    """
    完全マッチングが存在しない場合は,ペアなし,重み0(全ての頂点がちょうど1本の辺に含まれる場合)
    """
    adj_matrix = [
        [0, 1, 0], 
        [1, 0, 1], 
        [0, 1, 0]
    ]
    odd_vertices = []
    expected_pairs = []
    expected_cost = 0

    cpp = CPP()
    result_pairs, best_cost = cpp.compute_minimum_weight_perfect_matching_bruteforce(
        adj_matrix, odd_vertices
    )
    assert result_pairs == expected_pairs
    assert best_cost == expected_cost


def test_sum_all_edges_undirected():
    adj_matrix = [
        [0, 10, 0, 20],
        [10, 0, 30, 0],
        [0, 30, 0, 40],
        [20, 0, 40, 0],
    ]
    expected_sum = 10 + 20 + 30 + 40

    cpp = CPP()
    assert cpp.sum_all_edges_undirected(adj_matrix) == expected_sum


def test_calculate_shortest_path_matrix():
    """
    隣接行列から全頂点間の最短距離行列を計算
    """
    inf = float("inf")
    adj_matrix = [
        [0, 3, inf, 7],
        [3, 0, 2, inf],
        [inf, 2, 0, 1],
        [7, inf, 1, 0],
    ]
    # visualize_graph_from_adjmatrix(adj_matrix)
    expected_shortest_paths = [
        [0, 3, 5, 6],
        [3, 0, 2, 3],
        [5, 2, 0, 1],
        [6, 3, 1, 0],
    ]
    cpp = CPP()
    result = cpp.calculate_shortest_path_matrix(adj_matrix)
    assert result == expected_shortest_paths

def test_cpp_pipeline():
    """
    cpp.py の主要関数を統合的にテスト
    """
    inf = float("inf")
    adj_matrix = [
        [0, 3, inf, 7],
        [3, 0, 2, 5],
        [inf, 2, 0, 1],
        [7, 5, 1, 0],
    ]
    # visualize_graph_from_adjmatrix(adj_matrix)

    expected_shortest_paths = [
        [0, 3, 5, 6],
        [3, 0, 2, 3],
        [5, 2, 0, 1],
        [6, 3, 1, 0],
    ]
    expected_pairs = [(1, 3)]
    expected_total_edge_weight = 21
    cpp = CPP()

    # 距離行列を入力すると，最短距離，完全マッチングのペア, 全頂点間の最短距離行列を返す pipeline をテスト
    shortest_paths, pairs, total_edge_weight = cpp.cpp_pipeline(adj_matrix)
    assert shortest_paths == expected_shortest_paths
    assert set(pairs) == set(expected_pairs)
    assert total_edge_weight == expected_total_edge_weight
