import pytest
from cpp import (
    count_vertices_degree,
    compute_minimum_weight_perfect_matching_bruteforce,
    get_odd_degree_vertices,
    sum_all_edges_undirected,
)


def test_count_vertices_degree():
    adj_matrix = [
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [0, 1, 0, 1], 
        [1, 0, 1, 0]
    ]
    expected_degrees = [2, 2, 2, 2]
    assert count_vertices_degree(adj_matrix) == expected_degrees

    adj_matrix = [
        [0, 1, 1], 
        [1, 0, 0], 
        [1, 0, 0]
    ]
    expected_degrees = [2, 1, 1]
    assert count_vertices_degree(adj_matrix) == expected_degrees


def test_get_odd_degree_vertices():
    degree_count = [2, 3, 4, 1, 0]
    expected_odd_vertices = [1, 3]
    assert get_odd_degree_vertices(degree_count) == expected_odd_vertices


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

    result_pairs, best_cost = compute_minimum_weight_perfect_matching_bruteforce(
        adj_matrix, odd_vertices
    )
    assert set(result_pairs) == set(expected_pairs)
    assert best_cost == expected_cost


# 完全マッチングが存在しない場合は，ペアなし，重み0
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
    result_pairs, best_cost = compute_minimum_weight_perfect_matching_bruteforce(
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
    assert sum_all_edges_undirected(adj_matrix) == expected_sum
