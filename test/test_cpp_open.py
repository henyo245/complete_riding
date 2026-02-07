import pytest
from cpp_open import cpp_pipeline_open, visualize_graph_from_adjmatrix


def test_cpp_open_pipeline_start_end():
    inf = float("inf")
    adj_matrix = [
        [0, 3, inf, 7],
        [3, 0, 2, 5],
        [inf, 2, 0, 1],
        [7, 5, 1, 0],
    ]
    visualize_graph_from_adjmatrix(adj_matrix)

    expected_shortest_paths = [
        [0, 3, 5, 6],
        [3, 0, 2, 3],
        [5, 2, 0, 1],
        [6, 3, 1, 0],
    ]

    # 奇数次数頂点 [1,3]. 出発点=1, 到着点=2,
    # 奇数次数集合 = {1,3} XOR {1,2} = {2,3} -> ペア (2,3) コスト 1
    expected_pairs = [(2, 3)]
    expected_total_edge_weight = 19

    shortest_paths, pairs, total_edge_weight = cpp_pipeline_open(
        adj_matrix, start=1, end=2
    )

    assert shortest_paths == expected_shortest_paths
    assert set(pairs) == set(expected_pairs)
    assert total_edge_weight == expected_total_edge_weight

    visualize_graph_from_adjmatrix(
        adj_matrix, selected_pairs=[(2, 3)], start=1, end=2
    )
