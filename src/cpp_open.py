"""既存の CPP ユーティリティを用いた開路用ラッパー。

`cpp_pipeline_open(adj_matrix, start, end, method='auto')` を提供します。
この関数は最短経路行列を計算し，出発/到着が異なる開路に対応するために
奇次数頂点集合を対称差（symmetric difference）で調整し，調整後の集合に対して
最小重み完全マッチングを求め，総コストを返します。
"""
from typing import List, Tuple, Optional
import math

from cpp import CPP
from visualize_colored import VisualizerColored


def cpp_pipeline_open(
    adj_matrix: List[List], start: int, end: int, method: str = "auto"
) -> Tuple[List[List], List[Tuple[int, int]], float]:
    """CPP パイプラインの開路（start != end）バリエーションを計算する。

    引数:
        adj_matrix: 隣接行列（2次元リスト）。自己ループは `0`、未接続は `math.inf` の慣習を使用します（`CPP` と同じ）。
        start: 出発頂点のインデックス（int）
        end: 到着頂点のインデックス（int）
        method: `CPP.compute_minimum_weight_perfect_matching` に渡すマッチング手法

    戻り値:
        (最短経路行列, マッチングペアリスト, 全エッジ重量 + マッチング重みの合計)

    例外:
        ValueError: start/end が与えられていない、または範囲外の場合
    """
    if start is None or end is None:
        raise ValueError("start と end を指定してください（開路処理）")

    cpp = CPP()

    v_num = len(adj_matrix)
    if not (0 <= start < v_num) or not (0 <= end < v_num):
        raise ValueError("start/end index out of range")

    # 全点対最短経路行列を取得
    shortest_paths = cpp.calculate_shortest_path_matrix(adj_matrix)

    # 元の奇次数頂点集合を取得
    degree_count = cpp.count_vertices_degree(adj_matrix)
    odd_vertices = cpp.get_odd_degree_vertices(degree_count)

    # 開路対応のため，奇次数集合と {start,end} の対称差を取る
    # （対称差により，開始と終了の取り扱いが正しく調整される）
    adjusted_set = set(odd_vertices)
    for s in (start, end):
        if s in adjusted_set:
            adjusted_set.remove(s)
        else:
            adjusted_set.add(s)

    adjusted_odd = sorted(adjusted_set)

    # 理論上は偶数個になるはずだが，念のためチェック
    if len(adjusted_odd) % 2 == 1:
        raise RuntimeError("調整後の奇次数頂点数が奇数です：完全マッチングを計算できません")

    # 調整済み奇次数集合に対して最小重み完全マッチングを計算
    pairs, best_cost = cpp.compute_minimum_weight_perfect_matching(
        shortest_paths, adjusted_odd, method=method
    )

    total_edge_weight = cpp.sum_all_edges_undirected(adj_matrix) + best_cost

    return shortest_paths, pairs, total_edge_weight


def visualize_graph_from_adjmatrix(
    adj_matrix,
    seed: int = 0,
    selected_pairs: Optional[List[Tuple]] = None,
    selected_color: str = "red",
    selected_width: int = 3,
    selected_alpha: float = 0.9,
    start_color: str = "lightgreen",
    end_color: str = "royalblue",
    start_size: int = 700,
    end_size: int = 700,
    start: Optional[int] = None,
    end: Optional[int] = None,
    save_path: Optional[str] = None,
):
    """VisualizerColored を使って隣接行列からグラフを描画するユーティリティ。

    引数は `src/cpp.py` の `visualize_graph_from_adjmatrix` に合わせつつ、
    開始ノード／終了ノードの色やサイズ指定を追加しています。
    """
    VisualizerColored().visualize_graph_from_adjmatrix(
        adj_matrix,
        seed=seed,
        selected_pairs=selected_pairs,
        selected_color=selected_color,
        selected_width=selected_width,
        selected_alpha=selected_alpha,
        save_path=save_path,
        start_color=start_color,
        end_color=end_color,
        start_size=start_size,
        end_size=end_size,
        start=start,
        end=end,
    )
