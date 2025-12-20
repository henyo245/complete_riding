"""
路線の距離行列を読み込み，完乗に必要な最短距離を計算する
"""
from cpp import CPP
from generate_map import visualize_graph_with_selected_pairs
import pandas as pd
from pathlib import Path
import numpy as np

data_dir = "output"


def main():
    # adjmatrix をヘッダ付きで読み込む（先頭列は駅コードのインデックス）
    adj_df = pd.read_csv(Path(data_dir, "jrh_key_stations_adjmatrix.csv"), index_col=0)
    # 'inf' 文字列があれば np.inf に置換し、数値化
    adj_df = adj_df.replace("inf", np.inf).astype(float)

    # station codes はインデックスのリスト
    station_codes = adj_df.index.tolist()

    # 隣接行列をリスト形式に変換して CPP に渡す
    adj_matrix = adj_df.values.tolist()

    cpp = CPP()
    shortest_path_matrix, pairs, total_edge_weight = cpp.cpp_pipeline(adj_matrix)

    # 駅情報（駅名, lon, lat を含む想定）を読み込む
    station_info = pd.read_csv(Path(data_dir, "jrh_key_stations_stations.csv"))
    station_name_map = dict(zip(station_info["station_cd"], station_info["station_name"]))

    # 結果表示
    print("最小完全マッチングのペア:")
    for u, v in pairs:
        station_u = station_name_map.get(station_codes[u], station_codes[u])
        station_v = station_name_map.get(station_codes[v], station_codes[v])
        distance = shortest_path_matrix[u][v]
        print(f"{station_u} - {station_v}: {distance}")
    print("全エッジの重み合計 + 最小完全マッチングの重み合計 =", total_edge_weight)

    # 可視化: pairs はインデックス参照なので station_codes を使って station_cd タプルを作る
    selected_pairs_codes = [(station_codes[u], station_codes[v]) for u, v in pairs]

    # shortest_path_matrix を numpy 配列に変換して渡す（visualize 側は行列を期待する）
    dist_matrix = np.array(shortest_path_matrix, dtype=float)

    # visualize_graph_with_selected_pairs に渡す stations DataFrame は station_info
    visualize_graph_with_selected_pairs(
        station_info, None, distance_matrix=adj_matrix, selected_pairs=selected_pairs_codes
    )


if __name__ == "__main__":
    main()