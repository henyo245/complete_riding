"""
路線の距離行列を読み込み，完乗に必要な最短距離を計算する
"""
from cpp import CPP
from cpp import (
    visualize_graph_from_adjmatrix,
)
import pandas as pd
from pathlib import Path

data_dir = "output"

def main():
    # ファイルを読み込む
    adj_matrix_with_station = pd.read_csv(
        Path(data_dir, "jrh_key_stations_adjmatrix.csv"),
    ).values.tolist()

    # 上，左は駅コードなので削除
    adj_matrix = [row[1:] for row in adj_matrix_with_station[1:]]
    cpp = CPP()
    shortest_path_matrix, pairs, total_edge_weight = cpp.cpp_pipeline(adj_matrix)
    
    # 駅コードを戻す
    shortest_path_matrix_with_station = []
    station_codes = [row[0] for row in adj_matrix_with_station]
    shortest_path_matrix_with_station.append(station_codes)
    for i, row in enumerate(shortest_path_matrix):
        shortest_path_matrix_with_station.append([station_codes[i]] + row)
    
    # 駅コードを駅名に変換
    station_info = pd.read_csv(Path(data_dir, "jrh_key_stations_stations.csv"))
    station_name_map = dict(zip(station_info["station_cd"], station_info["station_name"]))
    adj_matrix_with_names = []
    for row in shortest_path_matrix_with_station:
        adj_matrix_with_names.append(
            [station_name_map.get(code, code) for code in row]
        )
    
    # 結果表示
    print("最小完全マッチングのペア:")
    for u, v in pairs:
        station_u = station_name_map.get(station_codes[u], station_codes[u])
        station_v = station_name_map.get(station_codes[v], station_codes[v])
        distance = shortest_path_matrix[u][v]
        print(f"{station_u} - {station_v}: {distance}")
    print("全エッジの重み合計 + 最小完全マッチングの重み合計 =", total_edge_weight)


if __name__ == "__main__":
    main()