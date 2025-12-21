"""
路線の距離行列を読み込み，完乗に必要な最短距離を計算する
"""
from cpp import CPP
from visualize import Visualizer
import pandas as pd
from pathlib import Path
import numpy as np
import argparse


data_dir = "output"


def main(prefix: str = "company_1_key_stations"):
    # adjmatrix をヘッダ付きで読み込む（先頭列は駅コードのインデックス）
    adj_path = Path(data_dir, f"{prefix}_adjmatrix.csv")
    stations_path = Path(data_dir, f"{prefix}_stations.csv")

    adj_df = pd.read_csv(adj_path, index_col=0)
    # 'inf' 文字列があれば np.inf に置換し、数値化
    adj_df = adj_df.replace("inf", np.inf).astype(float)

    # station codes はインデックスのリスト
    station_codes = adj_df.index.tolist()

    # 隣接行列をリスト形式に変換して CPP に渡す
    adj_matrix_list = adj_df.values.tolist()
    adj_matrix_np = adj_df.values.astype(float)

    cpp = CPP()
    shortest_path_matrix, pairs, total_edge_weight = cpp.cpp_pipeline(adj_matrix_list)

    # 駅情報（駅名, lon, lat を含む想定）を読み込む
    station_info = pd.read_csv(stations_path)
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

    # visualize using Visualizer (station_info contains lon/lat)
    image_path = Path("output") / "images" / f"{prefix}_line_cpp_selected_pairs.png"
    Visualizer().visualize_graph_with_selected_pairs(
        station_info,
        None,
        distance_matrix=adj_matrix_np,
        selected_pairs=selected_pairs_codes,
        save_path=str(image_path),
    )
    print(f"Saved visualization to: {image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CPP on reduced adjacency matrix CSVs produced by generate_map.py")
    parser.add_argument("--prefix", type=str, default="company_1_key_stations", help="prefix for files in output/, e.g. company_1_key_stations")
    args = parser.parse_args()
    main(prefix=args.prefix)