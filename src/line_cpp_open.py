"""
路線用の開路ラッパー: `cpp_open` をベースに、出発駅/到着駅を駅コードで指定できる CLI を提供します。

使い方例:
    python -m src.line_cpp_open --prefix company_1_key_stations --start_cd 01234 --end_cd 05678

出力: 最短経路行列、マッチングペア、総コストの表示と可視化（`cpp_open.visualize_graph_from_adjmatrix` を呼び出し）。
"""
from typing import List, Tuple, Optional
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

from cpp_open import cpp_pipeline_open
from visualize_colored import VisualizerColored

data_dir = "output"


def find_station_index(station_codes: List, code: str) -> int:
    """station_codes の中から code に一致するインデックスを返す。

    型の違い（文字列/整数）に配慮して複数パターンで検索する。
    """
    # 直接一致
    if code in station_codes:
        return station_codes.index(code)

    # try int
    try:
        icode = int(code)
    except Exception:
        icode = None

    if icode is not None and icode in station_codes:
        return station_codes.index(icode)

    # try str on elements
    str_codes = [str(c) for c in station_codes]
    if code in str_codes:
        return str_codes.index(code)

    raise ValueError(f"station code {code} not found in station list")


def main(prefix: str, start_cd: str, end_cd: str, method: str = "auto"):
    adj_path = Path(data_dir, f"{prefix}_adjmatrix.csv")
    stations_path = Path(data_dir, f"{prefix}_stations.csv")

    adj_df = pd.read_csv(adj_path, index_col=0)
    adj_df = adj_df.replace("inf", np.inf).astype(float)

    station_codes = adj_df.index.tolist()
    adj_matrix_list = adj_df.values.tolist()

    # find indices for given station codes
    start_idx = find_station_index(station_codes, start_cd)
    end_idx = find_station_index(station_codes, end_cd)

    # run cpp_open pipeline (open-route)
    shortest_paths, pairs, total_edge_weight = cpp_pipeline_open(adj_matrix_list, start_idx, end_idx, method=method)

    # load station info for name mapping
    station_info = pd.read_csv(stations_path)
    station_name_map = dict(zip(station_info["station_cd"].astype(str), station_info.get("station_name", station_info["station_cd"])))

    print("最小完全マッチングのペア:")
    for u, v in pairs:
        station_u = station_name_map.get(str(station_codes[u]), station_codes[u])
        station_v = station_name_map.get(str(station_codes[v]), station_codes[v])
        distance = shortest_paths[u][v]
        print(f"{station_u} - {station_v}: {distance}")

    # 可視化: pairs はインデックス参照なので station_codes を使って station_cd タプルを作る
    selected_pairs_codes = [(station_codes[u], station_codes[v]) for u, v in pairs]

    print("全エッジ + マッチング の合計:", total_edge_weight)

    # 可視化: cpp_open の visualize_graph_from_adjmatrix を使う
    image_path = Path("output") / "images" / f"{prefix}_line_cpp_open_selected.png"
    VisualizerColored().visualize_graph_with_selected_pairs(
        stations=station_info,
        join=None,
        distance_matrix=adj_df.values.astype(float),
        selected_pairs=selected_pairs_codes,
        start=start_cd,
        end=end_cd,
        save_path=str(image_path),
    )
    print(f"Saved visualization to: {image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run open-route CPP on adjacency matrix CSVs with station codes for start/end")
    parser.add_argument("--prefix", type=str, default="company_1_key_stations")
    parser.add_argument("--start_cd", type=int, required=True, help="出発駅の駅コード（index にある値と一致する）")
    parser.add_argument("--end_cd", type=int, required=True, help="到着駅の駅コード（index にある値と一致する）")
    parser.add_argument("--method", type=str, default="auto", help="matching method for cpp (auto/fast/bruteforce)")
    args = parser.parse_args()
    main(prefix=args.prefix, start_cd=args.start_cd, end_cd=args.end_cd, method=args.method)
