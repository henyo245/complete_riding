import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import networkx as nx
from pathlib import Path

rsc_dir = Path(__file__).parent.parent / "rsc"

# csvファイルからpandas形式のテーブルを作成
station = pd.read_csv(Path(rsc_dir, "station20251015free.csv"))
join = pd.read_csv(Path(rsc_dir, "join20250916.csv"))
pref = pd.read_csv(Path(rsc_dir, "pref.csv"))
line = pd.read_csv(Path(rsc_dir, "line20250604free.csv"))
company = (Path(rsc_dir, "company20251015.csv"))

"""
全国の路線を考える前に、小規模の問題としてJR北海道の路線を抽出し、作成する
"""
def preprocess_jr_hokkaido(line: pd.DataFrame, station: pd.DataFrame, join: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # JR北海道の路線コードを抽出する JR北海道...company_cd == 1 e_status == 0 運用中のみ
    jrh_lines = line[(line["company_cd"] == 1) & (line["e_status"] == 0)]
    # 海峡線は貨物駅のため削除
    jrh_lines = jrh_lines[jrh_lines["line_cd"]!=11118]

    # # 運用中の駅のみ抽出する e_status == 0 -> 宗谷本線が繋がらなくなるのでコメントアウト
    # station = station[station["e_status"] == 0]
    # 全国の駅からJR北海道の駅のみ抽出する JR北海道...company_cd == 1
    jrh = station[["station_cd", "station_name", "line_cd", "lon", "lat"]]
    jrh = pd.merge(jrh, jrh_lines, on = 'line_cd')
    jrh = jrh[jrh["company_cd"] == 1]
    jrh = jrh[["station_cd", "station_name", "line_cd", "lon_x", "lat_x", "line_name", "line_color_c", "line_color_t"]]
    lon = jrh["lon_x"]
    lat = jrh["lat_x"]
    jrh["lon"] = lon
    jrh["lat"] = lat
    jrh = jrh[["station_cd", "station_name", "line_cd", "lon", "lat", "line_name"]]

    # JR北海道の接続辺を抽出する
    jrh_join = join[join["line_cd"].isin(jrh_lines["line_cd"].tolist())]
    jrh_join = jrh_join[["station_cd1", "station_cd2"]]

    return jrh, jrh_join

# ネットワークグラフの作成
def visualize_graph(stations: pd.DataFrame, join: pd.DataFrame):
    G = nx.Graph()
    # 頂点を駅名にする
    G.add_nodes_from(stations["station_name"])
    # plot の座標を設定
    pos = {row["station_name"]: (row["lon"], row["lat"]) for _, row in stations.iterrows()}
    # 駅コードと駅名の辞書
    cd_to_name = dict(zip(stations["station_cd"], stations["station_name"]))
    # グラフに辺を追加する
    for station_cd1, station_cd2 in join[["station_cd1", "station_cd2"]].itertuples(index=False):
        name1 = cd_to_name.get(station_cd1)
        name2 = cd_to_name.get(station_cd2)
        if name1 and name2:
            G.add_edge(name1, name2)

    # グラフの描画
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(G, pos, with_labels=True, node_size=50, font_size=8, font_family='IPAexGothic')
    plt.title("JR北海道の路線図")
    plt.show()

"""
辺を減らすために，接続駅と終点駅を抽出する
"""
def extract_key_stations(stations: pd.DataFrame, join: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # グラフの作成
    G = nx.Graph()
    G.add_nodes_from(stations["station_cd"])
    for station_cd1, station_cd2 in join[["station_cd1", "station_cd2"]].itertuples(index=False):
        G.add_edge(station_cd1, station_cd2)

    # 接続駅と終点駅の抽出
    key_stations = [node for node, degree in G.degree() if degree not in (0,2)]
    key_stations_df = stations[stations["station_cd"].isin(key_stations)]

    # key_stations に基づいて join をフィルタリング
    key_join = join[
        (join["station_cd1"].isin(key_stations)) |
        (join["station_cd2"].isin(key_stations))
    ]

    return key_stations_df, key_join

def main():
    jrh, jrh_join = preprocess_jr_hokkaido(line, station, join)
    # visualize_graph(jrh, jrh_join)

    key_stations, key_join = extract_key_stations(jrh, jrh_join)
    visualize_graph(key_stations, key_join)

if __name__ == "__main__":
    main()
