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
# JR北海道の路線コードを抽出する JR北海道...company_cd == 1
jrh_lines = line[line["company_cd"] == 1]
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

# ネットワークグラフの作成
G = nx.Graph()
# 頂点を駅名にする
G.add_nodes_from(jrh["station_name"])
# plot の座標を設定
pos = {row["station_name"]: (row["lon"], row["lat"]) for _, row in jrh.iterrows()}
# 駅コードと駅名の辞書
cd_to_name = dict(zip(jrh["station_cd"], jrh["station_name"]))
# グラフに辺を追加する
for station_cd1, station_cd2 in jrh_join[["station_cd1", "station_cd2"]].itertuples(index=False):
    name1 = cd_to_name.get(station_cd1)
    name2 = cd_to_name.get(station_cd2)
    if name1 and name2:
        G.add_edge(name1, name2)
        if(name1 == "南稚内" and name2 == "勇知") or (name1 == "勇知" and name2 == "南稚内"):
            print("Found edge between 南稚内 and 勇知")

# グラフの描画
plt.figure(figsize=(10, 10))
nx.draw_networkx(G, pos, with_labels=True, node_size=50, font_size=4, font_family='IPAexGothic')
plt.title("JR北海道の路線図")
plt.show()