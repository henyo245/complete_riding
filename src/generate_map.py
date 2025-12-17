import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import networkx as nx
from pathlib import Path
import math

rsc_dir = Path(__file__).parent.parent / "rsc"
out_dir = Path(__file__).parent.parent / "output"

# csvファイルからpandas形式のテーブルを作成
station = pd.read_csv(Path(rsc_dir, "station20251015free.csv"))
join = pd.read_csv(Path(rsc_dir, "join20250916.csv"))
pref = pd.read_csv(Path(rsc_dir, "pref.csv"))
line = pd.read_csv(Path(rsc_dir, "line20250604free.csv"))
company = Path(rsc_dir, "company20251015.csv")

"""
全国の路線を考える前に、小規模の問題としてJR北海道の路線を抽出し、作成する
"""


def preprocess_jr_hokkaido(
    line: pd.DataFrame, station: pd.DataFrame, join: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # JR北海道の路線コードを抽出する JR北海道...company_cd == 1 e_status == 0 運用中のみ
    jrh_lines = line[(line["company_cd"] == 1) & (line["e_status"] == 0)]
    # 海峡線は貨物駅のため削除
    jrh_lines = jrh_lines[jrh_lines["line_cd"] != 11118]

    # # 運用中の駅のみ抽出する e_status == 0 -> 宗谷本線が繋がらなくなるのでコメントアウト
    # station = station[station["e_status"] == 0]
    # 全国の駅からJR北海道の駅のみ抽出する JR北海道...company_cd == 1
    jrh = station[["station_cd", "station_name", "line_cd", "lon", "lat"]]
    jrh = pd.merge(jrh, jrh_lines, on="line_cd")
    jrh = jrh[jrh["company_cd"] == 1]
    jrh = jrh[
        [
            "station_cd",
            "station_name",
            "line_cd",
            "lon_x",
            "lat_x",
            "line_name",
            "line_color_c",
            "line_color_t",
        ]
    ]
    lon = jrh["lon_x"]
    lat = jrh["lat_x"]
    jrh["lon"] = lon
    jrh["lat"] = lat
    jrh = jrh[["station_cd", "station_name", "line_cd", "lon", "lat", "line_name"]]

    # JR北海道の接続辺を抽出する
    jrh_join = join[join["line_cd"].isin(jrh_lines["line_cd"].tolist())]
    jrh_join = jrh_join[["station_cd1", "station_cd2"]]

    # station_cd を station_g_cd に変換
    station_cd_to_gcd = dict(zip(station["station_cd"], station["station_g_cd"]))
    jrh["station_cd"] = jrh["station_cd"].map(station_cd_to_gcd)
    jrh_join["station_cd1"] = jrh_join["station_cd1"].map(station_cd_to_gcd)
    jrh_join["station_cd2"] = jrh_join["station_cd2"].map(station_cd_to_gcd)

    return jrh, jrh_join


# ネットワークグラフの作成
def visualize_graph(
    stations: pd.DataFrame,
    join: pd.DataFrame,
    distance_matrix: np.ndarray | None = None,
):
    G = nx.Graph()
    # 頂点を駅名にする
    G.add_nodes_from(stations["station_name"])
    # plot の座標を設定
    pos = {
        row["station_name"]: (row["lon"], row["lat"]) for _, row in stations.iterrows()
    }
    # 駅コードと駅名の辞書
    cd_to_name = dict(zip(stations["station_cd"], stations["station_name"]))

    # 距離行列が渡されている場合、stations の並びと行列の並びは一致している想定
    station_list = stations["station_cd"].tolist()
    station_index = {station_cd: idx for idx, station_cd in enumerate(station_list)}

    # グラフに辺を追加する
    if distance_matrix is not None:
        # distance_matrix が渡された場合は、その行列に基づいて全てのキー駅間の
        # 路線距離（最短経路）を辺として追加する（key_stations 用の表示で用いる）
        station_list = stations["station_cd"].tolist()
        for i in range(len(station_list)):
            for j in range(i + 1, len(station_list)):
                cd_i = station_list[i]
                cd_j = station_list[j]
                name1 = cd_to_name.get(cd_i)
                name2 = cd_to_name.get(cd_j)
                if not (name1 and name2):
                    continue
                dist = distance_matrix[i][j]
                if not np.isfinite(dist) or dist == 0:
                    continue
                G.add_edge(name1, name2, weight=dist)
    else:
        # 距離行列が渡されていない場合は join データに基づき辺を追加する
        for station_cd1, station_cd2 in join[["station_cd1", "station_cd2"]].itertuples(
            index=False
        ):
            name1 = cd_to_name.get(station_cd1)
            name2 = cd_to_name.get(station_cd2)
            if name1 and name2:
                G.add_edge(name1, name2)

    # グラフの描画
    plt.figure(figsize=(10, 10))
    # ノードとラベル
    nx.draw_networkx_nodes(G, pos, node_size=50)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="IPAexGothic")
    # エッジ描画
    nx.draw_networkx_edges(G, pos)

    # エッジに距離ラベルを付ける（weight 属性があれば小数点2桁で表示）
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        w = data.get("weight")
        if w is not None:
            edge_labels[(u, v)] = f"{w:.2f}"

    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title("JR北海道の路線図")
    plt.axis("off")
    plt.show()


# 辺に重みとして駅間の距離を持たせるためのデータ作成
def calculate_distance_matrix(stations: pd.DataFrame, join: pd.DataFrame) -> np.ndarray:
    station_list = stations["station_cd"].tolist()
    station_index = {station_cd: idx for idx, station_cd in enumerate(station_list)}
    v_num = len(station_list)
    INF = math.inf

    # 隣接行列の初期化
    adj_matrix = np.full((v_num, v_num), INF)
    np.fill_diagonal(adj_matrix, 0)

    # 駅間距離の計算と隣接行列への反映
    for station_cd1, station_cd2 in join[["station_cd1", "station_cd2"]].itertuples(
        index=False
    ):
        if station_cd1 in station_index and station_cd2 in station_index:
            idx1 = station_index[station_cd1]
            idx2 = station_index[station_cd2]
            coord1 = stations[stations["station_cd"] == station_cd1][
                ["lon", "lat"]
            ].values[0]
            coord2 = stations[stations["station_cd"] == station_cd2][
                ["lon", "lat"]
            ].values[0]
            distance = np.sqrt(
                (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2
            )
            adj_matrix[idx1][idx2] = distance
            adj_matrix[idx2][idx1] = distance  # 無向グラフ

    return adj_matrix


"""
辺を減らすために，接続駅と終点駅を抽出する
"""


def extract_key_stations(
    stations: pd.DataFrame, join: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # グラフの作成
    G = nx.Graph()
    G.add_nodes_from(stations["station_cd"])
    for station_cd1, station_cd2 in join[["station_cd1", "station_cd2"]].itertuples(
        index=False
    ):
        G.add_edge(station_cd1, station_cd2)

    # 接続駅と終点駅の抽出
    key_stations = [node for node, degree in G.degree() if degree not in (0, 2)]
    key_stations_df = stations[stations["station_cd"].isin(key_stations)]

    # key_stations に基づいて join をフィルタリング
    key_join = join[
        (join["station_cd1"].isin(key_stations))
        | (join["station_cd2"].isin(key_stations))
    ]

    return key_stations_df, key_join


def compute_key_station_pairwise_distances(
    all_stations: pd.DataFrame, all_join: pd.DataFrame, key_stations: pd.DataFrame
) -> np.ndarray:
    """
    all_stations: 全駅の DataFrame (must contain 'station_cd','lon','lat')
    all_join: 全駅の接続情報 DataFrame (must contain 'station_cd1','station_cd2')
    key_stations: 抽出されたキー駅の DataFrame (subset of all_stations)

    戻り値: key_stations に対応する順序での距離行列 (numpy.ndarray)
    距離は路線に沿った最短経路距離（辺の重みは駅間のユークリッド距離）を使う。
    """
    # Build full graph with station_cd as nodes and weight as euclidean distance
    G_full = nx.Graph()
    # station_cd が重複している場合があるため、重複を排除して座標辞書を作る
    dedup_stations = all_stations.drop_duplicates(subset="station_cd", keep="first")
    coords = dedup_stations.set_index("station_cd")[["lon", "lat"]].to_dict("index")

    # add nodes
    for station_cd in all_stations["station_cd"].tolist():
        G_full.add_node(station_cd)

    # add edges with weights
    for s1, s2 in all_join[["station_cd1", "station_cd2"]].itertuples(index=False):
        # 座標が見つからない駅がある場合はスキップ
        if s1 in coords and s2 in coords:
            lon1, lat1 = coords[s1]["lon"], coords[s1]["lat"]
            lon2, lat2 = coords[s2]["lon"], coords[s2]["lat"]
            dist = math.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)
            G_full.add_edge(s1, s2, weight=dist)
        else:
            # missing coords -> skip that edge
            continue

    # prepare key station list in same order as key_stations dataframe
    key_list = key_stations["station_cd"].tolist()
    k = len(key_list)
    INF = math.inf
    dist_matrix = np.full((k, k), INF)
    np.fill_diagonal(dist_matrix, 0.0)

    # compute shortest path distances between key stations using networkx Dijkstra
    for i in range(k):
        for j in range(i + 1, k):
            s_i = key_list[i]
            s_j = key_list[j]
            try:
                d = nx.shortest_path_length(G_full, s_i, s_j, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                d = INF
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


def build_reduced_adj_matrix(
    all_stations: pd.DataFrame, all_join: pd.DataFrame, key_stations: pd.DataFrame
) -> np.ndarray:
    """
    key_stations 間の縮約グラフの隣接行列を作る。
    縮約グラフでは、キー駅ペア間の最短経路が他のキー駅を経由しない場合にのみ
    そのペアを直接接続し、重みは経路長（辺の重みは各駅間のユークリッド距離）とする。

    戻り値: (k,k) の numpy 行列（キー駅の順序は key_stations の順序に対応）。経路が無ければ np.inf。
    """
    # 全駅グラフを構築（ノードは station_cd、辺の重みは lon/lat のユークリッド距離）
    G_full = nx.Graph()
    # station_cd が重複している場合があるため、重複を排除して座標辞書を作る
    dedup_stations = all_stations.drop_duplicates(subset="station_cd", keep="first")
    coords = dedup_stations.set_index("station_cd")[["lon", "lat"]].to_dict("index")
    for station_cd in all_stations["station_cd"].tolist():
        G_full.add_node(station_cd)
    for s1, s2 in all_join[["station_cd1", "station_cd2"]].itertuples(index=False):
        if s1 in coords and s2 in coords:
            lon1, lat1 = coords[s1]["lon"], coords[s1]["lat"]
            lon2, lat2 = coords[s2]["lon"], coords[s2]["lat"]
            dist = math.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)
            G_full.add_edge(s1, s2, weight=dist)

    key_list = key_stations["station_cd"].tolist()
    k = len(key_list)
    INF = math.inf
    adj = np.full((k, k), INF)
    np.fill_diagonal(adj, 0.0)

    key_set = set(key_list)

    # 各キー駅ペアについて最短経路を取得し、中間ノードにキー駅が含まれない場合のみ辺を張る
    for i in range(k):
        for j in range(i + 1, k):
            s_i = key_list[i]
            s_j = key_list[j]
            try:
                path = nx.shortest_path(G_full, source=s_i, target=s_j, weight="weight")
                # 中間ノードをチェック
                mid_nodes = path[1:-1]
                contains_key = any(n in key_set for n in mid_nodes)
                if not contains_key:
                    # 距離の合計を計算
                    d = 0.0
                    for a, b in zip(path[:-1], path[1:]):
                        d += G_full[a][b]["weight"]
                    adj[i, j] = d
                    adj[j, i] = d
                else:
                    # 中間に別のキー駅がある -> 縮約グラフでは直接辺を張らない
                    adj[i, j] = INF
                    adj[j, i] = INF
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                adj[i, j] = INF
                adj[j, i] = INF

    return adj


def save_reduced_adjmatrix_csv(
    adj_matrix: np.ndarray,
    key_stations: pd.DataFrame,
    out_dir: Path,
    prefix: str = "key",
):
    """
    adj_matrix を CSV として保存する。行/列ラベルはキー駅の station_cd を利用する。
    またキー駅の一覧（station_cd, station_name）も保存する。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    station_list = key_stations["station_cd"].tolist()
    # pandas DataFrame にして保存（inf は文字列 'inf' になる）
    df = pd.DataFrame(adj_matrix, index=station_list, columns=station_list)
    csv_path = out_dir / f"{prefix}_adjmatrix.csv"
    df.to_csv(csv_path)

    list_path = out_dir / f"{prefix}_stations.csv"
    key_stations[["station_cd", "station_name"]].to_csv(list_path, index=False)

    return csv_path, list_path


def main():
    jrh_stations, jrh_join = preprocess_jr_hokkaido(line, station, join)
    distance_matrix = calculate_distance_matrix(jrh_stations, jrh_join)
    visualize_graph(jrh_stations, jrh_join, distance_matrix)

    # 接続駅と終点駅を抽出
    key_stations, key_join = extract_key_stations(jrh_stations, jrh_join)
    # 接続駅と終点駅の距離を計算
    reduced_adj = build_reduced_adj_matrix(jrh_stations, jrh_join, key_stations)
    visualize_graph(key_stations, key_join, reduced_adj)

    # 路線に沿った距離を計算するため，全駅のグラフを辿ってキー駅間の最短経路距離を求める
    key_distance_matrix = compute_key_station_pairwise_distances(
        jrh_stations, jrh_join, key_stations
    )
    visualize_graph(key_stations, key_join, key_distance_matrix)
    save_reduced_adjmatrix_csv(
        key_distance_matrix, key_stations, out_dir, prefix="jrh_key_stations"
    )


if __name__ == "__main__":
    main()
