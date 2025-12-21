import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import networkx as nx
from pathlib import Path
import math
import argparse

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
    # 保守的に既存の挙動を維持するため、汎用関数を呼び出す
    return preprocess_company(
        line, station, join, company_cd=1, exclude_line_cds=[11118]
    )


def preprocess_company(
    line: pd.DataFrame,
    station: pd.DataFrame,
    join: pd.DataFrame,
    company_cd: int,
    exclude_line_cds: list[int] | None = None,
    only_active_lines: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    指定した `company_cd` に対応する路線・駅・接続情報を抽出する汎用前処理。

    - `exclude_line_cds` が与えられた場合、その路線コードは除外する（例: 貨物線など）。
    - `only_active_lines` が True のときは `e_status == 0` の路線のみを抽出する。
    """
    # company に属する路線を抽出
    if only_active_lines:
        company_lines = line[(line["company_cd"] == company_cd) & (line["e_status"] == 0)]
    else:
        company_lines = line[line["company_cd"] == company_cd]

    if exclude_line_cds:
        company_lines = company_lines[~company_lines["line_cd"].isin(exclude_line_cds)]

    # 駅情報を結合して該当 company の駅を抽出
    subset_cols = ["station_cd", "station_name", "line_cd", "lon", "lat"]
    if not set(subset_cols).issubset(set(station.columns)):
        # station のカラムが期待通りでない場合は最小限で作る
        subset_cols = [c for c in subset_cols if c in station.columns]

    comp_stations = station[subset_cols]
    comp_stations = pd.merge(comp_stations, company_lines, on="line_cd")
    comp_stations = comp_stations[comp_stations["company_cd"] == company_cd]

    # 列名の整備（元の jrh_hokkaido と同じ列名を作る）
    rename_cols = {}
    if "lon_x" in comp_stations.columns:
        comp_stations["lon"] = comp_stations["lon_x"]
    if "lat_x" in comp_stations.columns:
        comp_stations["lat"] = comp_stations["lat_x"]

    keep_cols = [c for c in ["station_cd", "station_name", "line_cd", "lon", "lat", "line_name"] if c in comp_stations.columns]
    comp_stations = comp_stations[keep_cols]

    # station_cd を station_g_cd に変換（グローバル station 引数と一致する前提）
    if "station_g_cd" in station.columns:
        station_cd_to_gcd = dict(zip(station["station_cd"], station["station_g_cd"]))
        comp_stations["station_cd"] = comp_stations["station_cd"].map(station_cd_to_gcd)
        comp_stations = comp_stations.drop_duplicates(subset="station_cd", keep="first")

    # 接続データの抽出
    comp_join = join[join["line_cd"].isin(company_lines["line_cd"].tolist())]
    # 接続の station_cd を group id に変換しておく
    if "station_g_cd" in station.columns:
        comp_join["station_cd1"] = comp_join["station_cd1"].map(station_cd_to_gcd)
        comp_join["station_cd2"] = comp_join["station_cd2"].map(station_cd_to_gcd)

    # 最低限 station_cd 列のみを残す
    comp_join = comp_join[["station_cd1", "station_cd2"]]

    return comp_stations, comp_join


# ネットワークグラフの作成
def visualize_graph(
    stations: pd.DataFrame,
    join: pd.DataFrame,
    distance_matrix: np.ndarray | None = None,
):
    # delegate to Visualizer
    from visualize import Visualizer

    Visualizer().visualize_graph_of_stations(stations, join, distance_matrix)


def visualize_graph_with_selected_pairs(
    stations: pd.DataFrame,
    join: pd.DataFrame,
    distance_matrix: np.ndarray | None = None,
    selected_pairs=None,
    selected_color: str = "red",
    selected_width: int = 2,
    selected_alpha: float = 0.9,
    seed: int = 0,
):
    """
    `visualize_graph` の別バージョン。
    - `selected_pairs` は `(a, b)` のリストで与える（`station_cd` または `station_name` のどちらでも可）。
    - もし `a` と `b` が直接辺でつながっていなければ、グラフ上の最短経路上の辺をハイライトする。
    - 最短経路が存在しない場合は、ノード座標を破線で結ぶ。
    """
    G = nx.Graph()
    # 頂点を駅名にする
    # stations に lon/lat が含まれる場合はそれを座標に使う。含まれない場合は後で spring_layout で作る。
    G.add_nodes_from(stations["station_name"])
    pos = None
    # 駅コードと駅名の辞書
    cd_to_name = dict(zip(stations["station_cd"], stations["station_name"]))

    # グラフに辺を追加する
    if distance_matrix is not None:
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
        for station_cd1, station_cd2 in join[["station_cd1", "station_cd2"]].itertuples(index=False):
            name1 = cd_to_name.get(station_cd1)
            name2 = cd_to_name.get(station_cd2)
            if name1 and name2:
                G.add_edge(name1, name2)

    # 描画の準備
    # pos が None の場合は stations に lon/lat があるかをチェックして設定
    if pos is None:
        if "lon" in stations.columns and "lat" in stations.columns:
            pos = {row["station_name"]: (row["lon"], row["lat"]) for _, row in stations.iterrows()}
        else:
            # ノードとエッジが追加された G を使ってレイアウトを作る
            pos = nx.spring_layout(G, seed=seed)

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_size=50)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="IPAexGothic")
    # ベースのエッジは薄いグレーで描画
    nx.draw_networkx_edges(G, pos, edge_color="#888888")

    # 選択ペアに基づいてハイライトするエッジを集める
    highlight_edges = set()
    dashed_lines = []
    if selected_pairs:
        for a, b in selected_pairs:
            # a/b が station_cd で渡された場合は名前に変換、そうでなければそのまま駅名として扱う
            name_a = cd_to_name.get(a, a)
            name_b = cd_to_name.get(b, b)
            if name_a not in G.nodes or name_b not in G.nodes:
                continue
            # まず重み付き最短経路で試す（weight 属性が存在する場合）
            try:
                path = nx.shortest_path(G, source=name_a, target=name_b, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # 経路がない
                dashed_lines.append((name_a, name_b))
                continue
            except Exception:
                # weight 指定が失敗した場合は重みなし最短経路を試す
                try:
                    path = nx.shortest_path(G, source=name_a, target=name_b)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    dashed_lines.append((name_a, name_b))
                    continue

            # path を辺集合に変換
            for x, y in zip(path[:-1], path[1:]):
                if (x, y) in G.edges:
                    highlight_edges.add((x, y))
                else:
                    highlight_edges.add((y, x))

    # ハイライトエッジを上に重ねて描画
    if highlight_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=list(highlight_edges),
            width=selected_width,
            edge_color=selected_color,
            alpha=selected_alpha,
        )

    # 到達不能ペアは破線で直接結ぶ
    for u, v in dashed_lines:
        xu, yu = pos[u]
        xv, yv = pos[v]
        plt.plot([xu, xv], [yu, yv], linestyle="--", color=selected_color, alpha=0.7)

    # エッジラベル（距離）を表示
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        w = data.get("weight")
        if w is not None:
            edge_labels[(u, v)] = f"{w:.2f}"

    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title("JR北海道の路線図 (selected pairs highlighted)")
    plt.axis("off")
    plt.show()

def visualize_graph_with_selected_pairs(
    stations: pd.DataFrame,
    join: pd.DataFrame,
    distance_matrix: np.ndarray | None = None,
    selected_pairs=None,
    selected_color: str = "red",
    selected_width: int = 2,
    selected_alpha: float = 0.9,
    seed: int = 0,
):
    from visualize import Visualizer

    Visualizer().visualize_graph_with_selected_pairs(
        stations, join, distance_matrix, selected_pairs, selected_color, selected_width, selected_alpha
    )


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
    # 可能であれば lon/lat カラムも保存しておく（後で可視化で利用するため）
    cols = ["station_cd", "station_name"]
    if "lon" in key_stations.columns and "lat" in key_stations.columns:
        cols += ["lon", "lat"]
    key_stations[cols].to_csv(list_path, index=False)

    return csv_path, list_path


def main(company_cd: int = 1, exclude_line_cds: list[int] | None = None):
    # 選択された company_cd に基づいて前処理を行う
    stations_sel, join_sel = preprocess_company(line, station, join, company_cd=company_cd, exclude_line_cds=exclude_line_cds)
    distance_matrix = calculate_distance_matrix(stations_sel, join_sel)
    visualize_graph(stations_sel, join_sel, distance_matrix)

    # 接続駅と終点駅を抽出
    key_stations, key_join = extract_key_stations(stations_sel, join_sel)
    # 接続駅と終点駅の距離を計算
    reduced_adj = build_reduced_adj_matrix(stations_sel, join_sel, key_stations)
    visualize_graph(key_stations, key_join, reduced_adj)
    # ファイル名は会社コードを含めて一意化
    prefix = f"company_{company_cd}_key_stations"
    save_reduced_adjmatrix_csv(
        reduced_adj, key_stations, out_dir, prefix=prefix
    )

    # 路線に沿った距離を計算するため，全駅のグラフを辿ってキー駅間の最短経路距離を求める
    key_distance_matrix = compute_key_station_pairwise_distances(
        stations_sel, join_sel, key_stations
    )
    # visualize_graph(key_stations, key_join, key_distance_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate maps for a selected railway company (by company_cd).")
    parser.add_argument("--company-cd", type=int, default=1, help="company_cd to process (default: 1 = JR Hokkaido)")
    parser.add_argument(
        "--exclude-lines",
        type=str,
        default=None,
        help="comma-separated line_cd values to exclude (e.g. 11118)",
    )
    args = parser.parse_args()

    exclude_list = None
    if args.exclude_lines:
        try:
            exclude_list = [int(x.strip()) for x in args.exclude_lines.split(",") if x.strip()]
        except ValueError:
            exclude_list = None

    main(company_cd=args.company_cd, exclude_line_cds=exclude_list)
