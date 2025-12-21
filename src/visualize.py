import networkx as nx
import matplotlib.pyplot as plt
import japanize_matplotlib
import math
import numpy as np
from typing import List, Tuple, Optional
import pandas as pd
from pathlib import Path


class Visualizer:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def visualize_graph_of_stations(
        self,
        stations: pd.DataFrame,
        join: pd.DataFrame,
        distance_matrix: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 10),
        save_path: Optional[str] = None,
    ):
        G = nx.Graph()
        # 頂点を駅名にする
        G.add_nodes_from(stations["station_name"])
        # plot の座標を設定（存在しない場合は None にしておく）
        pos = None
        if "lon" in stations.columns and "lat" in stations.columns:
            pos = {row["station_name"]: (row["lon"], row["lat"]) for _, row in stations.iterrows()}

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

        # レイアウトが未定義なら作成
        if pos is None:
            pos = nx.spring_layout(G, seed=self.seed)

        # グラフの描画
        plt.figure(figsize=figsize)
        nx.draw_networkx_nodes(G, pos, node_size=50)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family="IPAexGothic")
        nx.draw_networkx_edges(G, pos)

        # エッジに距離ラベルを付ける（weight 属性があれば小数点2桁で表示）
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            w = data.get("weight")
            if w is not None:
                edge_labels[(u, v)] = f"{w:.2f}"

        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        plt.axis("off")
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()

    def visualize_graph_with_selected_pairs(
        self,
        stations: pd.DataFrame,
        join: pd.DataFrame,
        distance_matrix: Optional[np.ndarray] = None,
        selected_pairs: Optional[List[Tuple]] = None,
        selected_color: str = "red",
        selected_width: int = 2,
        selected_alpha: float = 0.9,
        save_path: Optional[str] = None,
    ):
        G = nx.Graph()
        G.add_nodes_from(stations["station_name"])
        pos = None
        if "lon" in stations.columns and "lat" in stations.columns:
            pos = {row["station_name"]: (row["lon"], row["lat"]) for _, row in stations.iterrows()}

        cd_to_name = dict(zip(stations["station_cd"], stations["station_name"]))

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

        if pos is None:
            pos = nx.spring_layout(G, seed=self.seed)

        plt.figure(figsize=(10, 10))
        nx.draw_networkx_nodes(G, pos, node_size=50)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family="IPAexGothic")
        nx.draw_networkx_edges(G, pos, edge_color="#888888")

        highlight_edges = set()
        dashed_lines = []
        if selected_pairs:
            for a, b in selected_pairs:
                name_a = cd_to_name.get(a, a)
                name_b = cd_to_name.get(b, b)
                if name_a not in G.nodes or name_b not in G.nodes:
                    continue
                try:
                    path = nx.shortest_path(G, source=name_a, target=name_b, weight="weight")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    dashed_lines.append((name_a, name_b))
                    continue
                except Exception:
                    try:
                        path = nx.shortest_path(G, source=name_a, target=name_b)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        dashed_lines.append((name_a, name_b))
                        continue

                for x, y in zip(path[:-1], path[1:]):
                    if (x, y) in G.edges:
                        highlight_edges.add((x, y))
                    else:
                        highlight_edges.add((y, x))

        if highlight_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=list(highlight_edges),
                width=selected_width,
                edge_color=selected_color,
                alpha=selected_alpha,
            )

        for u, v in dashed_lines:
            xu, yu = pos[u]
            xv, yv = pos[v]
            plt.plot([xu, xv], [yu, yv], linestyle="--", color=selected_color, alpha=0.7)

        edge_labels = {}
        for u, v, data in G.edges(data=True):
            w = data.get("weight")
            if w is not None:
                edge_labels[(u, v)] = f"{w:.2f}"

        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        plt.axis("off")
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()

    def visualize_graph_from_adjmatrix(
        self, adj_matrix: List[List], seed: int = 0, selected_pairs: Optional[List[Tuple]] = None,
        selected_color: str = "red", selected_width: int = 3, selected_alpha: float = 0.9,
        save_path: Optional[str] = None,
    ):
        n = len(adj_matrix)
        G = nx.Graph()
        G.add_nodes_from(range(n))

        INF = math.inf
        for i in range(n):
            for j in range(i + 1, n):
                w = adj_matrix[i][j]
                if (w != 0) and (w != INF):
                    G.add_edge(i, j, weight=w)

        pos = nx.spring_layout(G, seed=seed)

        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue")
        nx.draw_networkx_edges(G, pos, width=2, edge_color="#888888")
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

        highlight_edges = set()
        dashed_lines = []
        if selected_pairs:
            for u, v in selected_pairs:
                if u not in G.nodes or v not in G.nodes:
                    continue
                try:
                    path = nx.shortest_path(G, source=u, target=v, weight="weight")
                    for a, b in zip(path[:-1], path[1:]):
                        if (a, b) in G.edges:
                            highlight_edges.add((a, b))
                        else:
                            highlight_edges.add((b, a))
                except nx.NetworkXNoPath:
                    dashed_lines.append((u, v))

        if highlight_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=list(highlight_edges),
                width=selected_width,
                edge_color=selected_color,
                alpha=selected_alpha,
            )

        for u, v in dashed_lines:
            xu, yu = pos[u]
            xv, yv = pos[v]
            plt.plot([xu, xv], [yu, yv], linestyle="--", color=selected_color, alpha=0.7)

        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.axis("off")
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()
