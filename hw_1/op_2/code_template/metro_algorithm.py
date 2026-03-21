# Copyright 2026, Yumeng Liu @ USTC

"""
地铁网络算法模块 —— 数据加载、图构建、Dijkstra 求解
"""

import csv
import heapq
from pathlib import Path

import numpy as np


# ============================================================
# Graph 数据结构
# ============================================================

class Graph:
    """
    简单的无向加权图。

    需要实现的接口
    -------------
    - add_node(node_id, **attrs) : 添加节点
    - add_edge(u, v, weight)     : 添加无向边
    - neighbors(node_id)         : 返回邻居字典 {neighbor_id: weight}
    - number_of_nodes()          : 返回节点数
    - number_of_edges()          : 返回边数
    - edges()                    : 返回所有边列表 [(u, v, weight), ...]

    属性
    ----
    nodes : dict[int, dict]
        节点字典，{node_id: {"name": str, ...}}。
        GUI 会读取此属性来获取节点信息，请确保 add_node 时正确填充。

    提示
    ----
    你可以自由选择底层数据结构（邻接表、邻接矩阵、边列表等）。
    """

    def __init__(self):
        self.nodes = {}
        self.edges_ = []
        # TODO: 初始化你的数据结构

    def add_node(self, node_id, **attrs):
        """
        添加节点。

        Parameters
        ----------
        node_id : int
            节点编号。
        **attrs
            节点属性，例如 name="StationA"。
        """
        # TODO: 将节点及其属性存入 self.nodes，并初始化邻接结构
        self.nodes[node_id] = attrs
        if "neighbors" not in self.nodes[node_id].keys():
            self.nodes[node_id]["neighbors"] = {}

    def add_edge(self, u, v, weight=1.0):
        """
        添加无向边 (u, v)，权重为 weight。
        """
        # TODO: 在邻接结构中记录无向边及权重
        if u not in self.nodes.keys():
            raise ValueError("Point %f is not exist!"%u)
        if v not in self.nodes.keys():
            raise ValueError("Point %f is not exist!"%v)
        if v in self.nodes[u]["neighbors"].keys():
            index = -1
            for i in range(len(self.edges_)):
                if (self.edges_[i][0] == u and self.edges_[i][1] == v) \
                    or (self.edges_[i][0] == v and self.edges_[i][1] == u):
                    index = i
                    break
            if index != -1:
                self.edges_.pop(index)
                self.edges_.append((u, v, weight))
            else:
                self.edges_.append((u, v, weight))
        else:
            self.edges_.append((u, v, weight))
        self.nodes[u]["neighbors"][v] = weight
        self.nodes[v]["neighbors"][u] = weight


    def neighbors(self, node_id):
        """
        返回 node_id 的邻居字典 {neighbor_id: weight}。

        若节点不存在或无邻居，返回空字典。
        """
        # TODO: 返回邻居及对应权重
        if node_id not in self.nodes.keys():
            return {}
        else:
            return self.nodes[node_id]["neighbors"]

    def number_of_nodes(self):
        """返回图中节点数量。"""
        # TODO
        return len(self.nodes)

    def number_of_edges(self):
        """返回图中边的数量（每条无向边只计一次）。"""
        # TODO
        return len(self.edges_)

    def edges(self):
        """
        返回所有边的列表 [(u, v, weight), ...]，每条边只出现一次。

        GUI 的绘图函数会调用此方法来绘制网络边。
        """
        # TODO
        return self.edges_


# ============================================================
# 数据加载
# ============================================================

def load_station_map(tsv_path: str) -> dict[int, str]:
    """读取 station-id-map.tsv，返回 {id: name} 映射。"""
    stations: dict[int, str] = {}
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stations[int(row["id"])] = row["name"]
    return stations


def load_adjacency_matrix(csv_path: str) -> np.ndarray:
    """读取 adjacency-distance.csv，返回 N×N numpy 矩阵。"""
    return np.loadtxt(csv_path, delimiter=",")


def build_graph(stations: dict[int, str], adj: np.ndarray) -> Graph:
    """
    根据站点映射和邻接矩阵构建加权图。

    Parameters
    ----------
    stations : dict[int, str]
        站点 id → 名称映射（id 从 1 开始）。
    adj : np.ndarray
        N×N 邻接距离矩阵，adj[i,j] > 0 表示站点 i+1 与 j+1 之间有边。

    Returns
    -------
    Graph
        带权无向图，节点属性 name 为站名，边权 weight 为距离。

    提示
    ----
    - 使用 Graph.add_node(node_id, name=...) 添加节点
    - 使用 Graph.add_edge(u, v, weight=...) 添加边
    - 矩阵下标从 0 开始，站点 id 从 1 开始
    """
    # TODO: 构建加权图
    g = Graph()
    for station_id, name in stations.items():
        g.add_node(station_id, name=name)
    for i in range(adj.shape[0]):
        for j in range(i,adj.shape[1]):
            if adj[i, j] > 0:
                g.add_edge(i+1, j+1, adj[i, j])
    return g


# ============================================================
# Dijkstra 最短路径
# ============================================================

def dijkstra(G: Graph, src: int, dst: int) -> tuple[float, list[int]]:
    """
    实现 Dijkstra 求 src → dst 最短路径。

    Parameters
    ----------
    G : Graph
        带权图。
    src : int
        起点站点 id。
    dst : int
        终点站点 id。

    Returns
    -------
    (cost, path) : (float, list[int])
        cost 为最短距离，path 为站点 id 序列（含起终点）。
        若不可达，返回 (float("inf"), [])。

    提示
    ----
    - 使用 G.neighbors(u) 获取邻居字典 {neighbor_id: weight}
    - 使用 heapq 实现最小堆
    - 使用前驱字典 prev 回溯路径
    """
    # TODO: 实现 Dijkstra 算法
    node_remain = list(G.nodes.keys())
    if src not in node_remain or dst not in node_remain:
        return float("inf"), []
    curr_node = src
    node_prev = {}
    node_leng = {x:float('inf') for x in node_remain}
    node_leng[src] = 0
    node_remain.remove(src)
    while dst in node_remain:
        for neighbor_node, weight in G.neighbors(curr_node).items():
            if weight + node_leng[curr_node] < node_leng[neighbor_node]:
                node_prev[neighbor_node] = curr_node
                node_leng[neighbor_node] = node_leng[curr_node] + weight
        min_len = float('inf')
        next_node = None
        for node in node_remain:
            if node_leng[node] < min_len:
                min_len = node_leng[node]
                next_node = node
        if next_node is not None:
            curr_node = next_node
        else :
            return float("inf"), []
        node_remain.remove(curr_node)

    path = [dst]
    while path[0] != src:
        path.insert(0,node_prev[path[0]])
    return node_leng[dst], path


# ============================================================
# MetroSystem 高层封装
# ============================================================

class MetroSystem:
    """封装单个城市的地铁系统：加载数据、构建图、求解路径。"""

    def __init__(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        self.city = data_dir.name

        tsv = next(data_dir.glob("*station-id-map.tsv"))
        csv_f = next(data_dir.glob("*adjacency-distance.csv"))

        self.stations = load_station_map(str(tsv))
        adj = load_adjacency_matrix(str(csv_f))
        self.graph = build_graph(self.stations, adj)

        self.name_to_id: dict[str, int] = {
            name: sid for sid, name in self.stations.items()
        }

    def sorted_station_names(self) -> list[str]:
        """返回按字母排序的站名列表。"""
        return sorted(self.stations.values())

    def shortest_path(self, src_name: str, dst_name: str) -> tuple[float, list[int]]:
        """
        求两站之间的最短路径。

        Parameters
        ----------
        src_name : str
            起点站名。
        dst_name : str
            终点站名。

        Returns
        -------
        (cost, path) : (float, list[int])
            cost 为最短距离 (km)，path 为站点 id 序列。

        提示
        ----
        - 使用 self.name_to_id 将站名转为 id
        - 调用 dijkstra(self.graph, src_id, dst_id)
        """
        # TODO: 将站名转为 id，调用 dijkstra 函数求解
        src = self.name_to_id[src_name]
        dst = self.name_to_id[dst_name]
        leng, path = dijkstra(self.graph, src, dst)
        return leng, path


def detect_cities(data_root: str | Path) -> list[str]:
    """扫描 data_root 下所有包含数据文件的城市子目录。"""
    data_root = Path(data_root)
    cities: list[str] = []
    for d in sorted(data_root.iterdir()):
        if d.is_dir() and list(d.glob("*adjacency-distance.csv")):
            cities.append(d.name)
    return cities

if __name__ == "__main__":
    g = Graph()
    g.add_node(1)
    g.add_node(2)
    g.add_node(3)
    g.add_node(4)
    g.add_edge(1, 2, 10)
    g.add_edge(2, 3, 6)
    g.add_edge(3, 4, 10)
    g.add_edge(1, 4, 2)
    g.add_edge(2, 4, 1)
    print(dijkstra(g, 1, 3))