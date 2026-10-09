import numpy as np
import copy
import random

from .utils import fit_bezier_Endpointfixed, interpolate_lane_points, closest_curve

ego_width, ego_length = 1.85, 4.084


class Traj_Generator:
    def __init__(
        self,
        step=6,
        start_points=[
            np.array([0, 0, 0]),
            np.array([0, -3.5, 0]),
            np.array([0, 3.5, 0]),
        ],
    ):
        super().__init__()
        self.step = step
        self.start_points = start_points

    def build_connective_graph(self, lanes):
        graph = {}
        for i in range(len(lanes)):
            for j in range(len(lanes)):
                if i != j and np.linalg.norm(lanes[i][-1] - lanes[j][0]) <= 0.5:
                    if np.linalg.norm(lanes[i] - lanes[j]) >= 0.1:
                        graph.setdefault(i, []).append(j)
        return graph

    def build_starting_graph(self, lanes):
        graph = {}
        for i in range(len(lanes)):
            for j in range(len(lanes)):
                if i != j and np.linalg.norm(lanes[i][0] - lanes[j][0]) <= 0.5:
                    if np.linalg.norm(lanes[i] - lanes[j]) >= 0.1:
                        graph.setdefault(i, []).append(j)
        return graph

    def integrate_start_graph(self, graph, start_graph):
        # 将start_graph中的起点关系整合到原graph中
        for start_node, end_nodes in start_graph.items():
            if start_node in graph:
                graph[start_node].extend(end_nodes)
            else:
                graph[start_node] = end_nodes
        return graph

    def search_full_paths(self, connective_graph, starting_graph, nodes):
        graph = self.integrate_start_graph(connective_graph, starting_graph)
        # 找到所有起点和终点
        start_nodes = set(nodes) - set().union(*graph.values())  # 更新起点的定义
        end_nodes = set(nodes) - set(graph.keys())  # 更新终点的定义

        isolated_nodes = [
            node
            for node in nodes
            if node not in graph and node not in set().union(*graph.values())
        ]

        paths = []

        def dfs(current_path):
            current_node = current_path[-1]
            if current_node in end_nodes:
                paths.append(current_path)
                return
            for next_node in graph.get(current_node, []):
                if next_node not in current_path:  # 避免循环
                    dfs(current_path + [next_node])

        for start_node in start_nodes:
            dfs([start_node])

        for node in isolated_nodes:
            paths.append([node])

        return paths

    def generate_t(self, n):
        distribution_type = np.random.choice(["uniform", "increasing", "decreasing"])

        k = np.random.choice([0.2, 0.4, 0.6, 0.8])

        if distribution_type == "uniform":
            return [k / n] * n

        elif distribution_type == "increasing":
            increasing_list = sorted([random.random() for _ in range(n)])
            sum_original = sum(increasing_list)
            adjusted_list = [x / sum_original for x in increasing_list]

        elif distribution_type == "decreasing":
            decreasing_list = sorted([random.random() for _ in range(n)], reverse=True)
            sum_original = sum(decreasing_list)
            adjusted_list = [x / sum_original for x in decreasing_list]

        cum_sum = np.cumsum(adjusted_list)
        max_cum_sum = cum_sum[-1]
        final_list = [x / max_cum_sum * k for x in adjusted_list]

        return final_list

    def dfs(self, graph, start, path, visited, all_paths):
        visited.add(start)
        path.append(start)

        if start not in graph or not graph[start]:
            all_paths.append(path.copy())
        else:
            for next_lane in graph[start]:
                if next_lane not in visited:
                    self.dfs(graph, next_lane, path, visited, all_paths)

        path.pop()
        visited.remove(start)

    def search_path(self, lane_pts):
        all_paths = []
        full_paths = []
        start_pts_index = []
        start_lane_index = []

        inter_lane_pts = []
        for lane in lane_pts:
            inter_lane_pts.append(
                interpolate_lane_points(
                    fit_bezier_Endpointfixed(lane[..., :2], 4), 100
                ).numpy()
            )

        connective_graph = self.build_connective_graph(lane_pts)
        starting_graph = self.build_starting_graph(lane_pts)
        full_paths = self.search_full_paths(
            copy.deepcopy(connective_graph),
            starting_graph,
            [i for i in range(len(lane_pts))],
        )
        for pt in self.start_points:
            lane_index, dist, pts_index, angle_diff = closest_curve(pt, inter_lane_pts)
            if lane_index not in start_lane_index and angle_diff < 2.5:  # for u-turn
                sub_paths = []
                self.dfs(connective_graph, lane_index, [], set(), sub_paths)
                start_lane_index.append(lane_index)
                all_paths += sub_paths
                start_pts_index += [pts_index] * len(sub_paths)

        all_paths_pts = []
        for i, indices in enumerate(all_paths):
            pts_index = start_pts_index[i]
            sub_list = []
            for index in indices:
                sub_list.extend(inter_lane_pts[index])
            sub_path = np.stack([np.array([0, 0])] + sub_list[pts_index:], axis=0)
            all_paths_pts.append(sub_path)

        return all_paths_pts, full_paths
