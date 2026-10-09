"""
第五步：对每个 sigma，生成前 k 个顶点带固定标签的图同构类代表。
旗的顶点数 ell=(8+k)//2；前 k 个顶点诱导出的图必须等于 sigma。

读取 results/04/sigma_list.pkl，输出 results/05/F_sigma.pkl。
输出是 {sigma: 图列表}；sigma 和每张图都用 (顶点 tuple, 边 tuple) 表示。
同构判别只枚举无标签顶点的置换，不调用图同构库。
运行：python 05_generate_flags.py
"""

from importlib import import_module
from itertools import permutations
from pathlib import Path
import pickle


# 复用第二步的子集枚举函数；导入不会执行第二步的 main。
all_subsets = import_module("02_generate_tau8").all_subsets


def labeled_graphs_are_isomorphic(first_graph, second_graph, k):
    """两图顶点均为 1,...,n；判断是否存在逐个固定 1,...,k 的图同构。"""
    n = len(first_graph[0])
    if n != len(second_graph[0]):
        return False
    if not 0 <= k <= n:
        raise ValueError("标签规模 k 必须介于 0 和图的顶点数之间。")

    first_edges = set(first_graph[1])
    second_edges = set(second_graph[1])
    for free_images in permutations(range(k + 1, n + 1)):
        # phi[a] 是顶点 a 的像；第 0 项不用。
        # 前 k 个顶点的像仍是它们自己，后面的像由本次置换给出。
        phi = list(range(k + 1)) + list(free_images)
        preserves_adjacency = True
        for a in range(1, n + 1):
            for b in range(a + 1, n + 1):
                first_has_edge = (a, b) in first_edges #输出判断(a,b)是否属于 first_edges 的布尔值
                second_has_edge = tuple(sorted([phi[a], phi[b]])) in second_edges
                # 同时检查边和非边：两边的邻接关系必须完全相同。
                if first_has_edge != second_has_edge:
                    preserves_adjacency = False
                    break
            if not preserves_adjacency:
                break
        if preserves_adjacency:
            return True
    return False


def generate_sigma_flags(sigma):
    """从 sigma 开始增加无标签顶点，返回 ell 顶点的全部 sigma-旗代表。"""
    k = len(sigma[0])
    ell = (8 + k) // 2
    graphs = [sigma]
    for n in range(k + 1, ell + 1):
        new_graphs = []
        for graph in graphs:
            for neighbors in all_subsets(graph[0]):
                # 新顶点 n 的邻居必须构成独立集，否则会产生三角形。
                independent = True
                for a, b in graph[1]:
                    if a in neighbors and b in neighbors:
                        independent = False
                        break
                if not independent:
                    continue

                edges = list(graph[1])
                for vertex in neighbors:
                    edges.append((vertex, n))
                edges.sort()
                new_graph = (tuple(range(1, n + 1)), tuple(edges))

                already_present = False
                for representative in new_graphs:
                    if labeled_graphs_are_isomorphic(new_graph, representative, k):
                        already_present = True
                        break
                if not already_present:
                    new_graphs.append(new_graph)
        graphs = new_graphs
    return graphs


def main():
    root = Path(__file__).resolve().parent
    with (root / "results" / "04" / "sigma_list.pkl").open("rb") as file:
        sigma_list = pickle.load(file)

    F_sigma = {}
    for sigma in sigma_list:
        F_sigma[sigma] = generate_sigma_flags(sigma)

    output_folder = root / "results" / "05"
    output_folder.mkdir(parents=True, exist_ok=True)
    with (output_folder / "F_sigma.pkl").open("wb") as file:
        pickle.dump(F_sigma, file)
    print("Saved flag lists for " + str(len(F_sigma)) + " types to " + str(output_folder / "F_sigma.pkl"))


if __name__ == "__main__":
    main()
