"""
第六步：计算全部 C_H^sigma，保存为 {(sigma,H): 矩阵}。
matrix[p][p_prime] 对应 F_sigma[sigma][p]、F_sigma[sigma][p_prime]。
矩阵用行列表组成的列表存储，每个元素都是 Python 整数。

compute_C_matrix(H,sigma,flag_list) 一次计数整张矩阵，避免每个元素重复枚举。
两者使用同一个计数公式：固定有序根，选两组不相交的无标签顶点，
每次合格选择贡献 (m!)^2 * (8-k-2m)!，其中 m=ell-k。
这分别计入两组顶点的排列和剩余未使用顶点的排列，故等于完整双射数。

只读取第二、五步的 pickle，不读取旧的 C_H^sigma 数值。
输出 results/06/C.pkl。运行：python 06_compute_C.py
"""

from functools import cache, lru_cache
from importlib import import_module
from itertools import combinations, permutations
from math import factorial
from pathlib import Path
import pickle


labeled_graphs_are_isomorphic = import_module("05_generate_flags").labeled_graphs_are_isomorphic


def induced_labeled_graph(H, vertex_order):
    """取 H 的诱导子图，将 vertex_order 的第 a 个顶点重新编号为 a（从 1 开始）。"""
    # vertex_order 是互不相同的 H 顶点组成的 tuple。
    # 调用时根排在最前面，故重新编号后的前 k 个顶点就是固定根标签。
    H_edges = set(H[1])
    edges = []
    for a in range(len(vertex_order)):
        for b in range(a + 1, len(vertex_order)):
            edge_in_H = tuple(sorted([vertex_order[a], vertex_order[b]]))
            if edge_in_H in H_edges:
                edges.append((a + 1, b + 1))
    return (tuple(range(1, len(vertex_order) + 1)), tuple(edges))


@lru_cache(maxsize=7)
def root_orders_by_sigma(H, k):
    """返回 {根诱导图 sigma: 有序根 tuple 的列表}，包括所有 k 个互异根的选择。"""
    # 同一个 H 的同阶类型共用根枚举；只保留最近七次结果，控制内存。
    root_orders = {}
    for roots in permutations(H[0], k):
        sigma = induced_labeled_graph(H, roots)
        if sigma not in root_orders:
            root_orders[sigma] = []
        root_orders[sigma].append(roots)
    return root_orders


@cache
def find_flag_index(graph, flag_list, k):
    """返回 graph 所属旗在 flag_list 中的下标；flag_list 为 tuple，便于复用结果。"""
    # 同一张带标签图再次出现时，cache 直接返回原来找到的下标。
    for p, representative in enumerate(flag_list):
        if labeled_graphs_are_isomorphic(graph, representative, k):
            return p
    raise ValueError("诱导旗在第五步生成的旗列表中没有同构代表。")


def compute_C_matrix(H, sigma, flag_list):
    """返回整张 C_H^sigma；行列顺序就是输入 flag_list 的顺序。"""
    if len(H[0]) != 8:
        raise ValueError("H 必须有八个顶点。")
    k = len(sigma[0])
    ell = (8 + k) // 2
    m = ell - k
    flag_list = tuple(flag_list)
    matrix = []
    for p in range(len(flag_list)):
        matrix.append([0] * len(flag_list))

    # 两组无标签顶点各有 m! 种排列；未使用顶点有 (8-k-2m)! 种排列。
    # 根已经有顺序，不再乘 k!。每个完整双射唯一落入一次这样的选择。
    multiplicity = factorial(m) ** 2 * factorial(8 - k - 2 * m)
    for roots in root_orders_by_sigma(H, k).get(sigma, []):
    # roots 遍历所有V(H)的使得标签为sigma的k-tuple
        remaining = tuple(v for v in H[0] if v not in roots)
        for first_vertices in combinations(remaining, m):
            first_graph = induced_labeled_graph(H, roots + first_vertices)
            p = find_flag_index(first_graph, flag_list, k)
            remaining_after_first = tuple(v for v in remaining if v not in first_vertices)
            for second_vertices in combinations(remaining_after_first, m):
                second_graph = induced_labeled_graph(H, roots + second_vertices)
                p_prime = find_flag_index(second_graph, flag_list, k)
                matrix[p][p_prime] = matrix[p][p_prime] + multiplicity
    return matrix


def main():
    root = Path(__file__).resolve().parent
    with (root / "results" / "02" / "tau_8.pkl").open("rb") as file:
        tau_8 = pickle.load(file)
    with (root / "results" / "05" / "F_sigma.pkl").open("rb") as file:
        F_sigma = pickle.load(file)

    C = {}
    for h, H in enumerate(tau_8):
        for sigma in F_sigma:
            C[(sigma, H)] = compute_C_matrix(H, sigma, F_sigma[sigma])
        if (h + 1) % 10 == 0:
            print("Computed all type matrices for " + str(h + 1) + " graphs.", flush=True)

    output_folder = root / "results" / "06"
    output_folder.mkdir(parents=True, exist_ok=True)
    with (output_folder / "C.pkl.tmp").open("wb") as file:
        pickle.dump(C, file)
    (output_folder / "C.pkl.tmp").replace(output_folder / "C.pkl")
    print("Saved " + str(len(C)) + " matrices to " + str(output_folder / "C.pkl"))


if __name__ == "__main__":
    main()
