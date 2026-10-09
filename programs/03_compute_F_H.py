"""
第三步：加载 c_ij_J.pkl 和 tau_8.pkl，按公式
    (F_H)_{ij} = sum_J c_{ij,J} * t_inj(J,H)
计算并保存一个字典：{H: {(i,j): Fraction}}。

图 J、H 都用 (顶点 tuple, 边 tuple) 表示，顶点编号为 1,...,顶点数。
tinj(J,H) 只要求保持 J 的边，不要求保持非边。
NumPy 用于同时检查全部单射；计数为整数，除法和求和使用 Fraction。
运行：python 03_compute_F_H.py
"""

from fractions import Fraction
from functools import cache
from itertools import permutations
from pathlib import Path
import pickle
import numpy as np


# 同样大小的 J、H 使用同一张单射列表，避免重复枚举；不保存到文件。
injections_by_size = {}


@cache  # 相同的 J、H 再次出现时直接使用已算出的密度。
def tinj(J, H):
    """返回 t_inj(J,H) = 保持每条边的单射数 / 全部单射数。要求 |J| <= |H|。"""
    s = len(J[0])
    n = len(H[0])
    if s > n:
        raise ValueError("tinj 的定义要求 J 的顶点数不超过 H 的顶点数。")
    if s == 0:
        return Fraction(1)

    if (n, s) not in injections_by_size:
        injections_by_size[(n, s)] = np.array(
            list(permutations(range(1, n + 1), s)), dtype=np.int64
        )
    injections = injections_by_size[(n, s)]
    # 每一行是一个单射 phi；第 a-1 列记录 phi(a)。

    adjacency = np.zeros((n + 1, n + 1), dtype=bool)
    # adjacency 是 H 的邻接矩阵，存为 (n+1) 行、(n+1) 列的 NumPy 布尔数组。
    # adjacency[u,v] 为 True 当且仅当 {u,v} 是 H 的边，否则为 False。
    # H 的顶点编号为 1,...,n，所以第 0 行、第 0 列空着不用；对角线为 False。
    for a, b in H[1]:
        adjacency[a, b] = True
        adjacency[b, a] = True

    valid = np.ones(len(injections), dtype=bool)
    for a, b in J[1]:
        images_of_a = injections[:, a - 1]
        # images_of_a 是长度为 len(injections) 的一维整数数组。
        # 冒号 : 表示取所有行，a-1 表示取对应顶点 a 的那一列。
        # 若第 r 行对应单射 phi_r，则 images_of_a[r] = phi_r(a)，是 H 的顶点编号。
        images_of_b = injections[:, b - 1]
        # images_of_b 同样是一维整数数组，images_of_b[r] = phi_r(b)。
        # 例如两行单射分别把 (a,b) 映到 (2,5)、(4,1)，则这两个数组分别是
        # [2,4] 和 [5,1]；相同下标 r 的两个元素来自同一个单射。
        # adjacency[images_of_a, images_of_b] 按相同下标逐对查邻接矩阵，
        # 得到一维布尔数组；第 r 项是 adjacency[phi_r(a),phi_r(b)]，
        # 表示第 r 个单射是否保持当前边 {a,b}。
        valid = valid & adjacency[images_of_a, images_of_b]
    # valid[r] 为 True，恰好表示第 r 个单射保持 J 的全部边。
    return Fraction(int(np.count_nonzero(valid)), len(injections))


def main():
    root = Path(__file__).resolve().parent
    with (root / "results" / "01" / "c_ij_J.pkl").open("rb") as file:
        c_ij_J = pickle.load(file)
    with (root / "results" / "02" / "tau_8.pkl").open("rb") as file:
        tau_8 = pickle.load(file)

    F_H = {}
    for H in tau_8:
        F_H[H] = {}
        for i in range(5):
            for j in range(5):
                value = Fraction(0)
                for J, coefficient in c_ij_J[(i, j)].items():
                    value = value + coefficient * tinj(J, H)
                F_H[H][(i, j)] = value
        if len(F_H) % 10 == 0:
            print("Computed F_H for " + str(len(F_H)) + " graphs.", flush=True)

    output_folder = root / "results" / "03"
    output_folder.mkdir(parents=True, exist_ok=True)
    with (output_folder / "F_H.pkl").open("wb") as file:
        pickle.dump(F_H, file)
    print("Saved " + str(len(F_H)) + " matrices to " + str(output_folder / "F_H.pkl"))


if __name__ == "__main__":
    main()
