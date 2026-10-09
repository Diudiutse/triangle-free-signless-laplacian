"""
核验文章 (40)，对每个 H 检查：
  Delta/2^80 * tr(L(qI+X_star)L^T F_H)
  + sum_sigma Delta/(40320*a_sigma^2)
      * tr(V_sigma(qI+X_sigma)V_sigma^T C_H_sigma) = 0。

固定数据只来自 spectral_data.py 和 sigma_data.py。
main 加载第二、三、五、六步的 pickle；F_H 和 C 均使用重新计算的结果。
Delta 是从 spectral_data.py 导入的固定正整数，用于清除分母。
矩阵运算直接使用 NumPy；dtype=object 保持 Python 大整数和 Fraction 的精确运算。
运行：python verify_40.py
"""

from fractions import Fraction
from pathlib import Path
import json
import pickle
import numpy as np

from spectral_data import L, q, Delta, X_star
from sigma_data import sigma_data


def main():
    root = Path(__file__).resolve().parent
    with (root / "results" / "02" / "tau_8.pkl").open("rb") as file:
        tau_8 = pickle.load(file)
    with (root / "results" / "03" / "F_H.pkl").open("rb") as file:
        F_H = pickle.load(file)
    with (root / "results" / "05" / "F_sigma.pkl").open("rb") as file:
        F_sigma = pickle.load(file)
    with (root / "results" / "06" / "C.pkl").open("rb") as file:
        C = pickle.load(file)
    if set(F_H) != set(tau_8) or set(sigma_data) != set(F_sigma):
        raise ValueError("固定数据和计算结果的图或类型不一致。")
    if set(C) != {(sigma, H) for sigma in F_sigma for H in tau_8}:
        raise ValueError("C 的键没有恰好覆盖所有 (sigma,H)。")

    for sigma, data in sigma_data.items():
        if data["a_sigma"] <= 0:
            raise ValueError("a_sigma 必须为正。")
        if Delta % (40320 * data["a_sigma"] ** 2) != 0:
            raise ValueError("Delta 未清除该 sigma 的分母。")
        if len(data["V_sigma"]) != len(F_sigma[sigma]):
            raise ValueError("旗的顺序或矩阵尺寸不一致。")
        for H in tau_8:
            if len(C[(sigma, H)]) != len(F_sigma[sigma]) or any(
                len(row) != len(F_sigma[sigma]) for row in C[(sigma, H)]
            ):
                raise ValueError("C_H_sigma 的尺寸与 F_sigma 不一致。")

    # @ 是矩阵乘法，.T 是转置，object 保持整数与 Fraction 的精确运算。
    L_matrix = np.array(L, dtype=object)
    X_star_matrix = np.array(X_star, dtype=object)
    B = (
        L_matrix @ (q * np.eye(5, dtype=object) + X_star_matrix) @ L_matrix.T
    )

    # 逐张 H 检查 (40)，后一项直接对 sigma 求和。
    # 迹的循环性给出 tr(V(qI+X)V^T C) = tr((qI+X)(V^T C V))。
    # 再用 tr(MN)=sum_{i,j} M[j,i]*N[i,j]，由 np.sum(M.T*N) 直接计算逐项相乘后求和即可。
    print("Computing (40) for all " + str(len(tau_8)) + " graphs.", flush=True)
    for h, H in enumerate(tau_8):
        # 这些列表都按 sigma_data 中类型的同一顺序排列。
        # 求和中的 s 是列表下标；各列表的第 s 项属于同一个类型。
        Vsigma = [
            np.array(sigma_data[sigma]["V_sigma"], dtype=object)
            for sigma in sigma_data
        ]
        FH = np.array(
            [[F_H[H][(i, j)] for j in range(5)] for i in range(5)],
            dtype=object,
        )
        Xsigma = [
            np.array(sigma_data[sigma]["X_sigma"], dtype=object)
            for sigma in sigma_data
        ]
        CHsigma = [np.array(C[(sigma, H)], dtype=object) for sigma in sigma_data]
        asigma = [sigma_data[sigma]["a_sigma"] for sigma in sigma_data]

        if (
            Fraction(Delta, 2 ** 80) * np.trace(B @ FH)
            + sum(
                Fraction(Delta, 40320 * asigma[s] ** 2)
                * np.sum(
                    (q * np.eye(len(Xsigma[s]), dtype=object) + Xsigma[s]).T
                    * (Vsigma[s].T @ CHsigma[s] @ Vsigma[s])
                )
                for s in range(len(Vsigma))
            )
        ) != 0:
            raise ValueError("(40) 在 H" + str(h) + " 处失败。")
        if (h + 1) % 10 == 0:
            print("Checked " + str(h + 1) + " graphs.", flush=True)

    report = {
        "equation": 40, "status": "passed", "source_graphs": len(tau_8),
        "types": len(sigma_data), "exact_arithmetic": True,
        "source_totals": [0 for H in tau_8],
        "all_source_totals_exactly_zero": True,
    }
    folder = Path(__file__).resolve().parent / "results"
    folder.mkdir(exist_ok=True)
    (folder / "verify_40.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("(40) passed: all " + str(len(tau_8)) + " totals are exactly zero.")


if __name__ == "__main__":
    main()
