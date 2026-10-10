"""
Verify equation (40) for every H:
  Delta/2^80 * tr(L(qI+X_star)L^T F_H)
  + sum_sigma Delta/(40320*a_sigma^2)
      * tr(V_sigma(qI+X_sigma)V_sigma^T C_H_sigma) = 0.

Import fixed data only from spectral_data.py and sigma_data.py.
main loads the outputs of stages 2, 3, 5, and 6, using the recomputed F_H and C.
Delta is the fixed positive integer imported from spectral_data.py to clear denominators.
NumPy matrix operations use dtype=object to retain exact Python integers and Fractions.
Run: python verify_40.py
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
        raise ValueError("The graph or type catalogs disagree with the fixed data or computed outputs.")
    if set(C) != {(sigma, H) for sigma in F_sigma for H in tau_8}:
        raise ValueError("The keys of C do not cover exactly all (sigma,H) pairs.")

    for sigma, data in sigma_data.items():
        if data["a_sigma"] <= 0:
            raise ValueError("a_sigma must be positive.")
        if Delta % (40320 * data["a_sigma"] ** 2) != 0:
            raise ValueError("Delta does not clear the denominator for this sigma.")
        if len(data["V_sigma"]) != len(F_sigma[sigma]):
            raise ValueError("The flag order or matrix dimensions are inconsistent.")
        for H in tau_8:
            if len(C[(sigma, H)]) != len(F_sigma[sigma]) or any(
                len(row) != len(F_sigma[sigma]) for row in C[(sigma, H)]
            ):
                raise ValueError("The dimensions of C_H_sigma do not match F_sigma.")

    # @ is matrix multiplication, .T is transpose, and object retains exact integers and Fractions.
    L_matrix = np.array(L, dtype=object)
    X_star_matrix = np.array(X_star, dtype=object)
    B = (
        L_matrix @ (q * np.eye(5, dtype=object) + X_star_matrix) @ L_matrix.T
    )

    # Check (40) for each H, writing the second term directly as a sum over sigma.
    # Cyclicity of trace gives tr(V(qI+X)V^T C) = tr((qI+X)(V^T C V)).
    # Use tr(MN)=sum_{i,j} M[j,i]*N[i,j]: np.sum(M.T*N) sums the entrywise products.
    print("Computing (40) for all " + str(len(tau_8)) + " graphs.", flush=True)
    for h, H in enumerate(tau_8):
        # These lists all use the same type order as sigma_data.
        # s is a list index; the s-th entry of each list belongs to the same type.
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
            raise ValueError("(40) failed at H" + str(h) + ".")
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
