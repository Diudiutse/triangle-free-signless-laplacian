"""Verify only equations (61) and (62) in Lemma 5.

Read the existing flag lists, the two saved rational vectors b, and V_sigma.
All count vectors use the order of the flags in F_sigma[sigma].
Fractions and NumPy's dtype=object keep every calculation exact.

Run from the repository root: python programs/verify_equality.py
"""

from fractions import Fraction
from pathlib import Path
import json
import pickle

import numpy as np

from sigma_data import sigma_data


PROGRAMS = Path(__file__).resolve().parent


def main():
    # Step 1. Read the existing flags and compute d(P) for the one-root type.
    # A graph is (vertices, edges); every edge is stored as (a,b) with a<b.
    # The first k vertices are roots, so v>k means that v is unlabeled.
    with (PROGRAMS / "results" / "05" / "F_sigma.pkl").open("rb") as file:
        F_sigma = pickle.load(file)

    sigma_one_root = ((1,), ())
    sigma_two_roots = ((1, 2), ())

    dP_one_root = []
    for vertices, edges in F_sigma[sigma_one_root]:
        dP_one_root.append(len([v for v in vertices if v > 1 and (1, v) in edges]))
    dP_one_root = np.array(dP_one_root, dtype=object)

    # Step 2. Compute c(P) and d(P) for the two-root type.
    # c(P) counts unlabeled vertices adjacent to both roots.
    # d(P) counts unlabeled vertices adjacent to root 1; (62) needs it too.
    cP = []
    dP_two_roots = []
    for vertices, edges in F_sigma[sigma_two_roots]:
        cP.append(len([
            v for v in vertices if v > 2 and (1, v) in edges and (2, v) in edges
        ]))
        dP_two_roots.append(len([v for v in vertices if v > 2 and (1, v) in edges]))
    cP = np.array(cP, dtype=object)
    dP_two_roots = np.array(dP_two_roots, dtype=object)

    # Step 3. Read the two existing b vectors and the corresponding V_sigma.
    # The JSON stores only b_one_root and b_two_roots as rational-number strings.
    # Fraction converts each string to an exact rational number.
    with (PROGRAMS / "equality_vectors.json").open(encoding="utf-8") as file:
        saved_b = json.load(file)
    b_one_root = np.array([Fraction(value) for value in saved_b["b_one_root"]], dtype=object)
    b_two_roots = np.array([Fraction(value) for value in saved_b["b_two_roots"]], dtype=object)

    V_one_root = np.array(sigma_data[sigma_one_root]["V_sigma"], dtype=object)
    V_two_roots = np.array(sigma_data[sigma_two_roots]["V_sigma"], dtype=object)

    # Step 4. Check the two vector identities, coordinate by coordinate.
    # @ is matrix-vector multiplication; the other products are entrywise.
    # array_equal compares the complete vectors without broadcasting.
    if not np.array_equal(
        V_one_root @ b_one_root,
        25 * dP_one_root * (dP_one_root - 1) - 11 * dP_one_root,
    ):
        raise ValueError("Equation (61) failed.")
    print("(61) passed: all " + str(len(dP_one_root)) + " coordinates agree exactly.")

    if not np.array_equal(
        V_two_roots @ b_two_roots,
        25 * cP * (cP - 1) - 14 * cP + 3 * dP_two_roots,
    ):
        raise ValueError("Equation (62) failed.")
    print("(62) passed: all " + str(len(cP)) + " coordinates agree exactly.")


if __name__ == "__main__":
    main()
