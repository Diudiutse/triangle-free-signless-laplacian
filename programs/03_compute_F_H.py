"""
Stage 3: load c_ij_J.pkl and tau_8.pkl and use
    (F_H)_{ij} = sum_J c_{ij,J} * t_inj(J,H)
to compute and save the dictionary {H: {(i,j): Fraction}}.

Graphs J and H are (vertex tuple, edge tuple), with vertices labeled 1,...,order.
tinj(J,H) preserves every edge of J but need not preserve nonedges.
NumPy checks all injections simultaneously; counts are integers, and division
and summation use Fraction.
Run: python 03_compute_F_H.py
"""

from fractions import Fraction
from functools import cache
from itertools import permutations
from pathlib import Path
import pickle
import numpy as np


# Reuse the injection array for the same graph orders; it is not written to disk.
injections_by_size = {}


@cache  # Reuse a computed density when the same J and H occur again.
def tinj(J, H):
    """Return t_inj(J,H): edge-preserving injections divided by all injections; require |J|<=|H|."""
    s = len(J[0])
    n = len(H[0])
    if s > n:
        raise ValueError("tinj requires the order of J to be at most the order of H.")
    if s == 0:
        return Fraction(1)

    if (n, s) not in injections_by_size:
        injections_by_size[(n, s)] = np.array(
            list(permutations(range(1, n + 1), s)), dtype=np.int64
        )
    injections = injections_by_size[(n, s)]
    # Each row is an injection phi; column a-1 records phi(a).

    adjacency = np.zeros((n + 1, n + 1), dtype=bool)
    # adjacency is the adjacency matrix of H, an (n+1)-by-(n+1) NumPy Boolean array.
    # adjacency[u,v] is True exactly when {u,v} is an edge of H, and False otherwise.
    # Labels are 1,...,n, so row 0 and column 0 are unused; diagonal entries are False.
    for a, b in H[1]:
        adjacency[a, b] = True
        adjacency[b, a] = True

    valid = np.ones(len(injections), dtype=bool)
    for a, b in J[1]:
        images_of_a = injections[:, a - 1]
        # images_of_a is a one-dimensional integer array of length len(injections).
        # The colon selects all rows; a-1 selects the column corresponding to vertex a.
        # If row r is phi_r, images_of_a[r]=phi_r(a), a vertex label of H.
        images_of_b = injections[:, b - 1]
        # Likewise, images_of_b is one-dimensional and images_of_b[r]=phi_r(b).
        # If two injections send (a,b) to (2,5) and (4,1), these arrays are
        # [2,4] and [5,1]; entries with the same index r belong to the same injection.
        # adjacency[images_of_a, images_of_b] checks the paired entries at equal indices,
        # giving a Boolean array whose r-th entry is adjacency[phi_r(a),phi_r(b)].
        # This entry records whether injection r preserves the current edge {a,b}.
        valid = valid & adjacency[images_of_a, images_of_b]
    # valid[r] is True exactly when injection r preserves every edge of J.
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
