"""
Stage 6: compute every C_H^sigma and save {(sigma,H): matrix}.
matrix[p][p_prime] corresponds to F_sigma[sigma][p] and F_sigma[sigma][p_prime].
Store matrices as lists of row lists, with Python integer entries.

compute_C_matrix(H,sigma,flag_list) counts the whole matrix in one pass rather
than repeating the enumeration for each entry. Fix ordered roots and choose two
disjoint sets of unlabeled vertices. Each qualifying choice contributes
(m!)^2 * (8-k-2m)!, where m=ell-k. This counts the orders within both sets and
within the unused positions, and hence counts all the full bijections.

Read only the pickle outputs of stages 2 and 5, not previous C_H^sigma values.
Write results/06/C.pkl. Run: python 06_compute_C.py
"""

from functools import cache, lru_cache
from importlib import import_module
from itertools import combinations, permutations
from math import factorial
from pathlib import Path
import pickle


labeled_graphs_are_isomorphic = import_module("05_generate_flags").labeled_graphs_are_isomorphic


def induced_labeled_graph(H, vertex_order):
    """Induce a subgraph of H and relabel the a-th vertex in vertex_order as a, starting at 1."""
    # vertex_order is a tuple of distinct vertices of H.
    # Roots come first, so the first k relabeled vertices are the fixed root labels.
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
    """Return {induced root graph sigma: ordered root tuples}, including all distinct k-tuples."""
    # Types of the same order share root enumeration; retain only seven recent cached results.
    root_orders = {}
    for roots in permutations(H[0], k):
        sigma = induced_labeled_graph(H, roots)
        if sigma not in root_orders:
            root_orders[sigma] = []
        root_orders[sigma].append(roots)
    return root_orders


@cache
def find_flag_index(graph, flag_list, k):
    """Return the flag index of graph in flag_list; a tuple catalog allows caching."""
    # When the same labeled graph occurs again, the cache returns its previous index.
    for p, representative in enumerate(flag_list):
        if labeled_graphs_are_isomorphic(graph, representative, k):
            return p
    raise ValueError("The induced flag has no isomorphic representative in the stage 5 flag catalog.")


def compute_C_matrix(H, sigma, flag_list):
    """Return the whole C_H^sigma matrix in the row and column order of flag_list."""
    if len(H[0]) != 8:
        raise ValueError("H must have eight vertices.")
    k = len(sigma[0])
    ell = (8 + k) // 2
    m = ell - k
    flag_list = tuple(flag_list)
    matrix = []
    for p in range(len(flag_list)):
        matrix.append([0] * len(flag_list))

    # Each free group has m! orderings, and unused vertices have (8-k-2m)! orderings.
    # Roots are already ordered; do not multiply by k!. Each full bijection is counted once.
    multiplicity = factorial(m) ** 2 * factorial(8 - k - 2 * m)
    for roots in root_orders_by_sigma(H, k).get(sigma, []):
    # roots runs over every ordered k-tuple in V(H) inducing the root type sigma.
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
