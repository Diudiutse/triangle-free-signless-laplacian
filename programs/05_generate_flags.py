"""
Stage 5: for each sigma, generate graph isomorphism classes fixing the
first k vertex labels. The flag order is ell=(8+k)//2, and the first k vertices
must induce exactly sigma.

Read results/04/sigma_list.pkl and write results/05/F_sigma.pkl.
The output is {sigma: graph list}; each graph is (vertex tuple, edge tuple).
Test isomorphism by permuting only the unlabeled vertices, without an external
graph isomorphism library.
Run: python 05_generate_flags.py
"""

from importlib import import_module
from itertools import permutations
from pathlib import Path
import pickle


# Reuse the subset enumeration from stage 2; importing does not execute its main function.
all_subsets = import_module("02_generate_tau8").all_subsets


def labeled_graphs_are_isomorphic(first_graph, second_graph, k):
    """For graphs on 1,...,n, test for an isomorphism fixing each of 1,...,k."""
    n = len(first_graph[0])
    if n != len(second_graph[0]):
        return False
    if not 0 <= k <= n:
        raise ValueError("The label count k must lie between 0 and the graph order.")

    first_edges = set(first_graph[1])
    second_edges = set(second_graph[1])
    for free_images in permutations(range(k + 1, n + 1)):
        # phi[a] is the image of vertex a; entry 0 is unused.
        # The first k images are fixed; the remaining images are given by this permutation.
        phi = list(range(k + 1)) + list(free_images)
        preserves_adjacency = True
        for a in range(1, n + 1):
            for b in range(a + 1, n + 1):
                first_has_edge = (a, b) in first_edges # This Boolean value records whether (a,b) belongs to first_edges.
                second_has_edge = tuple(sorted([phi[a], phi[b]])) in second_edges
                # Check both edges and nonedges: the adjacency relations must agree exactly.
                if first_has_edge != second_has_edge:
                    preserves_adjacency = False
                    break
            if not preserves_adjacency:
                break
        if preserves_adjacency:
            return True
    return False


def generate_sigma_flags(sigma):
    """Add unlabeled vertices to sigma and return every ell-vertex sigma-flag representative."""
    k = len(sigma[0])
    ell = (8 + k) // 2
    graphs = [sigma]
    for n in range(k + 1, ell + 1):
        new_graphs = []
        for graph in graphs:
            for neighbors in all_subsets(graph[0]):
                # The neighbors of new vertex n must form an independent set, to avoid triangles.
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
