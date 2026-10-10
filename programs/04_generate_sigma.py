"""
Stage 4: generate every type sigma, a triangle-free graph representative
on 0,...,6 vertices. Reuse generate_triangle_free_graphs from 02_generate_tau8.py.

Write results/04/sigma_list.pkl, containing a graph list.
Each sigma=(vertex tuple, edge tuple), the same format used for J and H.
Order the list by increasing graph order, including the empty graph ((), ()).
After choosing a representative, its vertices 1,...,k become fixed root labels.
Run: python 04_generate_sigma.py
"""

from importlib import import_module
from pathlib import Path
import pickle


# The filename starts with digits, so ordinary from 02_generate_tau8 import ... is invalid.
# import_module loads its generating function without executing its main function.
generate_triangle_free_graphs = import_module("02_generate_tau8").generate_triangle_free_graphs


def main():
    sigma_list = []
    for k in range(7):
        sigma_list = sigma_list + generate_triangle_free_graphs(k)

    output_folder = Path(__file__).resolve().parent / "results" / "04"
    output_folder.mkdir(parents=True, exist_ok=True)
    with (output_folder / "sigma_list.pkl").open("wb") as file:
        pickle.dump(sigma_list, file)
    print("Saved " + str(len(sigma_list)) + " types to " + str(output_folder / "sigma_list.pkl"))


if __name__ == "__main__":
    main()
