"""
Stage 2: generate tau_8, the isomorphism classes of eight-vertex
triangle-free simple undirected graphs.

Start with the graph on no vertices and add one vertex at a time.
Its neighbors must form an independent set in the previous graph, to avoid triangles.
Enumerate all such neighbor sets, then use NetworkX isomorphism to retain one
representative of each class.

Completeness: deleting a vertex from any triangle-free graph leaves a triangle-free
graph. When that vertex is restored, its neighbors form an independent set and hence
occur in the enumeration. Induction up to eight vertices gives the complete catalog.

Write results/02/tau_8.pkl, containing a graph list.
As in stage 1, each graph is H=(vertex tuple, edge tuple). Examples:
    ((1,2,3,4,5,6,7,8), ())        # eight isolated vertices
    ((1,2,3,4,5,6,7,8), ((1,8),)) # one edge and six isolated vertices
Labels describe the chosen representative; isomorphisms need not preserve them.

Generate the graphs independently, without an old catalog or stage 1 coefficients.
Dependency: NetworkX. Run: python 02_generate_tau8.py
"""

from pathlib import Path
import pickle
import networkx as nx


def all_subsets(vertices):
    """List every subset of vertices, including the empty set, with each subset a tuple."""
    subsets = [()]
    for vertex in vertices:
        # Each existing subset has two choices: exclude vertex or include vertex.
        # The old list excludes it; the new list below includes it.
        subsets_with_vertex = []
        for subset in subsets:
            subsets_with_vertex.append(subset + (vertex,))
        subsets = subsets + subsets_with_vertex
    # For vertices=[1,2], return [(), (1,), (2,), (1,2)].
    return subsets


def is_independent(graph, vertices):
    """Check that no pair of vertices is adjacent in graph."""
    for first_vertex, second_vertex in graph.edges:
        if first_vertex in vertices and second_vertex in vertices:
            return False
    return True


def generate_triangle_free_graphs(number_of_vertices):
    """Return all triangle-free graphs of the specified order, one per isomorphism class."""
    # graphs holds representatives of the current order and is updated to order n at each stage.
    # Induction starts with the unique graph on no vertices, the empty graph.
    graphs = [nx.Graph()]
    for n in range(1, number_of_vertices + 1):
        new_graphs = []
        # Generate new graphs by adding a vertex to each graph on n-1 vertices.
        for graph in graphs:
            for neighbors in all_subsets(graph.nodes): # Consider every possible neighbor set of the new vertex in the previous graph.
                if not is_independent(graph, neighbors):# Discard sets that are not independent in the previous graph.
                    continue

                # graph has vertices 1,...,n-1; add vertex n and connect it to all its neighbors.
                # Add vertex n even if neighbors is empty, retaining isolated vertices.
                new_graph = graph.copy()
                new_graph.add_node(n)
                for vertex in neighbors:
                    new_graph.add_edge(vertex, n)

                # Compare with the current representatives; do not retain an isomorphic duplicate.
                already_present = False
                for representative in new_graphs:
                    if nx.is_isomorphic(new_graph, representative):
                        already_present = True
                        break
                if not already_present:
                    new_graphs.append(new_graph)

        graphs = new_graphs
        print(str(n) + " vertices: " + str(len(graphs)) + " isomorphism classes.", flush=True)

    # Convert NetworkX graphs to (vertex tuple, edge tuple) for subsequent scripts.
    result = []
    for graph in graphs:
        edges = []
        for first_vertex, second_vertex in graph.edges:
            edges.append(tuple(sorted([first_vertex, second_vertex])))
        edges.sort()
        result.append((tuple(sorted(graph.nodes)), tuple(edges)))
    return result


def main():
    tau_8 = generate_triangle_free_graphs(8)
    output_folder = Path(__file__).resolve().parent / "results" / "02"
    output_folder.mkdir(parents=True, exist_ok=True)
    with (output_folder / "tau_8.pkl").open("wb") as file:
        pickle.dump(tau_8, file)
    print("Saved " + str(len(tau_8)) + " graphs to " + str(output_folder / "tau_8.pkl"))


if __name__ == "__main__":
    main()
