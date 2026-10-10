"""
Stage 1: compute all c_{ij,J} from the definitions of the five U_i.

Target identity (F here denotes the paper's matrix mathsf F):
    F_{ij} = sum_J c_{ij,J} t(J,G),     0 <= i,j <= 4.

G is any finite nonempty triangle-free simple undirected graph, and A is its
adjacency matrix. Every E denotes independent uniform sampling with replacement:
    E_{a,b}[A_{za} A_{ab}] = (1/n^2) sum_{a,b in V(G)} A_{za} A_{ab}.
The values of a and b may coincide; the program imposes no condition a != b.

Reading order:
    1. Storage of monomials.
    2. Addition, scalar multiplication, and multiplication of polynomials.
    3. Coordinate definitions of the five U_i.
    4. Expansion of the complete F_{ij}.
    5. Simplification of adjacency products, ordinary graph isomorphism,
       and collection of coefficients.
    6. Computation of all c_{ij,J} and storage in one dictionary.

Run: python 01_expand_F.py
Dependencies: the Python standard library and NetworkX for graph isomorphism.
This script reads no previous coefficient file and does not compute t(J,H),
F_H, or the finite certificate verification.
"""

from fractions import Fraction
from pathlib import Path
import pickle
import networkx as nx

# ======================================================================
# 1. Storage of monomials
# ======================================================================
# A monomial stores exactly two entries:
#   coefficient: a rational coefficient, using Fraction to avoid rounding;
#   edges: indices of adjacency factors; each pair represents one A_{uv}.
#
# For example:
#   {"coefficient": Fraction(3, 50),
#    "edges": [("z", "a1"), ("a1", "x")]}
# This records the coefficient 3/50 and the factors A_{z,a1}, A_{a1,x}.
# In a complete F_{ij} monomial, average every distinct name independently
# and uniformly with replacement. One position has one name; values may coincide.
# The names x, y, z, t, a1, a2, ... all follow this same convention.
# Averaged variables absent from the product contribute 1 and can be removed.
# Call the defining function again when distinct summation positions need fresh names.
#
# A polynomial is a list of monomials, interpreted as their sum.
# The empty list represents zero. Duplicate terms are collected at the isomorphism step.
# For dictionaries, [] selects an entry, such as term["edges"]; for lists, a position.
# Python operations used in this file:
#   for item in items: process each element of items;
#   result.append(item): add item at the end of the list;
#   if condition: run the indented block only when the condition holds;
#   continue: skip the rest of this iteration and process the next element;
#   return value: finish the function and return value to the caller;
#   set: a collection without duplicates, suitable for undirected edges and vertices;
#   tuple: an immutable ordered collection that can serve as a dictionary key;
#   enumerate(items): provide each element together with its index, starting at 0.


def make_term(coefficient, edges):
    """Create a monomial, copying the input list to prevent later changes from affecting it."""
    return {
        "coefficient": Fraction(coefficient),
        "edges": list(edges),
    }


def adjacency_factor(first_vertex, second_vertex):
    """One factor A_{first_vertex,second_vertex}; use the averaging convention stated above."""
    term = make_term(1, [(first_vertex, second_vertex)])
    return [term]


# a1, a2, ... are symbolic variable names, not actual vertex labels in G.
# x, y, z, t, a1, a2, ... are all variable names.
# The counter produces a1, a2, ... in succession, with a fresh name at each call.
next_variable_number = 1


def new_variable_name():
    global next_variable_number
    name = "a" + str(next_variable_number)
    next_variable_number = next_variable_number + 1
    return name


# ======================================================================
# 2. Addition, scalar multiplication, and multiplication
# ======================================================================
# This section only expands by addition, scaling, and distributivity; it does not simplify.
# A polynomial is a monomial list, with its entries interpreted as a sum.
# Identical monomials remain separate, and terms with coefficient zero are retained.
# Repeated factors within an edge list are also retained at this stage.
# Adjacency simplification, removal of zero terms, and graph collection occur in section 5.
#
# If P = p_1 + ... + p_a and Q = q_1 + ... + q_b, then:
#   addition: P + Q is [p_1, ..., p_a, q_1, ..., q_b], with a+b terms;
#   scaling: cP is [c*p_1, ..., c*p_a], still with a terms;
#   multiplication: PQ = sum_{i=1}^a sum_{j=1}^b p_i*q_j, with ab terms.
# These counts refer to expanded list lengths, not to distinct simplified monomials.

def add_polynomials(*polynomials):
    """
    *polynomials accepts two, three, or more polynomials.
    Addition copies the terms in order and concatenates their lists.
    No terms are collected, so lists of lengths a and b produce exactly a+b terms.
    """
    result = []
    for polynomial in polynomials:
        for term in polynomial:
            copied_term = make_term(
                term["coefficient"], term["edges"]
            )
            result.append(copied_term)
    return result


def scale_polynomial(coefficient, polynomial):
    """
    Scale each coefficient by coefficient, leaving its factor list unchanged.
    No terms are collected or removed; even scaling by zero preserves the term count.
    """
    result = []
    for term in polynomial:
        new_coefficient = Fraction(coefficient) * term["coefficient"]
        new_term = make_term(
            new_coefficient, term["edges"]
        )
        result.append(new_term)
    return result


def multiply_polynomials(*polynomials):
    """
    *polynomials accepts any number of polynomials in one call; for example,
        multiply_polynomials(P, Q, R, S) represents P*Q*R*S.
    Use distributivity:
        (p_1 + ... + p_a)(q_1 + ... + q_b)
        = sum_{i=1}^a sum_{j=1}^b p_i*q_j.
    Every term on the left is multiplied by every term on the right, giving ab terms.
    Factors with a_1,...,a_r terms produce exactly a_1*...*a_r terms.
    Start with the constant polynomial 1 and multiply the inputs in order.
    At each multiplication, two for loops enumerate every pair of terms.
    Multiply coefficients and concatenate edge lists; retain repeated factors.
    Identical monomials and zero coefficients are retained without simplification.
    Factors requiring distinct positions have already been assigned distinct names.
    This function does not rename indices; equal indices mean the same position.
    """
    # The constant 1 has coefficient 1 and no adjacency factors; the empty product is 1.
    result = [make_term(1, [])]
    for polynomial in polynomials:
        new_result = []
        for left_term in result:
            for right_term in polynomial:
                new_coefficient = left_term["coefficient"] * right_term["coefficient"]
                new_edges = left_term["edges"] + right_term["edges"]
                new_term = make_term(new_coefficient, new_edges)
                new_result.append(new_term)
        result = new_result
    return result


# ======================================================================
# 3. Coordinate definitions of the five U_i
# ======================================================================
# Each function expands U_i^{xy}(z), using x, y, and its supplied coordinate z.
# Here z is a function argument: define_U0("t") gives U_0^{xy}(t).
#
# Use the expectation expressions for U_i(z) directly, without a degree abbreviation.
# Generate the required variable names a, b, ... and write the complete formula in one return.
# Write adjacency factors directly; no averaging function or averaging entry is stored.
# In the full F_{ij}, average all distinct names as specified at the start of this file.
# Additive terms may share a, b; factors use fresh names when distinct positions are required.
# No actual G or n is needed: the factor 1/n is already part of the definition of E_a.
# Every Fraction(p,q) is the exact rational number p/q.


def define_U0(z):
    # U_0(z) = E_a[A_za (A_ax - A_ay)] - (A_zx - A_zy)/50。
    a = new_variable_name()
    return add_polynomials(
        multiply_polynomials(
            adjacency_factor(z, a),
            add_polynomials(
                adjacency_factor(a, "x"),
                scale_polynomial(-1, adjacency_factor(a, "y")),
            ),
        ),
        scale_polynomial(
            Fraction(-1, 50),
            add_polynomials(
                adjacency_factor(z, "x"),
                scale_polynomial(-1, adjacency_factor(z, "y")),
            ),
        ),
    )


def define_U1(z):
    # U_1(z) = E_a[(A_xa-A_ya)(A_zx+A_zy)]。
    a = new_variable_name()
    return multiply_polynomials(
        add_polynomials(
            adjacency_factor("x", a),
            scale_polynomial(-1, adjacency_factor("y", a)),
        ),
        add_polynomials(
            adjacency_factor(z, "x"), adjacency_factor(z, "y"),
        ),
    )


def define_U2(z):
    # U_2(z) = E_a[(A_za-(A_xa+A_ya)/2)(A_zx-A_zy)]。
    a = new_variable_name()
    return multiply_polynomials(
        add_polynomials(
            adjacency_factor(z, a),
            scale_polynomial(
                Fraction(-1, 2),
                add_polynomials(
                    adjacency_factor("x", a), adjacency_factor("y", a),
                ),
            ),
        ),
        add_polynomials(
            adjacency_factor(z, "x"),
            scale_polynomial(-1, adjacency_factor(z, "y")),
        ),
    )


def define_U3(z):
    # U_3(z) = E_{a,b}[A_za A_ab (A_bx-A_by)]
    #          + (3/50) E_a[A_za (A_ax-A_ay)] - (A_zx-A_zy)/625。
    a = new_variable_name()
    b = new_variable_name()
    return add_polynomials(
        multiply_polynomials(
            adjacency_factor(z, a),
            adjacency_factor(a, b),
            add_polynomials(
                adjacency_factor(b, "x"),
                scale_polynomial(-1, adjacency_factor(b, "y")),
            ),
        ),
        scale_polynomial(
            Fraction(3, 50),
            multiply_polynomials(
                adjacency_factor(z, a),
                add_polynomials(
                    adjacency_factor(a, "x"),
                    scale_polynomial(-1, adjacency_factor(a, "y")),
                ),
            ),
        ),
        scale_polynomial(
            Fraction(-1, 625),
            add_polynomials(
                adjacency_factor(z, "x"),
                scale_polynomial(-1, adjacency_factor(z, "y")),
            ),
        ),
    )


def define_U4(z):
    # U_4(z) = E_{a,b}[(A_za-(A_xa+A_ya)/2)
    #                    (A_zb(A_bx-A_by)-(A_zx-A_zy)/50)]
    #          + (1/4) E_{a,b}[(A_xa-A_ya)(A_xb-A_yb)(A_zx-A_zy)]。
    a = new_variable_name()
    b = new_variable_name()
    return add_polynomials(
        multiply_polynomials(
            add_polynomials(
                adjacency_factor(z, a),
                scale_polynomial(
                    Fraction(-1, 2),
                    add_polynomials(
                        adjacency_factor("x", a), adjacency_factor("y", a),
                    ),
                ),
            ),
            add_polynomials(
                multiply_polynomials(
                    adjacency_factor(z, b),
                    add_polynomials(
                        adjacency_factor(b, "x"),
                        scale_polynomial(-1, adjacency_factor(b, "y")),
                    ),
                ),
                scale_polynomial(
                    Fraction(-1, 50),
                    add_polynomials(
                        adjacency_factor(z, "x"),
                        scale_polynomial(-1, adjacency_factor(z, "y")),
                    ),
                ),
            ),
        ),
        scale_polynomial(
            Fraction(1, 4),
            multiply_polynomials(
                add_polynomials(
                    adjacency_factor("x", a),
                    scale_polynomial(-1, adjacency_factor("y", a)),
                ),
                add_polynomials(
                    adjacency_factor("x", b),
                    scale_polynomial(-1, adjacency_factor("y", b)),
                ),
                add_polynomials(
                    adjacency_factor(z, "x"),
                    scale_polynomial(-1, adjacency_factor(z, "y")),
                ),
            ),
        ),
    )


def define_U(index, coordinate):
    """Associate the indices 0, 1, 2, 3, 4 with the five definitions above."""
    if index == 0:
        return define_U0(coordinate)
    if index == 1:
        return define_U1(coordinate)
    if index == 2:
        return define_U2(coordinate)
    if index == 3:
        return define_U3(coordinate)
    if index == 4:
        return define_U4(coordinate)
    raise ValueError("The index of U must be between 0 and 4.")


# ======================================================================
# 4. Expansion of the complete F_{ij}
# ======================================================================
# F_{ij} = E_{xyzt}[A_xy A_zt U_i(z) U_j(z)]
#        + E_{xyzt}[A_xy A_zt U_i(z) U_j(t)]
#        - (7/50) E_{xyz}[A_xy U_i(z) U_j(z)]。
# The first term contains E_t[A_zt], and both U factors use coordinate z.
# In the second term, t is the coordinate of U_j and the summed matrix column index.
# The third term does not involve t.
# One return writes the full formula as a single list containing all three terms.
# Only expansion occurs here; section 5 simplifies the list and collects graph coefficients.


def expand_F_entry(i, j):
    return add_polynomials(
        multiply_polynomials(
            adjacency_factor("x", "y"),
            adjacency_factor("z", "t"),
            define_U(i, "z"),
            define_U(j, "z"),
        ),
        multiply_polynomials(
            adjacency_factor("x", "y"),
            adjacency_factor("z", "t"),
            define_U(i, "z"),
            define_U(j, "t"),
        ),
        scale_polynomial(
            Fraction(-7, 50),
            multiply_polynomials(
                adjacency_factor("x", "y"),
                define_U(i, "z"),
                define_U(j, "z"),
            ),
        ),
    )


# ======================================================================
# 5. Simplify monomials, test graph isomorphism, and collect coefficients
# ======================================================================

def graph_from_averaged_term(term):
    """
    When the whole term is averaged over every variable, its factors define a
    simple undirected graph J, and the term equals coefficient * t(J,G).
    
    Rules:
    (a) A_uu=0, so a term with a loop is zero.
    (b) A_uv=A_vu and A_uv^m=A_uv for m>=1, so undirected edges are deduplicated.
    (c) A triangle product is zero because G is triangle-free.
    (d) Variables absent from every factor can be removed because E_a[1]=1.
    Different names mean distinct summation positions, not necessarily distinct values.
    """
    simplified_edges = set()
    for first_vertex, second_vertex in term["edges"]:
        if first_vertex == second_vertex:
            return None
        # Sorting gives the same pair for (u,v) and (v,u); set removes duplicates.
        ordered_edge = tuple(sorted([first_vertex, second_vertex]))
        simplified_edges.add(ordered_edge)

    # Retain only variables occurring in factors; absent averaged variables contribute 1.
    used_vertices = set()
    for first_vertex, second_vertex in simplified_edges:
        used_vertices.add(first_vertex)
        used_vertices.add(second_vertex)
    ordered_vertices = sorted(used_vertices)
    vertex_number = {}
    for index, vertex in enumerate(ordered_vertices):
        vertex_number[vertex] = index + 1

    # vertex_number is now a dictionary mapping each variable name to an integer label.
    # For example, if ordered_vertices = ["a1", "x", "z"], the loop above gives:
    #     vertex_number = {"a1": 1, "x": 2, "z": 3}
    # Thus vertex_number["x"] is 2, and the labels of edge ("z", "a1") are 3 and 1.
    # This only renumbers variable names; it does not assign actual vertices of G to them.

    graph = {"vertices": list(range(1, len(ordered_vertices) + 1)), "edges": set()}
    for first_vertex, second_vertex in simplified_edges:
        first_number = vertex_number[first_vertex]
        second_number = vertex_number[second_vertex]
        graph["edges"].add(tuple(sorted([first_number, second_number])))

    # Enumerate three distinct positions. If all three edges occur, the monomial is zero.
    number_of_vertices = len(graph["vertices"])
    for a in range(1, number_of_vertices + 1):
        for b in range(a + 1, number_of_vertices + 1):
            for c in range(b + 1, number_of_vertices + 1):
                if (a, b) in graph["edges"] and (a, c) in graph["edges"] and (b, c) in graph["edges"]:
                    return None

    return graph


def graphs_are_isomorphic(first_graph, second_graph):
    """
    Convert the two dictionaries with vertices and edges to simple NetworkX graphs.
    Then call nx.is_isomorphic, returning True for isomorphic graphs and False otherwise.
    Vertex labels list the vertices and edges; the isomorphism need not preserve them.
    """
    # nx.Graph() creates an empty simple undirected graph.
    # add_nodes_from adds vertices, and add_edges_from adds edges.
    # Add all vertices first so that isolated vertices are retained.
    first = nx.Graph()
    first.add_nodes_from(first_graph["vertices"])
    first.add_edges_from(first_graph["edges"])

    second = nx.Graph()
    second.add_nodes_from(second_graph["vertices"])
    second.add_edges_from(second_graph["edges"])

    return nx.is_isomorphic(first, second)


def collect_coefficients(polynomial):
    """
    Input an expanded monomial list; return {graph representative J: rational coefficient}.
    Representatives are constructed inside the function; the caller supplies no catalog.
    """
    # Dictionaries and lists cannot be dictionary keys, so encode each graph as a pair of tuples:
    #     J = (vertex tuple, edge tuple)
    # For example, V(J)={1,2,3} and E(J)={{1,2},{2,3}} give the key:
    #     J = ((1, 2, 3), ((1, 2), (2, 3)))
    # J[0] contains the vertices and J[1] the edges; coefficients remain Fraction objects.
    # For example, {J: Fraction(3, 50)} represents (3/50)*t(J,G).
    # NetworkX still tests isomorphism; equality of the tuples is not the isomorphism test.
    coefficients = {}
    for term in polynomial:
        graph = graph_from_averaged_term(term)
        if graph is None:
            continue

        # Look for an isomorphic representative already encountered in this calculation.
        representative = None
        for J in coefficients:
            if graphs_are_isomorphic(graph, {"vertices": J[0], "edges": J[1]}):
                representative = J
                break

        # If none exists, use this graph itself as a new dictionary key.
        if representative is None:
            representative = (tuple(graph["vertices"]), tuple(sorted(graph["edges"])))
            coefficients[representative] = Fraction(0)
        coefficients[representative] += term["coefficient"]

    # Remove graph classes whose coefficients cancel to zero, returning the final dictionary.
    result = {}
    for J in coefficients:
        if coefficients[J] != 0:
            result[J] = coefficients[J]
    return result


# ======================================================================
# 6. Compute all c_{ij,J} and save one dictionary
# ======================================================================
# The single output dictionary has the following structure:
#     c_ij_J = {
#         (0, 0): {J: Fraction(...), ...},
#         (0, 1): {J: Fraction(...), ...},
#         ...
#         (4, 4): {J: Fraction(...), ...},
#     }
# c_ij_J[(i, j)][J] equals c_{ij,J}; omitted graph classes have coefficient zero.
# Representatives are chosen separately for each (i,j), possibly with different vertex labels.
# pickle preserves the dictionary, tuples, and Fractions without assigning J1, J2, ... indices.


def main():
    global next_variable_number
    next_variable_number = 1

    c_ij_J = {}
    for i in range(5):
        for j in range(5):
            c_ij_J[(i, j)] = collect_coefficients(expand_F_entry(i, j))

    output_folder = Path(__file__).resolve().parent / "results" / "01"
    output_folder.mkdir(parents=True, exist_ok=True)
    with (output_folder / "c_ij_J.pkl").open("wb") as file:
        pickle.dump(c_ij_J, file)
    print("Saved all 25 coefficient dictionaries to " + str(output_folder / "c_ij_J.pkl"))


# Importing defines the functions; running this file directly starts the calculation.
if __name__ == "__main__":
    main()
