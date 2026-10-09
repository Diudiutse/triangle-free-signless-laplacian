"""
第一步：由五个 U_i 的定义，计算全部 c_{ij,J}。

目标公式（这里 F 指论文中的矩阵 mathsf F）：
    F_{ij} = sum_J c_{ij,J} t(J,G),     0 <= i,j <= 4.

G 是任意有限、非空、无三角形的简单无向图，A 是其邻接矩阵。
所有 E 都是独立、均匀、有放回的平均。例如
    E_{a,b}[A_{za} A_{ab}] = (1/n^2) sum_{a,b in V(G)} A_{za} A_{ab}.
变量 a、b 可以取到同一个顶点；程序没有添加 a != b 的限制。

阅读顺序：
    1. 单项式如何存储；
    2. 多项式的加法、数乘和乘法；
    3. 五个 U_i 的坐标定义；
    4. 整个 F_{ij} 的展开；
    5. 化简邻接矩阵乘积，判别普通图同构，合并系数；
    6. 计算全部 c_{ij,J}，保存一个字典。

运行：python 01_expand_F.py
依赖：Python 标准库和 NetworkX；图同构判别调用 NetworkX。
本程序不读取旧的系数文件，不计算 t(J,H)、F_H 或 Lemma 8。
"""

from fractions import Fraction
from pathlib import Path
import pickle
import networkx as nx

# ======================================================================
# 1. 单项式如何存储
# ======================================================================
# 一个单项式只存两个项目：
#   coefficient：有理数系数，使用 Fraction，避免浮点误差；
#   edges：邻接矩阵因子的下标，每个二元组对应一个 A_{uv}；
#
# 例如：
#   {"coefficient": Fraction(3, 50),
#    "edges": [("z", "a1"), ("a1", "x")]}
# 这明确记录了系数 3/50 与两个因子 A_{z,a1}、A_{a1,x}。
# 在完整 F_{ij} 的单项式中，对乘积里所有不同名字独立均匀有放回平均。
# 同一个位置使用同一个名字，不同位置使用不同名字；取值允许相同。
# 名字 x、y、z、t、a1、a2、... 都按这个约定解释，不分成不同类别。
# 未出现在乘积中的平均变量贡献 1，可以删去。
# 需要不同求和位置的因子重新调用定义函数，用不同名字明确写出因子。
#
# 多项式用“单项式的列表”存储；列表中的项相加。
# 空列表表示零。重复的项暂时保留，到图同构步骤再合并。
# 字典的 [] 是取项目，如 term["edges"]；列表的 [] 是取某个位置。
# 本文件常用的 Python 操作：
#   for item in items：逐个处理 items 中的元素；
#   result.append(item)：把 item 加到列表末尾；
#   if condition：条件成立才执行缩进的代码；
#   continue：跳过本轮剩余代码，处理下一个元素；
#   return value：结束当前函数，把 value 交给调用者；
#   set：集合，不保留重复元素，适合存储无向边和已使用的顶点；
#   tuple：不可修改的有序组，可用作字典的键；
#   enumerate(items)：同时给出列表下标（从 0 开始）与该位置的元素。


def make_term(coefficient, edges):
    """建立一个单项式，并复制输入列表，防止之后修改它们时连带改动此项。"""
    return {
        "coefficient": Fraction(coefficient),
        "edges": list(edges),
    }


def adjacency_factor(first_vertex, second_vertex):
    """单个 A_{first_vertex,second_vertex} 因子；平均方式按文件开头的文字约定解释。"""
    term = make_term(1, [(first_vertex, second_vertex)])
    return [term]


# a1、a2、... 是符号变量名，不是 G 中的具体顶点编号。
# x、y、z、t、a1、a2、... 都是变量名。
# 计数器依次给出 a1、a2、...，每次返回的具体名字都不同。
next_variable_number = 1


def new_variable_name():
    global next_variable_number
    name = "a" + str(next_variable_number)
    next_variable_number = next_variable_number + 1
    return name


# ======================================================================
# 2. 加法、数乘和乘法
# ======================================================================
# 本节只按加法、数乘和分配律展开，不做化简。
# 一个多项式是一个单项式列表，列表中的各项表示相加。
# 因此，相同的单项式仍分别保留，系数为 0 的项也不删除；
# 因子列表中的重复因子也不在这里合并。
# 邻接因子的化简、零项的删除和同构图的系数合并留到第 5 节。
#
# 若 P = p_1 + ... + p_a，Q = q_1 + ... + q_b，则：
#   加法：P + Q 对应 [p_1, ..., p_a, q_1, ..., q_b]，共 a+b 项；
#   数乘：cP 对应 [c*p_1, ..., c*p_a]，仍为 a 项；
#   乘法：PQ = sum_{i=1}^a sum_{j=1}^b p_i*q_j，共 ab 项。
# 这里的项数是展开后列表的长度，不是化简后不同单项式的数量。

def add_polynomials(*polynomials):
    """
    *polynomials 表示可以传入两个、三个或更多多项式。
    加法只是依次复制各列表的项，把后一个列表接到前一个列表末尾。
    不合并同类项，所以两个列表分别有 a、b 项时，结果恰有 a+b 项。
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
    逐项数乘：每项的系数乘以 coefficient，因子列表保持原样。
    不合并或删除任何项；即使 coefficient 为 0，结果的项数也不变。
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
    *polynomials 表示可以一次传入任意多个多项式，例如：
        multiply_polynomials(P, Q, R, S) 表示 P*Q*R*S。
    使用分配律：
        (p_1 + ... + p_a)(q_1 + ... + q_b)
        = sum_{i=1}^a sum_{j=1}^b p_i*q_j。
    左边的每项分别与右边的每项相乘，a 项与 b 项生成恰好 ab 项。
    多个因子的项数分别为 a_1,...,a_r 时，结果恰有 a_1*...*a_r 项。
    从常数多项式 1 开始，按输入顺序逐个乘入因子。
    每乘入一个因子，两层 for 循环枚举当前结果与该因子的所有项对。
    单项式相乘时，系数相乘，因子列表拼接，不合并重复因子。
    即使生成了相同的单项式或系数为 0 的项，也分别保留，不做化简。
    需要不同位置的因子已经用不同名字构造。
    此函数不改任何下标；相同下标表示同一个求和位置。
    """
    # 1 的系数是 1，没有邻接因子。不传入多项式时，空乘积也返回 1。
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
# 3. 五个 U_i 的坐标定义
# ======================================================================
# 每个函数返回 U_i^{xy}(z) 的展开，因子中明确写出 x、y 和传入的 z。
# 以下代码中的 z 是函数参数，因此 define_U0("t") 就得到 U_0^{xy}(t)。
#
# 直接使用正文中 U_i(z) 的期望表示；下面不再引入 d(v) 的简写。
# 每个定义只先生成公式所需的 a、b 等具体变量名，再用一个 return 写完整式子。
# return 中直接写邻接因子，不加入求期望函数或求期望的数据项目。
# 在完整 F_{ij} 中，仍按文件开头的约定对所有不同名字求期望。
# 同一公式的不同相加项可以共用 a、b；相乘时公式指定不同位置才用不同名字。
# 不需要实际输入图 G 或 n：1/n 已包含在 E_a 的定义中。
# 所有 Fraction(p,q) 都表示精确有理数 p/q。


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
    """把编号 0、1、2、3、4 对应到上面五个定义。"""
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
    raise ValueError("U 的编号必须在 0 到 4 之间。")


# ======================================================================
# 4. 展开整个 F_{ij}
# ======================================================================
# F_{ij} = E_{xyzt}[A_xy A_zt U_i(z) U_j(z)]
#        + E_{xyzt}[A_xy A_zt U_i(z) U_j(t)]
#        - (7/50) E_{xyz}[A_xy U_i(z) U_j(z)]。
# 第一项含有因子 E_t[A_zt]，其中两个 U 都取 z 坐标。
# 第二项的 t 是 U_j 的坐标，也是矩阵乘法中求和的列下标。
# 第三项没有 t。
# 一个 return 写出整个公式，返回三项相加后的一个单项式列表。
# 本节仍然只展开；第 5 节统一化简这个列表并合并同构图的系数。


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
# 5. 化简每个单项式，判别图同构，合并系数
# ======================================================================

def graph_from_averaged_term(term):
    """
    若整项已经对所有变量平均，则把邻接因子变成一个普通简单图 J。
    该项等于 coefficient * t(J,G)。

    规则：
    (a) A_uu=0，所以有自环的项为零；
    (b) A_uv=A_vu，A_uv^m=A_uv (m>=1)，所以无向边去重；
    (c) 三角形乘积为零，因为 G 无三角形；
    (d) 不出现在任何因子中的平均变量可以删去，因为 E_a[1]=1。
    注意：不同名字是不同的求和位置，不表示它们必须取不同的顶点。
    """
    simplified_edges = set()
    for first_vertex, second_vertex in term["edges"]:
        if first_vertex == second_vertex:
            return None
        # 排序把 (u,v) 和 (v,u) 写成同一个二元组；set 自动去掉重复项。
        ordered_edge = tuple(sorted([first_vertex, second_vertex]))
        simplified_edges.add(ordered_edge)

    # 只保留确实出现于因子的变量；未出现的平均变量贡献因子 1。
    used_vertices = set()
    for first_vertex, second_vertex in simplified_edges:
        used_vertices.add(first_vertex)
        used_vertices.add(second_vertex)
    ordered_vertices = sorted(used_vertices)
    vertex_number = {}
    for index, vertex in enumerate(ordered_vertices):
        vertex_number[vertex] = index + 1

    # 此时 vertex_number 是“变量名 -> 整数编号”的字典。
    # 例如 ordered_vertices = ["a1", "x", "z"] 时，上面的循环得到：
    #     vertex_number = {"a1": 1, "x": 2, "z": 3}
    # 因而 vertex_number["x"] 等于 2；边 ("z", "a1") 的两个编号是 3、1。
    # 这里只给不同变量名重新编号，没有把它们赋值为 G 中的具体顶点。

    graph = {"vertices": list(range(1, len(ordered_vertices) + 1)), "edges": set()}
    for first_vertex, second_vertex in simplified_edges:
        first_number = vertex_number[first_vertex]
        second_number = vertex_number[second_vertex]
        graph["edges"].add(tuple(sorted([first_number, second_number])))

    # 枚举三个不同的求和位置。三条边同时存在则原单项式恒为零。
    number_of_vertices = len(graph["vertices"])
    for a in range(1, number_of_vertices + 1):
        for b in range(a + 1, number_of_vertices + 1):
            for c in range(b + 1, number_of_vertices + 1):
                if (a, b) in graph["edges"] and (a, c) in graph["edges"] and (b, c) in graph["edges"]:
                    return None

    return graph


def graphs_are_isomorphic(first_graph, second_graph):
    """
    把两个存有 vertices、edges 的字典转换成 NetworkX 的简单无向图，
    再调用 nx.is_isomorphic：同构返回 True，否则返回 False。
    顶点编号只用于列出顶点与边，不要求同构映射保持这些编号。
    """
    # nx.Graph() 建立空的简单无向图。
    # add_nodes_from 加入顶点，add_edges_from 加入边。
    # 先加入全部顶点，使没有相接边的顶点也被保留。
    first = nx.Graph()
    first.add_nodes_from(first_graph["vertices"])
    first.add_edges_from(first_graph["edges"])

    second = nx.Graph()
    second.add_nodes_from(second_graph["vertices"])
    second.add_edges_from(second_graph["edges"])

    return nx.is_isomorphic(first, second)


def collect_coefficients(polynomial):
    """
    输入一个展开后的单项式列表，输出 {图代表 J: 有理数系数}。
    图代表在函数内从单项式中产生，不需要调用者提供代表列表。
    """
    # 字典和列表不能作字典的键，因此把代表图写成两个 tuple 组成的二元组：
    #     J = (顶点的 tuple, 边的 tuple)
    # 例如 V(J)={1,2,3}、E(J)={{1,2},{2,3}} 时，键写成：
    #     J = ((1, 2, 3), ((1, 2), (2, 3)))
    # J[0] 是顶点，J[1] 是边；系数直接保留为 Fraction。
    # 例如结果 {J: Fraction(3, 50)} 表示 (3/50)*t(J,G)。
    # 同构判别仍由 NetworkX 完成，不用 tuple 的相等来判断图同构。
    coefficients = {}
    for term in polynomial:
        graph = graph_from_averaged_term(term)
        if graph is None:
            continue

        # 在本次计算已经出现的代表图中寻找同构者。
        representative = None
        for J in coefficients:
            if graphs_are_isomorphic(graph, {"vertices": J[0], "edges": J[1]}):
                representative = J
                break

        # 没找到同构代表，就把当前图本身作为字典的一个新键。
        if representative is None:
            representative = (tuple(graph["vertices"]), tuple(sorted(graph["edges"])))
            coefficients[representative] = Fraction(0)
        coefficients[representative] += term["coefficient"]

    # 删除相加抵消后系数为零的图，只返回最后的系数字典。
    result = {}
    for J in coefficients:
        if coefficients[J] != 0:
            result[J] = coefficients[J]
    return result


# ======================================================================
# 6. 计算全部 c_{ij,J}，保存一个字典
# ======================================================================
# 最终只有一个结果字典，其结构是：
#     c_ij_J = {
#         (0, 0): {J: Fraction(...), ...},
#         (0, 1): {J: Fraction(...), ...},
#         ...
#         (4, 4): {J: Fraction(...), ...},
#     }
# c_ij_J[(i, j)][J] 就是 c_{ij,J}；未列出的图同构类系数为零。
# 每个 (i,j) 独立选择同构类的代表；不同项可以选择不同编号的同构图。
# pickle 直接保存这个字典以及其中的 tuple 和 Fraction，不另编 J1、J2 等编号。


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


# 导入此文件时只加载函数；直接运行此文件时才开始计算。
if __name__ == "__main__":
    main()
