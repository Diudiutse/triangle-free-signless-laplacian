"""
第二步：生成 tau_8，即所有八顶点无三角形简单无向图的同构类代表。

从零顶点空图开始，每次增加一个顶点。
新顶点的邻居必须在原图中构成独立集，否则会形成三角形。
枚举所有这样的邻居集合，再用 NetworkX 判别同构，每类只保留一张图。

为什么不会漏图：任意 n 顶点无三角形图删除一个顶点后，仍是无三角形图；
把删去的顶点加回来时，其邻居必为独立集，因而会出现在本程序的枚举中。
从零顶点图归纳到八顶点，就得到 tau_8 的全部代表。

输出 results/02/tau_8.pkl，其内容只是一个图列表。
与第一步的图代表使用同样的格式：H=(顶点 tuple, 边 tuple)。例如：
    ((1,2,3,4,5,6,7,8), ())                  # 八个孤立点
    ((1,2,3,4,5,6,7,8), ((1,8),))           # 一条边和六个孤立点
顶点编号只表示所选代表的编号；同构判别不要求保持编号。

本程序独立生成图，不读取旧图目录，也不依赖第一步的系数文件。
依赖：已安装的 NetworkX。运行：python 02_generate_tau8.py
"""

from pathlib import Path
import pickle
import networkx as nx


def all_subsets(vertices):
    """列出顶点集的所有子集，包括空集；每个子集用 tuple 表示。"""
    subsets = [()]
    for vertex in vertices:
        # 已列出的每个子集有两种选择：不加入 vertex，或加入 vertex。
        # 原来的列表对应不加入；下面的新列表对应加入。
        subsets_with_vertex = []
        for subset in subsets:
            subsets_with_vertex.append(subset + (vertex,))
        subsets = subsets + subsets_with_vertex
    # 例如 vertices=[1,2] 时，返回 [(), (1,), (2,), (1,2)]。
    return subsets


def is_independent(graph, vertices):
    """检查 vertices 中是否没有任何一对顶点在 graph 中相邻。"""
    for first_vertex, second_vertex in graph.edges:
        if first_vertex in vertices and second_vertex in vertices:
            return False
    return True


def generate_triangle_free_graphs(number_of_vertices):
    """返回指定顶点数的全部无三角形图代表，每个同构类恰保留一个。"""
    # graphs用来存储固定order的所有无三角形图的同构类，随着n增大，graphs被同步更新为n顶点的所有无三角形图的同构类
    # 归纳起点：零顶点图只有一个，就是空图。
    graphs = [nx.Graph()]
    for n in range(1, number_of_vertices + 1):
        new_graphs = []
        #用往n-1顶点的图中加入新顶点的方式生成新图
        for graph in graphs:
            for neighbors in all_subsets(graph.nodes): #关于新加入的点与原先的图的哪些点相邻做讨论
                if not is_independent(graph, neighbors):#如果新加入的点在原图的邻域不是独立集，则排除
                    continue

                # graph 有顶点 1,...,n-1；加入新顶点 n，再连接其全部邻居。
                # 即使 neighbors 为空，也加入顶点 n，保留孤立点。
                new_graph = graph.copy()
                new_graph.add_node(n)
                for vertex in neighbors:
                    new_graph.add_edge(vertex, n)

                # 与本轮已有代表逐个比较。已有同构代表，就不重复保存。
                already_present = False
                for representative in new_graphs:
                    if nx.is_isomorphic(new_graph, representative):
                        already_present = True
                        break
                if not already_present:
                    new_graphs.append(new_graph)

        graphs = new_graphs
        print(str(n) + " vertices: " + str(len(graphs)) + " isomorphism classes.", flush=True)

    # 把 NetworkX 图转成明确的 (顶点 tuple, 边 tuple)，方便后续程序直接读取。
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
