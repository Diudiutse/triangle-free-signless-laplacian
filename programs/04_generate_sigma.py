"""
第四步：生成所有类型 sigma，即 0,...,6 顶点无三角形图的同构类代表。
直接复用 02_generate_tau8.py 中的 generate_triangle_free_graphs。

输出 results/04/sigma_list.pkl，内容只是一个图列表。
每张 sigma=(顶点 tuple, 边 tuple)，与前面的 J、H 使用相同格式。
列表按顶点数从小到大排列，包括零顶点空图 ((), ())。
选出代表之后，顶点 1,...,k 就作为该类型的固定根标签。
运行：python 04_generate_sigma.py
"""

from importlib import import_module
from pathlib import Path
import pickle


# 文件名以数字开头，不能写普通的 from 02_generate_tau8 import ...。
# import_module 导入该文件，并取出它已经定义的生成函数；不会执行其 main。
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
