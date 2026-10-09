"""
把原始固定整数写成两个 Python 数据文件：
  spectral_data.py：L、q、Delta、X_star；
  sigma_data.py：{sigma: {"V_sigma": ..., "X_sigma": ..., "a_sigma": ...}}。

sigma 使用第四、五步生成的图代表，V_sigma 的第 p 行对应 F_sigma[sigma][p]。
原始类型的根标签和旗顺序先对应到新列表，再重排 V_sigma 的行。
X_sigma 的指标对应 V_sigma 的列，列顺序保持，因此 X_sigma 不重排。
不复制旗列表或 C_H_sigma；不读取原文件中的核验结论。
运行：python prepare_readable_data.py
"""

from pathlib import Path
import hashlib
import json
import pickle
import sys
from importlib import import_module
from itertools import permutations

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

labeled_graphs_are_isomorphic = import_module("05_generate_flags").labeled_graphs_are_isomorphic

ROOT = Path(__file__).resolve().parent
OLD_DATA = ROOT.parent / "arxiv" / "anc"


def decode_edge_set(code, number_of_vertices):
    """旧文件的位编码转成显式边集；新文件中的顶点编号从 1 开始。"""
    edges = []
    bit = 0
    for a in range(number_of_vertices):
        for b in range(a + 1, number_of_vertices):
            if (code >> bit) & 1:
                edges.append([a + 1, b + 1])
            bit = bit + 1
    return edges


def integer_expression(value, indentation):
    """
    普通整数直接写数字。很长的整数用 int("..." "...") 分行写。
    Python 将括号内相邻的字符串连接为一个字符串，int 再转成整数。
    这样既保存完整十进制数值，也避免一行超过一万字符。
    """
    digits = str(value)
    if len(digits) <= 100:
        return digits
    lines = ["int("]
    for start in range(0, len(digits), 100):
        lines.append(indentation + "    " + repr(digits[start:start + 100]))
    lines.append(indentation + ")")
    return "\n".join(lines)


def symmetric_matrix_expression(dimension, corrections, indentation):
    lines = ["symmetric_integer_matrix(" + str(dimension) + ", ["]
    for correction in corrections:
        i = correction["i"]
        j = correction["j"]
        value = integer_expression(correction["numerator"], indentation + "    ")
        lines.append(indentation + "    (" + str(i) + ", " + str(j) + ", " + value + "),")
    lines.append(indentation + "])")
    return "\n".join(lines)


MATRIX_READER = '''
def symmetric_integer_matrix(dimension, upper_entries):
    """上三角中未列出的位置为零；把每个 (i,j,value) 同时填到 (i,j) 与 (j,i)。"""
    matrix = []
    for i in range(dimension):
        row = []
        for j in range(dimension):
            row.append(0)
        matrix.append(row)
    for i, j, value in upper_entries:
        matrix[i][j] = value
        matrix[j][i] = value
    return matrix

'''


def find_type_relabeling(old_sigma, sigma_list):
    """返回新 sigma 和根重编号 phi；phi[i] 是原根 i 的新标签。"""
    k = len(old_sigma[0])
    for sigma in sigma_list:
        if len(sigma[0]) != k:
            continue
        for images in permutations(range(1, k + 1)):
            phi = (0,) + images
            edges = []
            for a, b in old_sigma[1]:
                edges.append(tuple(sorted([phi[a], phi[b]])))
            if tuple(sorted(edges)) == sigma[1]:
                return sigma, phi
    raise ValueError("原始类型在生成的 sigma 列表中没有同构代表。")


def main():
    certificate = json.loads((OLD_DATA / "filtered-sharp-exact-certificate.json").read_text(encoding="utf-8"))
    metadata = json.loads((OLD_DATA / "filtered-eight-data.json").read_text(encoding="utf-8"))
    with (ROOT / "results" / "05" / "F_sigma.pkl").open("rb") as file:
        F_sigma = pickle.load(file)

    corrections_by_name = {}
    for correction in certificate["corrections"]:
        name = correction["key"]
        if name not in corrections_by_name:
            corrections_by_name[name] = []
        corrections_by_name[name].append(correction)
    factors_by_name = {group["key"]: group for group in certificate["groups"]}

    spectral_lines = [
        '"""固定整数 L、q、Delta、X_star；矩阵指标为 0,...,4。"""',
        "import sys",
        "# 固定整数有一万多位，允许把其十进制字符串转为 Python 整数。",
        "if hasattr(sys, 'set_int_max_str_digits'):",
        "    sys.set_int_max_str_digits(0)",
        MATRIX_READER,
        "L = [",
        *["    " + repr([int(value) for value in row]) + "," for row in certificate["spectral_L_integer"]],
        "]",
        "", "q = " + integer_expression(certificate["correction_denominator"], ""),
        "", "Delta = " + str(certificate["global_source_denominator"]),
        "", "X_star = " + symmetric_matrix_expression(5, corrections_by_name.get("spectral", []), ""),
        "",
    ]
    (ROOT / "spectral_data.py").write_text("\n".join(spectral_lines), encoding="utf-8")

    # 每个新类型只对应一个原始类型；仅存重排好的固定数值。
    matched_data = {}
    for group in metadata["groups"]:
        k = group["k"]
        old_sigma = (tuple(range(1, k + 1)), tuple(tuple(edge) for edge in decode_edge_set(group["type_code"], k)))
        sigma, phi = find_type_relabeling(old_sigma, list(F_sigma))
        factor = factors_by_name[group["key"]]
        old_V = [[int(value) for value in row] for row in factor["V_integer"]]
        new_V = [None] * len(F_sigma[sigma])
        for old_p, code in enumerate(group["flag_codes"]):
            edges = []
            for a, b in decode_edge_set(code, group["l"]):
                a = phi[a] if a <= k else a
                b = phi[b] if b <= k else b
                edges.append(tuple(sorted([a, b])))
            flag = (tuple(range(1, group["l"] + 1)), tuple(sorted(edges)))
            for p, representative in enumerate(F_sigma[sigma]):
                if labeled_graphs_are_isomorphic(flag, representative, k):
                    if new_V[p] is not None:
                        raise ValueError("两个原始旗对应到同一个新旗。")
                    new_V[p] = old_V[old_p]
                    break
            else:
                raise ValueError("原始旗在新旗列表中没有标签同构代表。")
        if any(row is None for row in new_V):
            raise ValueError("存在没有原始数值行的新旗。")
        if sigma in matched_data:
            raise ValueError("两个原始类型对应到同一个新类型。")
        matched_data[sigma] = (group["key"], new_V, int(factor["denominator"]))
    if set(matched_data) != set(F_sigma):
        raise ValueError("固定数值的类型与第五步的类型不一致。")

    with (ROOT / "sigma_data.py").open("w", encoding="utf-8") as output:
        output.write('"""固定数据 sigma_data={sigma: {V_sigma, X_sigma, a_sigma}}。\n'
                     'sigma 是生成的图 tuple；V_sigma 第 p 行对应 F_sigma[sigma][p]。\n'
                     'X_sigma 的行列对应 V_sigma 的列，a_sigma 为正整数。\n'
                     '不保存旗列表、图列表或 C_H_sigma。"""\n'
                     'from spectral_data import symmetric_integer_matrix\n\n'
                     'sigma_data = {\n')
        for sigma in F_sigma:
            name, V_sigma, a_sigma = matched_data[sigma]
            output.write("    " + repr(sigma) + ": {\n        'V_sigma': [\n")
            for row in V_sigma:
                output.write("            " + repr(row) + ",\n")
            output.write("        ],\n        'X_sigma': " + symmetric_matrix_expression(len(V_sigma[0]), corrections_by_name.get(name, []), "        ") + ",\n")
            output.write("        'a_sigma': " + str(a_sigma) + ",\n    },\n")
        output.write("}\n")

    report = {
        "operation": "two fixed-data files; type labels and V rows matched to generated flags",
        "input_sha256": {name: hashlib.sha256((OLD_DATA / name).read_bytes()).hexdigest() for name in ["filtered-sharp-exact-certificate.json", "filtered-eight-data.json"]},
        "F_sigma_sha256": hashlib.sha256((ROOT / "results/05/F_sigma.pkl").read_bytes()).hexdigest(),
        "generated_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ["spectral_data.py", "sigma_data.py"]},
        "types": len(matched_data),
    }
    (ROOT / "results" / "data_conversion.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Saved L,q,X_star and fixed data for " + str(len(matched_data)) + " types in two Python files.")


if __name__ == "__main__":
    main()
