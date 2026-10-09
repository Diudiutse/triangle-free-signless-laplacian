"""
判别 X_star 和各 X_sigma 是否满足 (39)：
    q + X[i][i] > sum_{j != i} abs(X[i][j])，对每一行 i 成立。
q 是导入的固定整数，所以单次判别只需要输入矩阵 X。
运行：python verify_39.py
"""

from spectral_data import q, X_star
from sigma_data import sigma_data


def satisfies_39(X):
    """输入矩阵 X；满足 (39) 返回 True，否则返回 False。"""
    for i in range(len(X)):
        off_diagonal_sum = 0
        for j in range(len(X)):
            if j != i:
                off_diagonal_sum = off_diagonal_sum + abs(X[i][j])
        if q + X[i][i] <= off_diagonal_sum:
            return False
    return True


def main():
    if not satisfies_39(X_star):
        print("X_star does not satisfy (39).")
        return
    for sigma, data in sigma_data.items():
        if not satisfies_39(data["X_sigma"]):
            print("X_sigma does not satisfy (39), sigma = " + repr(sigma))
            return
    print("X_star and all X_sigma satisfy (39).")


if __name__ == "__main__":
    main()
