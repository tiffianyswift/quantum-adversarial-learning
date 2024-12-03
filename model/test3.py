import numpy as np

if __name__ == '__main__':
    arr = np.array([0.6, 0.5, 0.5, 0.5])

    # 计算欧几里得范数（L2范数）
    norm = np.linalg.norm(arr)

    # 将向量归一化为单位向量
    unit_vector = arr / norm

    print(unit_vector*unit_vector)