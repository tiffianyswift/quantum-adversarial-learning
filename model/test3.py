import numpy as np

if __name__ == '__main__':
    # arr = np.array([1/2+1/(2*np.sqrt(2)), -1/(2*np.sqrt(2)), 1/2-1/(2*np.sqrt(2)), 1/(2*np.sqrt(2))])
    #
    #
    # # 计算欧几里得范数（L2范数）
    # norm = np.linalg.norm(arr)
    #
    # a = np.array([1/2+1/(2*np.sqrt(2)), -1/(2*np.sqrt(2)), -1/2+1/(2*np.sqrt(2)), -1/(2*np.sqrt(2))])
    # b = np.array([1/2+1/(2*np.sqrt(2)), -1/(2*np.sqrt(2)), 1/2-1/(2*np.sqrt(2)), 1/(2*np.sqrt(2))])
    #
    # # 使用 numpy.dot 函数计算内积
    # dot_product = np.dot(a, b)
    #
    # print(norm)
    # print(dot_product)
    from copy import deepcopy

    a = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    b = np.array([1/np.sqrt(2) + 0.1, 1/np.sqrt(2)])

    norm = np.linalg.norm(b)

    b_norm = b/norm
    b_norm_z = deepcopy(b_norm)
    b_norm_z[1] = -b_norm_z[1]
    print(b_norm_z)
    print(b_norm)

    print(np.dot(b_norm_z, b_norm))

    a = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    b = np.array([1 / np.sqrt(2) - 0.1, 1 / np.sqrt(2)])

    norm = np.linalg.norm(b)

    b_norm = b / norm
    b_norm_z = deepcopy(b_norm)
    b_norm_z[1] = -b_norm_z[1]
    print(b_norm_z)
    print(b_norm)

    print(np.dot(b_norm_z, b_norm))


    # # 将向量归一化为单位向量
    # unit_vector = arr / norm
    #
    # print(unit_vector*unit_vector)