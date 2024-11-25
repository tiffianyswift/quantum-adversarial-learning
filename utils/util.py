import numpy as np
import os
import re


def get_grad(model, x, lossgrad):
    org_param = model.param
    grad = np.zeros(org_param.shape)
    for i, param in enumerate(org_param):
        model.param[i] = param + np.pi / 2
        exps_right = model(x)

        model.param[i] = param - np.pi / 2
        exps_left = model(x)

        grad[i] = np.sum(lossgrad * (exps_right - exps_left) / 2)

        model.param[i] = param

    return grad


def save_val(val, dir_path="./metric", file_name="val"):
    if not os.path.exists(dir_path):
        # 文件夹不存在，创建文件夹
        os.makedirs(dir_path)
    with open(dir_path+"/"+file_name, 'a') as file:
        file.write(str(val) + '\n')


def set_model_param_from_file(model, file_path):
    # 编译正则表达式，匹配小括号内的数字（可能包含负号和小数点）
    pattern = re.compile(r'\((-?\d*\.?\d+(?:[eE][-+]?\d+)?)\)')
    with open(file_path, 'r') as file:
        content = file.read()  # 读取整个文件内容

    # 存储提取出的数字
    matches = pattern.findall(content)

    # 提取匹配的数字，因为 findall 会返回一个包含元组的列表，我们需要第一个元素
    numbers = [float(match) for match in matches]
    model.param = np.array(numbers)


def save_model(model, dir_path="./model_saved", file_name="circuit.qasm"):
    # save circuit
    object_circuit = model.ansatz.circuit
    param = model.param

    binded_circuit = object_circuit.assign_parameters(param)

    qasm_str = binded_circuit.qasm()

    if not os.path.exists(dir_path):
        # 文件夹不存在，创建文件夹
        os.makedirs(dir_path)
    # 将QASM字符串写入文件
    with open(dir_path+"/"+file_name, "w") as file:
        file.write(qasm_str)
    if not os.path.exists(dir_path+"/model_info"):
        with open(dir_path+"/model_info", 'w') as file:
            # save encoding
            file.write(model.encoding.__class__.__name__)
            # save observable
            file.write(model.observables.__class__.__name__)
    # save encoding
    # encoding 信息包含 使用的编码方式{AmplitudeEncoding(n_qubits)，AngleEncoding(n_features)}
    # save observable
    # observable 信息包含 使用的可观测量{getTwoClassfiyObservable(n_qubits)}

    # save loss
    # loss 信息包含 使用的损失函数

if __name__ == '__main__':
    # import re
    #
    # text = "ry(-0.2725) q[0]; ry(-0.417229) q[1]; ry(-0.21928) q[2];"
    #
    # # 使用正则表达式匹配括号内的数字
    # pattern = re.compile(r'\((-?\d+(\.\d+)?)\)')
    # matches = pattern.findall(text)
    #
    # # 提取匹配的数字，因为 findall 会返回一个包含元组的列表，我们需要第一个元素
    # numbers = [float(match[0]) for match in matches]
    #
    # print(numbers)

    print(set_model_param_from_file("../mnist/mnist01/model_saved/saved9128"))




