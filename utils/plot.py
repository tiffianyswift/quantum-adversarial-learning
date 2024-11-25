import matplotlib.pyplot as plt

def load_val(file_path):
    losses = []
    with open(file_path, 'r') as file:
        for line in file:
            # 将读取的行转换为浮点数并添加到列表中
            losses.append(float(line.strip()))
    return losses


def process_val(losses, window_size=10, stride=1):
    # 初始化处理后的损失值列表
    processed_losses = []

    # 计算滑动平均，考虑stride
    i = 0
    while i < len(losses)/8:
        # 如果当前索引小于窗口大小，则只取从开始到当前索引的平均
        if i < window_size:
            window_average = sum(losses[:i + 1]) / (i + 1)
        else:
            # 取当前索引往前窗口大小的元素进行平均
            window_average = sum(losses[i - window_size + 1:i + 1]) / window_size

        # 添加到处理后的列表中
        processed_losses.append(window_average)
        i += stride  # 移动窗口的步长

    return processed_losses


def plot_loss(filepath, window_size, stride):
    losses = process_val(load_val(filepath), window_size, stride)
    # 创建一个图形窗口
    plt.figure(figsize=(8, 4))

    # 绘制损失曲线
    plt.plot(losses, label='Loss', marker='o')  # marker是可选的，用于标示每个点

    # 添加标题和标签
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 添加图例
    plt.legend()

    # 显示网格（可选）
    plt.grid(True)

    # 显示图形
    plt.show()


if __name__ == '__main__':
    plot_loss("../mnist/mnist01/metric/val", 20, 20)
