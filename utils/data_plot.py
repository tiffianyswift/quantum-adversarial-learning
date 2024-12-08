import matplotlib.pyplot as plt


# data shape (n, n)
def plot_mnist(data):
    """
    该函数绘制一个单独的 MNIST 图像
    :param data: 形状为 (n, n) 的二维数组或张量
    """
    plt.imshow(data, cmap='gray')  # 使用灰度色图显示图像
    plt.axis('off')  # 关闭坐标轴显示
    plt.show()


# data shape (batch_size, 1, n, n)
def plot_mnist_batch(data):
    """
    该函数绘制一个批次的 MNIST 图像
    :param data: 形状为 (batch_size, 1, n, n) 的张量或数组
    :param batch_size: 批次大小，即要绘制的图像数量
    """
    # 创建一个批次大小的子图网格
    rows = 1  # 行数
    cols = data.shape[0]  # 列数，确保网格适应批次大小
    batch_size = rows*cols

    # 创建一个合适大小的图像画布
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()

    # 循环绘制每张图像
    for i in range(batch_size):
        axes[i].imshow(data[i].squeeze(), cmap='gray')  # 去掉多余的维度并显示图像
        axes[i].axis('off')  # 关闭坐标轴显示

    plt.tight_layout()  # 自动调整子图间距
    plt.show()


# data shape (classes, batch_size, 1, n, n)
def plot_mnist_batch_classes(data):
    """
    该函数绘制一个批次的 MNIST 图像
    :param data: 形状为 (batch_size, 1, n, n) 的张量或数组
    :param batch_size: 批次大小，即要绘制的图像数量
    """
    # 创建一个批次大小的子图网格
    rows = data.shape[1]  # 行数
    cols = data.shape[0]  # 列数，确保网格适应批次大小
    batch_size = rows*cols

    # 创建一个合适大小的图像画布
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()

    data = data.reshape(batch_size, data.shape[-2], data.shape[-1])
    # 循环绘制每张图像
    for i in range(batch_size):
        axes[i].imshow(data[i].squeeze(), cmap='gray')  # 去掉多余的维度并显示图像
        axes[i].axis('off')  # 关闭坐标轴显示

    plt.tight_layout()  # 自动调整子图间距
    plt.show()
