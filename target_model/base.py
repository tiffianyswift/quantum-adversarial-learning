import torch


class BaseTargetModel:
    def __init__(self, model, loss_func, optimizer=None):
        # 默认属性
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer

    def get_model_output(self, x):
        return self.model(x)

    def get_input_grad(self, x, target):
        batch_size = x.shape[0]

        output = torch.tensor(self.model(x), dtype=torch.float32, requires_grad=True)
        loss = self.loss_func(output, target.float())
        loss.backward()

        loss_grad = output.grad.reshape(batch_size, 1)
        encoding_grad = torch.tensor(self.model.evaluate_encoding_gradient(x))

        return loss_grad * encoding_grad

    def predict(self, x):
        """
        默认的预测方法
        :param x:
        :return: 预测结果
        """
        pass

