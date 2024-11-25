from model_utils import train, evaluate


if __name__ == '__main__':
    mp = {
        # 模式配置
        "mode": "evaluate",

        # 公共参数
        "n_qubits": 8,
        "rep": 3,
        "ansatz_type": "real_amplitude",
        "observable_type": "two_classify",
        "encoding_type": "amplitude",
        "dataset_name": "mnist",
        "dataset_code": "default",
        "loss_func_name": "bce",

        # 训练配置
        "train_param_load_path": "param.npy",
        "param_dump_path": "param.npy",
        "optimizer_type": "adam",
        "lr": 0.01,
        "epoch": 1000,
        "n_train_samples": 128,
        "train_dataset_type": "train",
        "train_batch_size": 4,
        "train_shuffle": True,

        # 测试配置
        "test_param_load_path": "param.npy",
        "n_test_samples": 64,
        "test_dataset_type": "test",
        "test_batch_size": 4,
        "test_shuffle": True,
    }

    if mp['mode'] == 'train':
        train(mp)
    elif mp['mode'] == 'evaluate':
        evaluate(mp)
