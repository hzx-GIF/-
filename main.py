from common import build_model
from data_loader import *
from nn import *
from train import train

if __name__ == '__main__':
    EPOCHS = 1000
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01

    # 使用 get_network 来构建网络结构
    network = get_network()

    # 加载 CIFAR-10 数据集
    x_train, y_train, x_test, y_test = load_and_process_cifar10('./cifar-10-batches-py')

    # 创建模型并训练
    model = build_model(
        network=network,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        optimizer=SGD,
        loss_function=CrossEntropyLoss
    )

    # 开始训练
    train(model)
