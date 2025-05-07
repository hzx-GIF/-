
import numpy as np

# 训练函数

def train(model, optimizer, loss_function, epochs, batch_size, x_train, y_train, x_test, y_test):
    num_batches = len(x_train) // batch_size

    # 用于记录每一轮的训练损失、验证损失和验证准确率
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_accuracy = 0
    best_model_state = None

    for epoch in range(epochs):
        # 初始化本轮的损失和正确预测数量
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0

        # 训练过程，按批次处理数据
        for batch_idx in range(num_batches):
            # 计算当前批次的起始和结束索引
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            # 提取当前批次的样本和标签
            x_batch = x_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]

            # 前向传播计算输出
            output = model.forward(x_batch)
            # 计算损失
            loss = loss_function.forward(output, y_batch)
            epoch_loss += loss

            # 反向传播计算梯度
            gradients = loss_function.backward()

            # 设置模型为训练模式
            model.set_mode(True)

            # 反向传播更新各层梯度
            for layer in reversed(model.layers):
                if hasattr(layer, "backward"):
                    gradients = layer.backward(gradients)
                    if isinstance(gradients, tuple):
                        gradients = gradients[0]

            # 收集需要更新的参数和对应的梯度
            params = []
            grads = []
            for layer in model.layers:
                if hasattr(layer, 'weights_gradients') and hasattr(layer, 'bias_gradients'):
                    params.extend([layer.weights, layer.bias])
                    grads.extend([layer.weights_gradients, layer.bias_gradients])

            # 使用优化器更新参数
            optimizer.step(params, grads)

            # 计算预测标签
            predicted_labels = np.argmax(output, axis=1).get()
            true_labels = np.argmax(y_batch, axis=1).get()
            # 统计正确预测的数量
            correct_predictions += sum(predicted_labels == true_labels)
            total_samples += len(true_labels)

        # 计算本轮的平均损失
        epoch_loss /= num_batches
        # 计算本轮的准确率
        accuracy = correct_predictions / total_samples

        # 记录本轮的训练损失
        train_losses.append(epoch_loss)

    return train_losses, val_losses, val_accuracies







