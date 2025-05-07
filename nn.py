import numpy as np

class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((1, out_features))

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, gradients):
        dW = np.dot(self.inputs.T, gradients)
        db = np.sum(gradients, axis=0, keepdims=True)
        dX = np.dot(gradients, self.weights.T)
        return dX, dW, db


class ReLU:
    def __init__(self):
        pass

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, gradients):
        return gradients * (self.inputs > 0)


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.input_data = None

    def forward(self, input_data):
        self.input_data = input_data
        return np.where(input_data > 0, input_data, self.alpha * input_data)

    def backward(self, output_gradient):
        d_input = np.where(self.input_data > 0, 1, self.alpha)
        return output_gradient * d_input


class Dropout:
    def __init__(self, dropout_rate: float):
        self.dropout_rate = dropout_rate

    def forward(self, inputs, train_mode=True):
        if not train_mode:
            return inputs
        self.mask = np.random.binomial(
            1, 1 - self.dropout_rate, size=inputs.shape)
        return inputs * self.mask / (1 - self.dropout_rate)

    def backward(self, gradients, train_mode=True):
        if not train_mode:
            return gradients
        return gradients * self.mask / (1 - self.dropout_rate)


class Softmax:
    def __init__(self):
        pass

    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        return probabilities

    def backward(self, gradients):
        batch_size, num_classes = gradients.shape
        d_inputs = np.empty_like(gradients)
        for i in range(batch_size):
            softmax_output = self.inputs[i]
            jacobian_matrix = np.diag(
                softmax_output) - np.outer(softmax_output, softmax_output)
            d_inputs[i] = np.dot(jacobian_matrix, gradients[i])
        return d_inputs


class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, predicted, true):
        self.predicted = predicted
        self.true = true
        epsilon = 1e-12
        predicted = np.clip(predicted, epsilon, 1. - epsilon)
        loss = -np.sum(true * np.log(predicted)) / true.shape[0]
        return loss

    def backward(self):
        d_predicted = (self.predicted - self.true) / \
            self.true.shape[0]
        return d_predicted


class SGD:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def step(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad


class Adam:
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + \
                (1 - self.beta2) * (grads[i] ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            params[i] -= self.learning_rate * \
                m_hat / (np.sqrt(v_hat) + self.epsilon)


class BatchNorm:
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, num_features, height, width = inputs.shape
        mean = np.mean(inputs, axis=(0, 2, 3), keepdims=True)
        var = np.var(inputs, axis=(0, 2, 3), keepdims=True)
        self.x_hat = (inputs - mean) / np.sqrt(var + self.epsilon)
        output = self.gamma * self.x_hat + self.beta
        self.running_mean = self.momentum * self.running_mean + \
            (1 - self.momentum) * mean.squeeze()
        self.running_var = self.momentum * self.running_var + \
            (1 - self.momentum) * var.squeeze()
        return output

    def backward(self, gradients):
        batch_size, num_features, height, width = gradients.shape
        d_x_hat = gradients * self.gamma
        d_var = np.sum(d_x_hat * (self.inputs - self.running_mean) * -0.5 * np.power(
            self.running_var + self.epsilon, -1.5), axis=(0, 2, 3), keepdims=True)
        d_mean = np.sum(d_x_hat * -1 / np.sqrt(self.running_var + self.epsilon), axis=(0, 2, 3), keepdims=True) + \
            d_var * np.mean(-2 * (self.inputs - self.running_mean),
                            axis=(0, 2, 3), keepdims=True)
        d_inputs = d_x_hat / np.sqrt(self.running_var + self.epsilon) + d_var * 2 * (
            self.inputs - self.running_mean) / (height * width) + d_mean / (height * width)
        d_gamma = np.sum(gradients * self.x_hat,
                         axis=(0, 2, 3), keepdims=False)
        d_beta = np.sum(gradients, axis=(0, 2, 3), keepdims=False)
        return d_inputs, d_gamma, d_beta


class Flatten:
    def __init__(self):
        pass

    def forward(self, inputs):
        self.inputs_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, gradients):
        return gradients.reshape(self.inputs_shape)


class Conv:
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(
            output_channels, input_channels, kernel_size, kernel_size) * 0.01
        self.bias = np.zeros(output_channels)

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, input_channels, input_height, input_width = inputs.shape
        filter_height, filter_width = self.weights.shape[2], self.weights.shape[3]
        col = self.im2col(inputs, filter_height, filter_width)
        col_w = self.weights.reshape(self.output_channels, -1).T
        out = np.dot(col, col_w) + self.bias
        output_height = (input_height + 2 * self.padding -
                         filter_height) // self.stride + 1
        output_width = (input_width + 2 * self.padding -
                        filter_width) // self.stride + 1
        out = out.reshape(batch_size, output_height,
                          output_width, self.output_channels)
        out = out.transpose(0, 3, 1, 2)
        return out

    def backward(self, gradients):
        batch_size, output_channels, output_height, output_width = gradients.shape
        filter_height, filter_width = self.weights.shape[2], self.weights.shape[3]
        col_grad = gradients.transpose(
            0, 2, 3, 1).reshape(-1, self.output_channels)
        col_input = self.im2col(self.inputs, filter_height, filter_width)
        d_weights = np.dot(col_input.T, col_grad)
        d_bias = np.sum(col_grad, axis=0)
        d_col = np.dot(col_grad, self.weights.reshape(
            self.output_channels, -1))
        d_input = self.col2im(d_col, self.inputs.shape, filter_height,
                              filter_width, stride=self.stride, padding=self.padding)
        return d_input, d_weights, d_bias

class MaxPooling:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.ctx = None

    def forward(self, input_data):
        self.input_data = input_data
        B, C, H, W = input_data.shape
        PH = PW = self.pool_size
        S = self.stride

        H_out = (H - PH) // S + 1
        W_out = (W - PW) // S + 1
        out = np.zeros((B, C, H_out, W_out))
        self.mask = np.zeros_like(input_data)  # 用于记录最大值位置

        for b in range(B):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        h_end = h_start + PH
                        w_start = j * S
                        w_end = w_start + PW
                        region = input_data[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(region)
                        out[b, c, i, j] = max_val
                        # 记录最大值位置（用于反向传播）
                        max_mask = (region == max_val)
                        self.mask[b, c, h_start:h_end,
                                  w_start:w_end] += max_mask

        self.ctx = (input_data.shape, out.shape)
        return out

    def backward(self, output_gradient, learning_rate=None):
        B, C, H_out, W_out = output_gradient.shape
        PH = PW = self.pool_size
        S = self.stride

        dX = np.zeros_like(self.input_data)

        for b in range(B):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        h_end = h_start + PH
                        w_start = j * S
                        w_end = w_start + PW
                        mask = self.mask[b, c, h_start:h_end, w_start:w_end]
                        dX[b, c, h_start:h_end, w_start:w_end] += mask * \
                            output_gradient[b, c, i, j]

        return dX

class Dense:
    def __init__(self, input_size, output_size, activation_function=None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function

        # 初始化权重和偏置
        self.weights = np.random.randn(
            input_size, output_size) * 0.01  # 小随机数初始化
        self.bias = np.zeros((1, output_size))  # 偏置初始化为0

        # 保存前向传播的输入值，用于反向传播
        self.inputs = None
        self.outputs = None
        self.weights_gradients = None
        self.bias_gradients = None

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.bias  # 线性变换

        if self.activation_function:
            self.outputs = self.activation_function(self.outputs)  # 激活函数

        return self.outputs

    def backward(self, gradients, learning_rate=0.01):
        d_weights = np.dot(self.inputs.T, gradients)
        d_bias = np.sum(gradients, axis=0, keepdims=True)
        d_inputs = np.dot(gradients, self.weights.T)

        self.weights_gradients = d_weights
        self.bias_gradients = d_bias
        return d_inputs, d_weights, d_bias


def get_network():
    return [
        Conv(3, 32, 3, 1, 1),
        LeakyReLU(),
        MaxPooling(),
        Conv(32, 64, 3, 1, 1),
        LeakyReLU(),
        MaxPooling(),
        Conv(64, 64, 3, 1, 1),
        LeakyReLU(),
        MaxPooling(),
        Flatten(),
        Dense(4 * 4 * 64, 32),
        LeakyReLU(),
        Dense(32, 10)
    ]
