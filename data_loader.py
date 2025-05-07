import random
import os
import pickle
import numpy as np


def load_cifar10_batch(batch_file):
    with open(batch_file, 'rb') as f:
        batch_data = pickle.load(f, encoding='bytes')
    data = batch_data[b'data']
    labels = batch_data[b'labels']

    data = data.reshape((len(data), 3, 32, 32))
    data = data.astype(np.float32)

    return data, np.array(labels)


def load_cifar10_data(data_dir):
    x_train = []
    y_train = []

    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f"data_batch_{i}")
        data, labels = load_cifar10_batch(batch_file)
        x_train.append(data)
        y_train.append(labels)

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    test_batch_file = os.path.join(data_dir, "test_batch")
    x_test, y_test = load_cifar10_batch(test_batch_file)

    return x_train, y_train, x_test, y_test


def one_hot_encode(labels, num_classes=10):
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    one_hot_labels[np.arange(labels.shape[0]), labels] = 1
    return one_hot_labels


def load_and_process_cifar10(data_dir):
    x_train, y_train, x_test, y_test = load_cifar10_data(data_dir)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    return x_train, y_train, x_test, y_test


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img[:, :, ::-1]  # (C, H, W)
        return img

