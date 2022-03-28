import torch
import numpy as np


def get_sobel(in_chan, out_chan):
    ''''''
    filter_x = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1],
                         ]).astype(np.float32)
    filter_y = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1],
                         ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    return filter_x, filter_y


def DCT_mat(size):
    m = []
    for i in range(size):
        n = []
        for j in range(size):
            if i == 0:
                x = np.sqrt(1. / size)
            else:
                x = np.sqrt(2. / size)
            x = x * np.cos((j + 0.5) * np.pi * i / size)
            n.append(x)
        m.append(n)
    return m


def generate_filter(start, end, size):
    m = []
    for i in range(size):
        n = []
        for j in range(size):
            if i + j > end or i + j <= start:
                x = 0.
            else:
                x = 1.
            n.append(x)
        m.append(n)
    return m


def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.


def TP(predict, label):
    predict_ = np.where(predict <= 0.5, 0, 1)
    predict_ = np.array(predict_).flatten()
    label = np.array(label).flatten()
    tp = ((predict_ == 1) & (label == 1)).astype('int')
    tp = np.sum(tp)
    return tp


def FN(predict, label):
    fn = ((predict == 0) & (label == 1)).astype('int')
    fn = np.sum(fn)
    return fn


def FP(predict, label):
    fp = ((predict == 1) & (label == 0)).astype('int')
    fp = np.sum(fp)
    return fp


def precision(predict, label):
    smooth = 1e-5
    tp = TP(predict, label)
    fp = FP(predict, label)
    precision = tp / (tp + fp + smooth)
    return precision


def recall(predict, label):
    smooth = 1e-5
    tp = TP(predict, label)
    fn = FN(predict, label)
    recall = tp / (tp + fn + smooth)
    return recall


def F1(predict, label):
    smooth = 1e-5
    p = precision(predict, label)
    r = recall(predict, label)
    return 2 * (p * r) / (p + r + smooth)


def IOU(predict, label):
    smooth = 1e-5
    tp = TP(predict, label)
    fp = FP(predict, label)
    fn = FN(predict, label)
    return tp / (tp + fp + fn + smooth)


if __name__ == '__main__':
    pass
