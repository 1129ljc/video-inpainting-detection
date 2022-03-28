import cv2
import numpy as np

def post_deal(predict_mask):
    predict_mask = predict_mask.astype(np.uint8)
    img = cv2.medianBlur(predict_mask, 5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_clo = cv2.dilate(binary, kernel2, iterations=2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)
    # print('num_labels = ', num_labels)
    # 连通域的信息：对应各个轮廓的x、y、width、height和面积
    # print('stats = ', stats)
    # 连通域的中心点
    # print('centroids = ', centroids)
    # 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
    # print('labels = ', labels)
    # 不同的连通域赋予不同的颜色
    output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(1, num_labels):
        mask = labels == i
        output[:, :, 0][mask] = np.random.randint(0, 255)
        output[:, :, 1][mask] = np.random.randint(0, 255)
        output[:, :, 2][mask] = np.random.randint(0, 255)
    return num_labels, output