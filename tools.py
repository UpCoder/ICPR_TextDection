# -*- coding=utf-8 -*-
from PIL import Image
import numpy as np
import cv2
import sys
from utils.tools import draw_rect, order_points
from icdar import expand_poly
sys.setrecursionlimit(300000000)


def show_image(image_arr):
    img = Image.fromarray(image_arr)
    img.show()

def find_connected(score_map, threshold=0.7):
    binary_map = (score_map > threshold).astype(np.uint8)
    connectivity = 8
    output = cv2.connectedComponentsWithStats(binary_map, connectivity=connectivity, ltype=cv2.CV_32S)
    label_map = output[1]
    # show_image(np.asarray(label_map * 100.0, np.uint8))
    return np.max(label_map), label_map


def calculate_boundingbox_score(score_map, threshold=0.7):
    # score_map = score_map[::-1, :]
    score_map[score_map < threshold] = 0.0
    h, w = np.shape(score_map)
    # show_image(np.asarray(score_map * 255, np.uint8))
    flag = np.zeros([h, w])
    boundingboxs = []
    rects = []
    count_connecter, label_map = find_connected(score_map, threshold)
    label_map = np.array(label_map)
    bbox_image = np.zeros(np.shape(label_map), np.uint8)
    expand_image = np.zeros(np.shape(label_map), np.uint8)
    for idx in range(1, count_connecter+1):
        connected = np.array(np.where(label_map == idx)).transpose((1, 0))
        rect = cv2.minAreaRect(np.array(connected))
        rects.append(rect)
        bbox = order_points(cv2.boxPoints(rect)[:, ::-1])
        r = [None, None, None, None]
        for i in range(4):
            r[i] = min(np.linalg.norm(bbox[i] - bbox[(i + 1) % 4]),
                       np.linalg.norm(bbox[i] - bbox[(i - 1) % 4]))
        expand_bbox = expand_poly(bbox.copy(), r).astype(np.int32)
        boundingboxs.append(expand_bbox)
        cur_points = []
        expand_points = []
        for i in range(4):
            for j in range(2):
                cur_points.append(bbox[i, j])
                expand_points.append(expand_bbox[i, j])
        expand_image = draw_rect(expand_image, expand_points)
        bbox_image = draw_rect(bbox_image, cur_points)

    for i in range(len(rects)):
        for j in range(len(rects)):
            if i == j:
                continue
            rect1 = rects[i]
            rect2 = rects[j]
            theta1 = rect1[2]
            theta2 = rect2[2]
            if abs(theta1 - theta2) < 5:
                center1 = rect1[0]
                center2 = rect2[1]
                center_distance = (center1[0] - center2[0])**2 + (center1[1] - center2[1])**2
                # dis_sub_width = center_distance - rect1[1][]
                # print 'ok'
    points = []
    for bbox in boundingboxs:
        cur_points = []
        for i in range(4):
            for j in range(2):
                cur_points.append(bbox[i, j])
        points.append(cur_points)
    return np.array(points)

if __name__ == '__main__':
    test = np.zeros([100, 100])
    test[10:21, 10:81] = 1.0

    test[30:40, 10:81] = 1.0

    test[42:50, 10:81] = 1.0

    test[62:70, 10:81] = 1.0

    test[82:90, 10:81] = 1.0
    find_connected(test)
    show_image(np.asarray(test * 255, np.uint8))
    # xys = np.argwhere(test != 0)
    # rect = cv2.minAreaRect(xys)
    # print rect