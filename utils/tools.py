# -*- coding=utf-8 -*-
import numpy as np
from PIL import ImageDraw, Image
from scipy.spatial import distance as dist
import cv2

def read_from_gt(gt_file):
    with open(gt_file) as file:
        lines = file.readlines()
        gt_bboxs = []
        txts = []
        for line in lines:
            splited_line = line.split(',')
            splited_line_num = splited_line[:8]
            splited_line_num = [int(float(ele)) for ele in splited_line_num]
            gt_bboxs.append(splited_line_num)
            txts.append(splited_line[8])
        return gt_bboxs, txts


def show_image_from_array(image_arr):
    from PIL import Image
    img = Image.fromarray(image_arr)
    img.show()


def vis_img_bbox(img_file, gt_file):
    img = cv2.imread(img_file)[:, :, ::-1]
    gtbboxes = np.asarray(read_from_gt(gt_file)[0])
    print np.shape(gtbboxes)
    for box in gtbboxes:
        cv2.polylines(img[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0),
                      thickness=1)
    show_image_from_array(img)

def save_gt_file(save_gt_path, coordinations, txts=None):
    with open(save_gt_path, 'wb+') as f:
        strs = []
        start_index = 0
        for idx in range(len(coordinations)):
            cur_str = ','.join([str(element) for element in coordinations[start_index]])
            if txts is None:
                # cur_str += (',' + txts[idx])
                cur_str += ',TXT\n'
            else:
                txts[start_index] = str(txts[start_index]).replace('\n', '')
                cur_str += (',' + txts[start_index] + '\n')
            strs.append(cur_str)
            start_index += 1
        f.writelines(strs)
        f.close()

def cal_TP(overlaps, threshold=0.7):
    shape = list(np.shape(overlaps))
    count = 0
    for i in range(shape[0]):
        max_val = np.max(overlaps[i])
        if max_val >= threshold:
            count += 1
    return count


def cal_FP(overlaps, len_pred, threshold=0.7):
    return len_pred - cal_TP(overlaps, threshold)


def cal_FN(overlaps, len_gt, threshold=0.7):
    '''
    找到被漏检的，也就是说实际上是gt，但是没有被任何bbox检测出来
    :param overlaps:
    :param len_gt:
    :param threshold:
    :return:
    '''
    if (len_gt - cal_TP(overlaps, threshold)) < 0:
        print 'Error, FN is negative'
        assert False
    max_value = np.max(overlaps, axis=0)
    return np.sum(max_value < threshold)
    # return len_gt - cal_TP(overlaps, threshold)
def draw_rect(img_arr, box):
    points = [[0, 0], [0, 0], [0, 0], [0, 0]]
    ind = 0
    for i in range(4):
        for j in range(2):
            points[i][j] = box[ind]
            ind += 1
        points[i] = tuple(points[i])
    img = Image.fromarray(np.array(img_arr))
    img_draw = ImageDraw.Draw(img)
    img_draw.polygon(points, fill=128)
    return img
def draw_rects(image_arr, boxs):
    img = Image.fromarray(image_arr)
    img_draw = ImageDraw.Draw(img)
    for box in boxs:
        points = [[0, 0], [0, 0], [0, 0], [0, 0]]
        ind = 0
        for i in range(4):
            for j in range(2):
                points[i][j] = box[ind]
                ind += 1
            points[i] = tuple(points[i])
        img_draw.polygon(points, fill=128)
    return img
def bbox_overlaps(boxes, query_boxes, im_size):
    '''

    :param boxes: (N, 8) ndarray of float, pred
    :param query_boxes: (K, 8) ndarray of float, gt
    :param im_size: 图像的大小
    四个点的顺序如下所示
    1   4
    2   3
    :return: (N, K) ndarray of overlap between boxes and query_boxes
    '''
    def cal_overlap(img1, img2):
        img1 = np.array(img1)
        img2 = np.array(img2)
        img1 = (img1 == 128)
        img2 = (img2 == 128)
        return np.sum(np.logical_and(img1, img2))
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    # draw_rects(np.zeros(im_size), query_boxes).show()
    # draw_rects(np.zeros(im_size), boxes).show()
    for k in range(K):
        # 计算ground truth的面积
        cur_gt = query_boxes[k]
        gt_white_img = np.zeros(im_size, np.uint8)
        gt_box_img = draw_rect(gt_white_img, cur_gt)
        # gt_box_img.show()
        gt_area = np.sum(np.array(gt_box_img) == 128)
        for n in range(N):
            # 也就是说最小的左下方的横坐标减去最大的右上方的横坐标代表的就是IoU部分的宽度
            cur_bbox = boxes[n]
            pred_white_img = np.zeros(im_size, np.uint8)
            pred_box_img = draw_rect(pred_white_img, cur_bbox)
            # pred_box_img.show()
            pred_area = np.sum(np.array(pred_box_img) == 128)
            overlap_area = cal_overlap(gt_box_img, pred_box_img)
            overlaps[n, k] = (overlap_area * 1.0) / ((pred_area + gt_area - overlap_area) * 1.0)
    return overlaps


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

if __name__ == '__main__':
    img_size = [400, 300]
    pred_points = np.array(
        [
            [0, 0, 0, 100, 100, 100, 100, 0],
            [77, 92, 77, 195, 483, 195, 483, 92]
        ]
    )
    gt_points = np.array(
        [
            [0, 0, 0, 50, 50, 50, 50, 0],
            [80, 93, 77, 195, 483, 195, 483, 92]
        ]
    )
    print bbox_overlaps(pred_points, gt_points, img_size)