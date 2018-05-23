# -*- coding=utf-8 -*-
import numpy as np
import logging
from icdar import restore_rectangle
from multiprocessing import Pool
import logging
from shapely.geometry import Polygon
import multiprocessing

def cal_IoU_gt_py_multiprocess(pred_geo, pred_cls, gt, threshold=0.8):
    def compute_IoU(polygon1, polygon2):
        '''
        计算两个rect的IoU值
        :param polygon1: 4， 2
        :param polygon2: 4， 2
        :return: 0~1 value
        '''
        polygon1 = Polygon(polygon1)
        if not polygon1.is_valid:
            polygon1 = polygon1.buffer(0)
        polygon2 = Polygon(polygon2)
        if not polygon2.is_valid:
            polygon2 = polygon2.buffer(0)
        intersection_polygon = polygon1.intersection(polygon2)
        if not intersection_polygon.is_valid:
            return 0.0
        intersection_area = intersection_polygon.area
        uion_area = polygon1.area + polygon2.area - intersection_area
        return (1.0 * intersection_area) / (1.0 * uion_area)

    '''
    根据预测得到的pred_geo 和 pred_cls 我们针对每个pixel都可以计算他和ground truth的IoU值
    :param pred_geo: N, W, H, 5
    :param pred_cls: N, W, H, 1
    :param gt: N, M, 4, 2
    :param threshold: 0.8
    :return:
    '''
    # 删除纬度数是1的纬度

    print 'hello0'
    pred_cls = np.squeeze(pred_cls)
    shape = np.shape(pred_geo)
    IoU_gt = np.zeros([shape[0], shape[1], shape[2], 1], np.float32)

    for batch_id in range(shape[0]):
        process_num = 8
        pool = Pool(processes=process_num)
        print 'hello1'
        score_map = pred_cls[batch_id]
        geo_map = pred_geo[batch_id]
        cur_gt = gt[batch_id]

        print 'hello2'
        # print 'the shape of score_map is ', np.shape(score_map)
        # print 'the shape of geo_map is ', np.shape(geo_map)
        if len(np.shape(score_map)) != 2:
            logging.log(logging.ERROR, 'score map shape isn\'t correct!')
            assert False
        xy_text = np.argwhere(score_map > threshold)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        # print 'The number of points that satisfy the condition is ', len(xy_text)
        text_box_restored = restore_rectangle(xy_text[:, ::-1], geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
        # print np.shape(text_box_restored)

        pre_process_num = len(xy_text) / process_num + 1
        xss = {}
        yss = {}
        boxss = {}

        print 'hello3'
        for idx, ((x, y), box) in enumerate(zip(xy_text, text_box_restored)):
            process_id = idx / pre_process_num
            if process_id not in xss.keys():
                xss[process_id] = []
                yss[process_id] = []
                boxss[process_id] = []
                xss[process_id].append(x)
                yss[process_id].append(y)
                boxss[process_id].append(box)
            else:
                xss[process_id].append(x)
                yss[process_id].append(y)
                boxss[process_id].append(box)

        print 'hello4'

        def process_single_test():
            return 1.0
        def process_single(boxs, cur_gt):
            print 'hello4-0'
            IoU_values = []
            print 'hello4-1'
            return np.random.random(len(boxs))
            for box in boxs:
                cur_IoU_value = 0.0
                print 'hello4-2'
                for gt_id in range(len(cur_gt)):
                    if np.sum(cur_gt[gt_id]) == -8:
                        break
                    cur_IoU_value = max(cur_IoU_value, compute_IoU(np.asarray(box), np.asarray(cur_gt[gt_id])))
                IoU_values.append(cur_IoU_value)
                print 'hello4-3'
            print 'hello4-3'
            return IoU_values
        results = []

        print 'hello5'
        for process_id in range(process_num):
            print 'hello6'
            # results.append(pool.apply_async(func=process_single, args=(boxss[process_id], cur_gt, )))
            results.append(pool.apply_async(func=process_single_test, args=()))
            print 'hello7'
        pool.close()
        pool.join()

        print 'hello8'
        for process_id, res in enumerate(results):
            xs = xss[process_id]
            ys = yss[process_id]

            print 'hello9'
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            print np.shape(xs)
            print np.shape(ys)
            IoU_values = res.get()
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            print np.shape(IoU_values)
            print np.shape(xs)
            print np.shape(ys)
            IoU_gt[batch_id, xs, ys, 0] = IoU_values

            print 'hello10'

        print 'hello11'
    return IoU_gt

def cal_IoU_gt_py(pred_geo, pred_cls, gt, threshold=0.8):
    def compute_IoU(polygon1, polygon2):
        '''
        计算两个rect的IoU值
        :param polygon1: 4， 2
        :param polygon2: 4， 2
        :return: 0~1 value
        '''
        polygon1 = Polygon(polygon1)
        if not polygon1.is_valid:
            polygon1 = polygon1.buffer(0)
        polygon2 = Polygon(polygon2)
        if not polygon2.is_valid:
            polygon2 = polygon2.buffer(0)
        intersection_polygon = polygon1.intersection(polygon2)
        if not intersection_polygon.is_valid:
            return 0.0
        intersection_area = intersection_polygon.area
        uion_area = polygon1.area + polygon2.area - intersection_area
        return (1.0 * intersection_area) / (1.0 * uion_area)

    '''
    根据预测得到的pred_geo 和 pred_cls 我们针对每个pixel都可以计算他和ground truth的IoU值
    :param pred_geo: N, W, H, 5
    :param pred_cls: N, W, H, 1
    :param gt: N, M, 4, 2
    :param threshold: 0.8
    :return:
    '''
    # 删除纬度数是1的纬度

    pred_cls = np.squeeze(pred_cls)
    shape = np.shape(pred_geo)
    IoU_gt = np.zeros([shape[0], shape[1], shape[2], 1], np.float32)

    for batch_id in range(shape[0]):
        score_map = pred_cls[batch_id]
        geo_map = pred_geo[batch_id]
        cur_gt = gt[batch_id]

        if len(np.shape(score_map)) != 2:
            logging.log(logging.ERROR, 'score map shape isn\'t correct!')
            assert False
        xy_text = np.argwhere(score_map > threshold)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        # print 'The number of points that satisfy the condition is ', len(xy_text)
        text_box_restored = restore_rectangle(xy_text[:, ::-1], geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
        # print np.shape(text_box_restored)

        for idx, ((x, y), box) in enumerate(zip(xy_text, text_box_restored)):
            cur_IoU_value = 0.0
            for gt_id in range(len(cur_gt)):
                if np.sum(cur_gt[gt_id]) == -8:
                    break
                cur_IoU_value = max(cur_IoU_value, compute_IoU(np.asarray(box), np.asarray(cur_gt[gt_id])))
            IoU_gt[batch_id, x, y, 0] = cur_IoU_value
    return IoU_gt


if __name__ == '__main__':
    # pred_geo = np.random.random([2, 512, 512, 5])
    # pred_cls = np.random.random([2, 512, 512, 1])
    # cal_IoU_gt_py(pred_geo, pred_cls, None)
    def process_single_test():
        print 'test'
        return 1.0
    process_num = 8
    pool = Pool(processes=process_num)
    results = []
    for i in range(process_num):
        results.append(pool.apply_async(func=process_single_test, args=()))
    pool.close()
    pool.join()
    for i in range(process_num):
        print results[i].get()

