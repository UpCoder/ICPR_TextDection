import cv2
from utils.tools import read_from_gt
import imutils
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import time
from utils.tools import save_gt_file, vis_img_bbox
ia.seed(int((time.time()*1000)%100000))

def data_agumentation(img, gt_bbox, operation_obj, txts=None, save_flag=None):
    shape = np.shape(gt_bbox)
    [h, w, _] = np.shape(img)
    if shape[1] == 8:
        bboxes = np.reshape(gt_bbox, [-1, 4, 2])
    else:
        bboxes = gt_bbox
    keypoints_on_images = []
    keypoints_imgaug_obj = []
    # print bboxes
    # print np.shape(bboxes)
    for key_points in bboxes:
        # print key_points
        for key_point in key_points:
            keypoints_imgaug_obj.append(ia.Keypoint(x=key_point[0], y=key_point[1]))
    keypoints_on_images.append(ia.KeypointsOnImage(keypoints_imgaug_obj, shape=img.shape))

    seq_det = operation_obj.to_deterministic()

    img_aug = seq_det.augment_image(img)
    key_points_aug = seq_det.augment_keypoints(keypoints_on_images)
    key_points_after = []
    for idx, (keypoints_before, keypoints_after) in enumerate(zip(keypoints_on_images, key_points_aug)):
        for kp_idx, keypoint in enumerate(keypoints_after.keypoints):
            keypoint.x = keypoint.x if keypoint.x < w else w
            keypoint.x = keypoint.x if keypoint.x > 0 else 0
            keypoint.y = keypoint.y if keypoint.y < h else h
            keypoint.y = keypoint.y if keypoint.y > 0 else 0
            key_points_after.append([keypoint.x, keypoint.y])
    # print np.shape(key_points_after)
    key_points_after = np.reshape(key_points_after, [-1, 4, 2])
    if save_flag:
        save_gt_file('./rotated_10.txt', np.reshape(key_points_after, [-1, 8]), txts=txts)
        cv2.imwrite('./rotated_10.png', img_aug)
        vis_img_bbox('./rotated_10.png', './rotated_10.txt')
    return img_aug, np.asarray(key_points_after, np.float32)

if __name__ == '__main__':


    # using imgaug package
    import random
    img = cv2.imread('/home/give/Game/OCR/data/ICPR/rename/100/image_100/10.png')
    gt_data = read_from_gt('/home/give/Game/OCR/data/ICPR/rename/100/txt_100/10.txt')
    coords = gt_data[0]
    bboxes = np.reshape(coords, [-1, 4, 2])

    angle = np.random.random() * 90
    operation_obj = iaa.Affine(rotate=(-angle, angle))
    # operation_obj = iaa.Sequential([iaa.Flipud(1.0)])
    # operation_obj = iaa.Sequential([iaa.Fliplr(1.0)])
    # operation_obj = iaa.Sequential([iaa.Dropout(p=(0, 0.1), random_state=np.random.randint(0, 10000))])
    # operation_obj = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=np.random.random() * 30)])
    # operation_obj = iaa.Affine(shear=(-10, 10))
    # fliplr_rate = 0.5
    # angle = 10
    # additive, contrast_norm = (45, 0.1)
    # gaussian_noise, dropout = (0.05, 0.01)
    # shear, shift = (2, 20)
    # operation_obj = iaa.Sequential([
    #     iaa.Sometimes(0.5, iaa.OneOf([
    #         iaa.Affine(rotate=(-angle, angle)),
    #         iaa.ContrastNormalization((1 - contrast_norm, 1 + contrast_norm))
    #     ])),
    #     iaa.Sometimes(0.5, iaa.OneOf([
    #         iaa.Sequential([iaa.Flipud(0.5)]),
    #         iaa.Dropout(dropout)
    #     ]))
    # ])
    data_agumentation(img, bboxes, operation_obj, save_flag=True, txts=gt_data[1])


    keypoints_on_images = []
    start = 0
    keypoints_imgaug_obj = []
    for key_points in bboxes:
        for key_point in key_points:
            keypoints_imgaug_obj.append(ia.Keypoint(x=key_point[0], y=key_point[1]))
    keypoints_on_images.append(ia.KeypointsOnImage(keypoints_imgaug_obj, shape=img.shape))
    # seq = iaa.Sequential([iaa.GaussianBlur((0, 3.0))])
    # seq = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=10)])
    # seq = iaa.Sequential([iaa.Flipud(0.5)])
    # seq = iaa.Sequential([iaa.Fliplr(0.5)])
    # seq = iaa.Sequential([iaa.Dropout(p=(0, 0.2))])
    # seq = iaa.Sequential([iaa.Affine(rotate=(-10, 10), shear=(-10, 10))])
    seq = iaa.Affine(rotate=(-60, 60), shear=(-20, 20))
    # seq = iaa.Sometimes(
    #     0.5,
    #     iaa.GaussianBlur(sigma=2.0),
    #     iaa.Sequential([iaa.Affine(rotate=45), iaa.Sharpen(alpha=1.0)])
    # )
    seq_det = seq.to_deterministic()