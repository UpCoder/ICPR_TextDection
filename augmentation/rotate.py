import numpy as np
import cv2
from utils.tools import generate_mask, read_from_gt, generate_bbox_from_mask, show_image_from_array
import os
from multiprocessing import Process, Pool


def rotate_single(image, gt_boxes, angle_range=10):
    def rotate_bounding_box(image, box, M, center):
        mask_image = generate_mask(image, [box])
        mask_image = cv2.warpAffine(mask_image, M, center)
        rotated_box = generate_bbox_from_mask(mask_image)
        # print 'rotated_box shape: ', np.shape(rotated_box)
        return rotated_box
    '''
    rotate single image with random angle
    :param image: image
    :param gt_boxes: bounding box in image that will rotate at same time. [N, 8]
    :param angle_range: 
    :return:
    '''
    (h, w) = np.shape(image)[:2]
    # if np.random.random() > 0.5:
    #     center = (0, 0)
    # else:
    #     center = (w, h)
    center = (w, h)
    scale = 1.0

    random_angle = np.random.random()
    # if np.random.random() >= 0.5:
    #     random_angle *= angle_range
    # else:
    #     random_angle *= (angle_range * -1)
    random_angle *= angle_range
    cv2.rotate(image, int(random_angle))
    M = cv2.getRotationMatrix2D(center, random_angle, scale)
    image_rotated = cv2.warpAffine(image, M, center)
    gt_boxes_rotated = []
    ignore = []
    for index, gt_box in enumerate(gt_boxes):
        cur_box = np.squeeze(rotate_bounding_box(image, gt_box, M, center))
        if len(cur_box) == 0:
            ignore.append(index)
            continue
        if len(np.shape(cur_box)) == 3:
            gt_boxes_rotated.extend(cur_box)
        if len(np.shape(cur_box)) == 2:
            gt_boxes_rotated.append(cur_box)
    return image_rotated, generate_mask(image, gt_boxes_rotated), gt_boxes_rotated, ignore


def generate_rotate_image_gtfile(image_path, gt_path, save_image_path, save_gt_path, save_mask_path):
    image = cv2.imread(image_path)
    gt_boxes, txts = read_from_gt(gt_path)
    image_rotated, mask_image_rotated, coordination_rotated, ignore = rotate_single(image, gt_boxes)
    coordination_shape = np.shape(coordination_rotated)
    try:
        if len(coordination_rotated) != 0 and coordination_shape[1] == 4 and coordination_shape[2] == 2:
            coordination_rotated = np.reshape(coordination_rotated, [coordination_shape[0], -1])
    except Exception, e:
        print len(coordination_rotated)
        print np.shape(coordination_rotated)
        for idx in range(len(coordination_rotated)):
            print coordination_rotated[idx], np.shape(coordination_rotated[idx])
        raise False
    cv2.imwrite(save_image_path, image_rotated)
    with open(save_gt_path, 'wb+') as f:
        strs = []
        start_index = 0
        for idx in range(len(coordination_rotated)):
            cur_str = ','.join([str(element) for element in coordination_rotated[start_index]])
            # cur_str += (',' + txts[idx])
            cur_str += ',TXT\n'
            strs.append(cur_str)
            start_index += 1
        f.writelines(strs)
        f.close()
    if save_mask_path is not None:
        rotated_mask_image = generate_mask(image_rotated, coordination_rotated)
        cv2.imwrite(save_mask_path, rotated_mask_image)


def generate_rotate_images_gtfiles(image_dir, gt_dir, save_img_dir, save_gt_dir, save_mask_dir=None, rotated_num_per_img=10, process_num=8):
    def generate_rotate_images(image_pathes, gt_pathes, save_img_dir, save_gt_dir, save_mask_dir, rotated_num_per_img, process_id):
        for img_id, img_path in enumerate(image_pathes):
            basename = os.path.basename(img_path)
            for rotated_idx in range(rotated_num_per_img):
                if save_mask_dir is not None:
                    mask_path = os.path.join(save_mask_dir, basename.split('.')[0] + '_' + str(rotated_idx) + '.png')
                else:
                    mask_path = None
                save_img_path = os.path.join(save_img_dir, basename.split('.')[0] + '_' + str(
                    rotated_idx) + '.png')
                if os.path.exists(save_img_path):
                    continue
                generate_rotate_image_gtfile(img_path, gt_pathes[img_id],
                                             os.path.join(save_img_dir, basename.split('.')[0] + '_' + str(rotated_idx) + '.png'),
                                             os.path.join(save_gt_dir, basename.split('.')[0] + '_' + str(rotated_idx) + '.txt'),
                                             mask_path)
                print 'saving at ', os.path.join(save_img_dir, basename.split('.')[0] + '_' + str(
                    rotated_idx) + '.png'), ' at Process ', process_id
            if img_id % 100 == 0:
                print "-" * 10, 'Process %d / %d' % (name_id, len(image_pathes)), ' at Process ', process_id, '-' * 10
    img_pathes = []
    gt_pathes = []
    names = os.listdir(image_dir)
    for name_id, name in enumerate(names):
        # if not name.endswith('6692.png'):
        #     continue
        img_path = os.path.join(image_dir, name)
        gt_path = os.path.join(gt_dir, name.split('.')[0]+'.txt')
        img_pathes.append(img_path)
        gt_pathes.append(gt_path)
    start = 0
    pre_process_num = int(len(img_pathes) / process_num + 1)
    for process_id in range(process_num):
        end = start + pre_process_num
        cur_img_pathes = img_pathes[start:end]
        cur_gt_pathes = gt_pathes[start:end]
        process = Process(target=generate_rotate_images,
                          args=[cur_img_pathes, cur_gt_pathes, save_img_dir, save_gt_dir, save_mask_dir,
                                rotated_num_per_img, process_id, ])
        process.start()
        start = end
        #
        # for idx in range(rotated_num_per_img):
        #     if save_mask_dir is not None:
        #         mask_path = os.path.join(save_mask_dir, name.split('.')[0] + '_' + str(idx) + '.png')
        #     else:
        #         mask_path = None
        #     generate_rotate_image_gtfile(img_path, gt_path,
        #                                  os.path.join(save_img_dir, name.split('.')[0] + '_' + str(idx) + '.png'),
        #                                  os.path.join(save_gt_dir, name.split('.')[0] + '_' + str(idx) + '.txt'),
        #                                  mask_path)
        #     print 'saving at ', os.path.join(save_img_dir, name.split('.')[0] + '_' + str(idx) + '.png')
        # if name_id % 100 == 0:
        #     print "-"*10, 'Process %d / %d' % (name_id, len(names)), '-'*10

def rotate(image, angle, center=None, scale=1.0):

    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

if __name__ == '__main__':

    # generate_rotate_images_gtfiles(
    #     '/home/give/Game/OCR/data/ICPR/rename/9000/image_9000',
    #     '/home/give/Game/OCR/data/ICPR/rename/9000/txt_9000_clockwise',
    #     '/home/give/Game/OCR/data/ICPR/rename/rotated/image',
    #     '/home/give/Game/OCR/data/ICPR/rename/rotated/txt'
    #     # '/home/give/Game/OCR/Papers-code/EAST-master/augmentation/rotated/image',
    #     # '/home/give/Game/OCR/Papers-code/EAST-master/augmentation/rotated/txt',
    #     # '/home/give/Game/OCR/Papers-code/EAST-master/augmentation/rotated/mask'
    # )



    # image_path = '/home/give/Game/OCR/data/ICPR/rename/100/image_100/2.png'
    # gt_path = '/home/give/Game/OCR/data/ICPR/rename/100/txt_100/2.txt'
    # generate_rotate_image_gtfile(image_path, gt_path, './rotated_image.png', './rotated_gt.txt')
    # image = cv2.imread(image_path)
    # gt_boxes = read_from_gt(gt_path)
    # mask_image = generate_mask(image, gt_boxes)
    # show_image_from_array(mask_image)
    # print np.shape(gt_boxes)
    # image_rotated, mask_image_rotated, _ = rotate_single(image, gt_boxes)
    # show_image_from_array(image_rotated)
    # show_image_from_array(mask_image_rotated)

    image_path = '/home/give/Game/OCR/data/ICPR/rename/100/image_100/10.png'
    image = cv2.imread(image_path)
    print np.shape(image)
    rotated_image = rotate(image, 90, scale=0.5)

    print np.shape(rotated_image)
    cv2.namedWindow("Image")
    cv2.imshow("Image", rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()