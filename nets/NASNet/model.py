# -*- coding=utf-8 -*-
from pnasnet import build_pnasnet_large, large_imagenet_config, pnasnet_large_arg_scope
import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
tf.app.flags.DEFINE_integer('text_scale', 512, '')
tf.app.flags.DEFINE_bool('uppool_conv', False, '')

FLAGS = tf.app.flags.FLAGS
def unpool(inputs):
    shape = inputs.get_shape().as_list()
    res = tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])
    res = slim.conv2d(res, num_outputs=shape[-1], kernel_size=[3, 3], stride=1)
    return res

def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def model(images, weight_decay=1e-5, is_training=True):
    images = mean_image_subtraction(images)
    with slim.arg_scope(pnasnet_large_arg_scope(weight_decay=weight_decay)):
        logits, end_points = build_pnasnet_large(images, num_classes=None, is_training=is_training, config=large_imagenet_config())
    for key in end_points.keys():
        print(key, end_points[key])
    print(end_points.keys())
    with tf.variable_scope('feature_fusion', values=[end_points.values()]):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['Cell_11'],     # 16
                 end_points['Cell_7'],  # 32
                 end_points['Cell_3'],  # 64
                 end_points['scale-2'],     # 128
                 end_points['scale-1']]     # 256
            g = [None, None, None, None, None]
            h = [None, None, None, None, None]
            num_outputs = [None, 1080, 128, 64, 32]
            for i in range(5):
                if i == 0:
                    h[i] = f[i]
                else:
                    # 相当于一个融合，减少维度的过程，kernel size等于1
                    c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs=num_outputs[i], kernel_size=1)
                    h[i] = slim.conv2d(c1_1, num_outputs=num_outputs[i], kernel_size=3)
                if i <= 3:
                    g[i] = unpool(h[i])
                    # g[i] = slim.conv2d(g[i], num_outputs[i + 1], 1)
                    # g[i] = slim.conv2d(g[i], num_outputs[i + 1], 3)
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print("Shape of f_{} {}, h_{} {}, g_{} {}".format(i, f[i].shape, i, h[i].shape, i, g[i].shape))
            F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            if FLAGS.geometry == 'RBOX':
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                print 'RBOX'
                geo_map = slim.conv2d(g[4], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
                angle_map = (slim.conv2d(g[4], 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)
            else:
                # LD modify
                # concated_score_map = tf.concat([F_score, g[3]], axis=-1)
                # F_geometry = slim.conv2d(g[4], 8, 1, activation_fn=parametric_relu,
                #                          normalizer_fn=None) * FLAGS.text_scale
                assert False
    return F_score, F_geometry


def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss


def smooth_l1_dist(deltas, sigma2=9.0, name='smooth_l1_dist'):
    with tf.name_scope(name=name) as scope:
        deltas_abs = tf.abs(deltas)
        smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
        return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                    (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)


def loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    type_str = 'QUAD'
    # if type_str == 'RBOX':
    if FLAGS.geometry == 'RBOX':
        classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01

        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
        L_theta = 1 - tf.cos(theta_pred - theta_gt)
        tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
        tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
        L_g = L_AABB + 20 * L_theta

        L_g = tf.reduce_mean(L_g * y_true_cls * training_mask)
        L_s = classification_loss
        return L_g + L_s, L_g, L_s
    # elif type_str == 'QUAD':
    elif FLAGS.geometry == 'QUAD':
        print 'build loss'
        print y_true_cls
        print y_true_geo
        print y_pred_cls
        print y_pred_geo
        print training_mask
        classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.25
        print 'classfication_loss is ', classification_loss
        weights = tf.expand_dims((1.0 / (1e-5 + y_true_geo[:, :, :, 8])), axis=3) * y_true_cls * training_mask
        # only calculate the default point order
        # L_g = tf.reduce_mean(
        #     weights * smooth_l1_dist(y_true_cls * training_mask * (y_pred_geo - y_true_geo[:, :, :, :8])),
        #     reduction_indices=[1, 2, 3])
        # L_g = tf.reduce_mean(L_g)

        # loop all ordering, find the minimum ordering loss
        L_g_min = None
        for i in range(4):
            gt_geo = y_true_geo[:, :, :, :8]
            gt_geo = tf.concat([
                tf.expand_dims(gt_geo[:, :, :, (0 + i * 2) % 8], axis=3),
                tf.expand_dims(gt_geo[:, :, :, (1 + i * 2) % 8], axis=3),
                tf.expand_dims(gt_geo[:, :, :, (2 + i * 2) % 8], axis=3),
                tf.expand_dims(gt_geo[:, :, :, (3 + i * 2) % 8], axis=3),
                tf.expand_dims(gt_geo[:, :, :, (4 + i * 2) % 8], axis=3),
                tf.expand_dims(gt_geo[:, :, :, (5 + i * 2) % 8], axis=3),
                tf.expand_dims(gt_geo[:, :, :, (6 + i * 2) % 8], axis=3),
                tf.expand_dims(gt_geo[:, :, :, (7 + i * 2) % 8], axis=3)
            ], axis=-1)
            print gt_geo
            L_g = tf.reduce_mean(
                weights * smooth_l1_dist(y_true_cls * training_mask * (y_pred_geo - gt_geo[:, :, :, :8])),
                reduction_indices=[1, 2, 3])

            L_g = tf.reduce_mean(L_g)
            if L_g_min is None:
                L_g_min = L_g
            else:
                L_g_min = tf.minimum(L_g_min, L_g)
        L_s = classification_loss
        print 'Loss of geometry is ', L_g
        return L_g_min + L_s, L_g_min, L_s
    else:
        print 'Assign geometry error!'
        assert False

if __name__ == '__main__':
    input_tensor = tf.placeholder(tf.float32, [None, 512, 512, 3], 'x-input')

    # model(input_tensor)
    #
    # logits, end_points = build_pnasnet_large(input_tensor, 10, config=large_imagenet_config())
    # print('-'*15, 'PNASNet', '-'*15)
    # print(logits)
    # for key in end_points.keys():
    #     print(key, end_points[key])
    # print(end_points.keys())


    # logits, end_points = build_pnasnet_large(input_tensor, 10)
    # print('-' * 15, 'NASNet', '-' * 15)
    # print(logits)
    # for key in end_points.keys():
    #     print(key, end_points[key])
    # print(end_points.keys())

    # logits, end_points = resnet_v1_50(input_tensor, num_classes=10)
    logits, end_points = inception_resnet_v2(input_tensor, num_classes=10)
    print('-' * 15, 'ResNet', '-' * 15)
    print(logits)
    for key in end_points.keys():
        print(key, end_points[key])
    print(end_points.keys())