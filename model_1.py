# -*- coding=utf-8 -*-
# 相比于model.py，我们优化loss的时候多了一个branch，前者优化了pixel的分类结果、几何的loss。后者在前者的基础上，多了一个对整个预测得到bounding box计算的IoU的Smooth L1 loss
import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('text_scale', 512, '')
tf.app.flags.DEFINE_bool('uppool_conv', True, '')
tf.app.flags.DEFINE_boolean('OHEM_GEM', False, '')
tf.app.flags.DEFINE_boolean('BLSTM', True, '')
tf.app.flags.DEFINE_boolean('IoU_Loss', True, 'the flag determine weather computing the IoU loss branch and optimize it')
# tf.app.flags.DEFINE_integer('batch_size_per_gpu', 16, '')
# tf.app.flags.DEFINE_integer('cross_loss', False, '')
# tf.app.flags.DEFINE_integer('input_size', 512, '')
from nets import resnet_v1
from nets.Inception_ResNet_V2 import model as InceptionResNet
from cal_IoU_gt_py import cal_IoU_gt_py
import icdar
FLAGS = tf.app.flags.FLAGS

unpool_idx = 0
def unpool(inputs):
    if FLAGS.uppool_conv:
        print 'uppool: uppool_conv'
        global unpool_idx
        shape = inputs.get_shape().as_list()
        res = tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])
        res = slim.conv2d(res, num_outputs=shape[-1], kernel_size=[3, 3], stride=1, scope='unpool_' + str(unpool_idx))
        unpool_idx += 1
        return res
    else:
        print 'uppool: resized'
        return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])


def Bilstm(input, d_i, d_h, d_o, name, trainable=True, weight_decay=1e-5):
    def l2_regularizer(weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight_decay')
                #return tf.mul(l2_weight, tf.nn.l2_loss(tensor), name='value')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
        return regularizer
    def make_var(name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)
    img = input
    with tf.variable_scope(name) as scope:
        shape = tf.shape(img)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        img = tf.reshape(img, [N * H, W, C])
        img.set_shape([None, None, d_i])

        lstm_fw_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple=True)

        lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, img, dtype=tf.float32)
        lstm_out = tf.concat(lstm_out, axis=-1)

        lstm_out = tf.reshape(lstm_out, [N * H * W, 2*d_h])

        init_weights = tf.truncated_normal_initializer(stddev=0.1)
        init_biases = tf.constant_initializer(0.0)
        weights = make_var('weights', [2 * d_h, d_o], init_weights, trainable,
                           regularizer=l2_regularizer(weight_decay))
        biases = make_var('biases', [d_o], init_biases, trainable)
        outputs = tf.matmul(lstm_out, weights) + biases

        outputs = tf.reshape(outputs, [N, H, W, d_o])
        return outputs

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

def model_InceptionResNet_symmetry(images, weight_decay=1e-5, is_training=True):
    '''
    相对不对称的U-Net结构
    :param images:
    :param weight_decay:
    :param is_training:
    :return:
    '''
    images = mean_image_subtraction(images)
    logits, end_points = InceptionResNet.model(images, is_training=is_training, weight_decay=weight_decay)
    print 'end_points is ', end_points
    print 'end_points key are', end_points.keys()
    print 'logits is ', logits
    with tf.variable_scope('feature_fusion', values=[end_points.values]):
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
            f = [end_points['Scale-5'], end_points['Scale-4'],
                 end_points['Scale-3'], end_points['Scale-2'], end_points['Scale-1'], end_points['Scale-0']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))

            # Scale-5:
            output_channel = 256
            h = end_points['Scale-5']
            h = slim.conv2d(h, output_channel, kernel_size=3)
            h = slim.conv2d(tf.concat([h, end_points['Scale-5-2']], axis=-1), output_channel, 1)
            h = slim.conv2d(h, output_channel, kernel_size=3)
            h = slim.conv2d(tf.concat([h, end_points['Scale-5-1']], axis=-1), output_channel, 1)
            h = slim.conv2d(h, output_channel, kernel_size=3)
            h = slim.conv2d(tf.concat([h, end_points['Scale-5-0']], axis=-1), output_channel, 1)
            h = slim.conv2d(h, output_channel, kernel_size=3)
            g = unpool(h)

            # Scale-4
            output_channel = 128
            h = slim.conv2d(tf.concat([g, end_points['Scale-4']], axis=-1), output_channel, 1)
            h = slim.conv2d(h, output_channel, kernel_size=3)
            h = slim.conv2d(tf.concat([h, end_points['Scale-4-0']], axis=-1), output_channel, 1)
            h = slim.conv2d(h, output_channel, kernel_size=3)
            g = unpool(h)

            # Scale-3
            output_channel = 64
            h = slim.conv2d(tf.concat([g, end_points['Scale-3']], axis=-1), output_channel, 1)
            h = slim.conv2d(h, output_channel, kernel_size=3)
            h = slim.conv2d(tf.concat([h, end_points['Scale-3-1']], axis=-1), output_channel, 1)
            h = slim.conv2d(h, output_channel, kernel_size=3)
            h = slim.conv2d(tf.concat([h, end_points['Scale-3-0']], axis=-1), output_channel, 1)
            h = slim.conv2d(h, output_channel, kernel_size=3)
            g = unpool(h)

            # Scale-2
            output_channel = 32
            h = slim.conv2d(tf.concat([g, end_points['Scale-2']], axis=-1), output_channel, 1)
            h = slim.conv2d(h, output_channel, kernel_size=3)
            h = slim.conv2d(tf.concat([h, end_points['Scale-2-1']], axis=-1), output_channel, 1)
            h = slim.conv2d(h, output_channel, kernel_size=3)
            h = slim.conv2d(tf.concat([h, end_points['Scale-2-0']], axis=-1), output_channel, 1)
            h = slim.conv2d(h, output_channel, kernel_size=3)
            g = unpool(h)

            # Scale-1
            output_channel = 32
            h = slim.conv2d(tf.concat([g, end_points['Scale-1']], axis=-1), output_channel, 1)
            h = slim.conv2d(h, output_channel, kernel_size=3)
            h = slim.conv2d(tf.concat([h, end_points['Scale-1-1']], axis=-1), output_channel, 1)
            h = slim.conv2d(h, output_channel, kernel_size=3)
            h = slim.conv2d(tf.concat([h, end_points['Scale-1-0']], axis=-1), output_channel, 1)
            h = slim.conv2d(h, output_channel, kernel_size=3)
            g = unpool(h)

            # Scale-1
            output_channel = 32
            h = slim.conv2d(tf.concat([g, end_points['Scale-0']], axis=-1), output_channel, 1)
            h = slim.conv2d(h, output_channel, kernel_size=3)
            h = slim.conv2d(tf.concat([h, end_points['Scale-0-1']], axis=-1), output_channel, 1)
            h = slim.conv2d(h, output_channel, kernel_size=3)
            h = slim.conv2d(tf.concat([h, end_points['Scale-0-0']], axis=-1), output_channel, 1)
            h = slim.conv2d(h, output_channel, kernel_size=3)
            g = slim.conv2d(h, output_channel, 3)

            if is_training:
                if FLAGS.cross_loss:
                    print 'loss: cross_loss'
                    F_score = slim.conv2d(g, 1, 1, activation_fn=None, normalizer_fn=None)
                else:
                    print 'loss: IoU loss'
                    F_score = slim.conv2d(g, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            else:
                F_score = slim.conv2d(g, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            input_tensor_for_geo = tf.concat([g, F_score], axis=-1)

            input_tensor_for_geo = slim.conv2d(input_tensor_for_geo, 33, kernel_size=3, stride=1,
                                               activation_fn=tf.nn.relu, scope='merge_conv1')
            input_tensor_for_geo = slim.conv2d(input_tensor_for_geo, 33, kernel_size=3, stride=1,
                                               activation_fn=tf.nn.relu, scope='merge_conv2')
            geo_map = slim.conv2d(input_tensor_for_geo, 4, 1, activation_fn=tf.nn.sigmoid,
                                  normalizer_fn=None) * FLAGS.text_scale
            angle_map = (slim.conv2d(input_tensor_for_geo, 1, 1, activation_fn=tf.nn.sigmoid,
                                     normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return F_score, F_geometry

def model_InceptionResNet(images, weight_decay=1e-5, is_training=True):
    '''
    相对不对称的U-Net结构
    :param images:
    :param weight_decay:
    :param is_training:
    :return:
    '''
    images = mean_image_subtraction(images)
    logits, end_points = InceptionResNet.model(images, is_training=is_training, weight_decay=weight_decay)
    print 'end_points is ', end_points
    print 'end_points key are', end_points.keys()
    print 'logits is ', logits
    with tf.variable_scope('feature_fusion', values=[end_points.values]):
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
            f = [end_points['Scale-5'], end_points['Scale-4'],
                 end_points['Scale-3'], end_points['Scale-2'], end_points['Scale-1'], end_points['Scale-0']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None, None, None]
            h = [None, None, None, None, None, None]
            num_outputs = [None, 128, 64, 32, 32, 32]
            g_last = 6
            for i in range(g_last):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(tf.concat([g[i - 1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= g_last - 2:
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
            print 'g[', str(g_last - 1), '] shape is ', g[g_last - 1]
            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            if is_training:
                if FLAGS.cross_loss:
                    print 'loss: cross_loss'
                    F_score = slim.conv2d(g[g_last - 1], 1, 1, activation_fn=None, normalizer_fn=None)
                else:
                    print 'loss: IoU loss'
                    F_score = slim.conv2d(g[g_last - 1], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            else:
                F_score = slim.conv2d(g[g_last - 1], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            input_tensor_for_geo = tf.concat([g[g_last - 1], F_score], axis=-1)

            input_tensor_for_geo = slim.conv2d(input_tensor_for_geo, 33, kernel_size=3, stride=1,
                                               activation_fn=tf.nn.relu, scope='merge_conv1')
            input_tensor_for_geo = slim.conv2d(input_tensor_for_geo, 33, kernel_size=3, stride=1,
                                               activation_fn=tf.nn.relu, scope='merge_conv2')
            geo_map = slim.conv2d(input_tensor_for_geo, 4, 1, activation_fn=tf.nn.sigmoid,
                                  normalizer_fn=None) * FLAGS.text_scale
            angle_map = (slim.conv2d(input_tensor_for_geo, 1, 1, activation_fn=tf.nn.sigmoid,
                                     normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)

            input_tensor_for_IoU = tf.concat([g[g_last-1], F_score, F_geometry], axis=-1)
            input_tensor_for_IoU = slim.conv2d(input_tensor_for_IoU, 32, kernel_size=3, stride=1,
                                               activation_fn=tf.nn.relu, scope='merge_conv3')
            input_tensor_for_IoU = slim.conv2d(input_tensor_for_IoU, 32, kernel_size=3, stride=1,
                                               activation_fn=tf.nn.relu, scope='merge_conv4')
            IoU_map = slim.conv2d(input_tensor_for_IoU, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            F_IoU = IoU_map
    return F_score, F_geometry, F_IoU



def model_InceptionResNet_BLSTM(images, weight_decay=1e-5, is_training=True):
    '''
    相对不对称的U-Net结构, 并在Inception提出特征后,使用BLSTM提取全局的一个特征
    :param images:
    :param weight_decay:
    :param is_training:
    :return:
    '''
    images = mean_image_subtraction(images)
    logits, end_points = InceptionResNet.model(images, is_training=is_training, weight_decay=weight_decay)
    print 'end_points is ', end_points
    print 'end_points key are', end_points.keys()
    print 'logits is ', logits
    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
        }
        # 提取Global的特征
        # end_points['Scale-5'] = Bilstm(end_points['Scale-5'], d_i=1536, d_h=256, d_o=256, name='Bilstm',
        #                                weight_decay=weight_decay)
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['Scale-5'], end_points['Scale-4'],
                 end_points['Scale-3'], end_points['Scale-2'], end_points['Scale-1'], end_points['Scale-0']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None, None, None]
            h = [None, None, None, None, None, None]
            num_inputs = [1536, 1088, 320, 192, 64, 64]
            num_outputs = [None, 128, 64, 32, 32, 32]
            g_last = 6
            for i in range(g_last):
                if i == 0:
                    h[i] = f[i]
                else:
                    f[i] = slim.conv2d(f[i], num_outputs=num_outputs[i], kernel_size=1)
                    f[i] = Bilstm(f[i], num_outputs[i], num_outputs[i] * 2, num_outputs[i], name='Bilstm-' + str(i))
                    c1_1 = slim.conv2d(tf.concat([g[i - 1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= g_last - 2:
                    g[i] = unpool(h[i])
                    if i == 0:
                        continue
                    # g[i] = Bilstm(g[i], num_outputs[i], num_outputs[i] * 2, num_outputs[i], name='Bilstm-' + str(i))
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
            print 'g[', str(g_last - 1), '] shape is ', g[g_last - 1]
            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            if is_training:
                if FLAGS.cross_loss:
                    print 'loss: cross_loss'
                    F_score = slim.conv2d(g[g_last - 1], 1, 1, activation_fn=None, normalizer_fn=None)
                else:
                    print 'loss: IoU loss'
                    F_score = slim.conv2d(g[g_last - 1], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            else:
                F_score = slim.conv2d(g[g_last - 1], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            input_tensor_for_geo = tf.concat([g[g_last - 1], F_score], axis=-1)

            input_tensor_for_geo = slim.conv2d(input_tensor_for_geo, 33, kernel_size=3, stride=1,
                                               activation_fn=tf.nn.relu, scope='merge_conv1')
            input_tensor_for_geo = slim.conv2d(input_tensor_for_geo, 33, kernel_size=3, stride=1,
                                               activation_fn=tf.nn.relu, scope='merge_conv2')
            geo_map = slim.conv2d(input_tensor_for_geo, 4, 1, activation_fn=tf.nn.sigmoid,
                                  normalizer_fn=None) * FLAGS.text_scale
            angle_map = (slim.conv2d(input_tensor_for_geo, 1, 1, activation_fn=tf.nn.sigmoid,
                                     normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return F_score, F_geometry

def model(images, weight_decay=1e-5, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')
    print 'end_points is ', end_points
    print 'end_points key are', end_points.keys()
    print 'logits is ', logits
    with tf.variable_scope('feature_fusion', values=[end_points.values]):
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
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2'], end_points['conv1_2'], end_points['conv0_2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None, None, None]
            h = [None, None, None, None, None, None]
            num_outputs = [None, 128, 64, 32, 32, 32]
            g_last = 6
            for i in range(g_last):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= g_last-2:
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
            print 'g[', str(g_last-1), '] shape is ', g[g_last-1]
            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            if is_training:
                if FLAGS.cross_loss:
                    print 'loss: cross_loss'
                    F_score = slim.conv2d(g[g_last-1], 1, 1, activation_fn=None, normalizer_fn=None)
                else:
                    print 'loss: IoU loss'
                    F_score = slim.conv2d(g[g_last - 1], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            else:
                F_score = slim.conv2d(g[g_last-1], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            input_tensor_for_geo = tf.concat([g[g_last-1], F_score], axis=-1)
            geo_map = slim.conv2d(input_tensor_for_geo, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
            angle_map = (slim.conv2d(input_tensor_for_geo, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)

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


# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def cross_entropy_EAST(y_true_cls, y_pred_cls, training_mask):
    '''
    计算交叉熵, EAST中提出的balanced cross_entropy loss
    :param y_true_cls: numpy array
    :param y_pred_cls: numpy array
    :param training_mask: numpy array
    :return:　标量的tensor
    '''
    # eps = 1e-10
    # y_pred_cls = y_pred_cls * training_mask + eps
    # y_true_cls = y_true_cls * training_mask + eps
    # shape = list(np.shape(y_true_cls))
    # beta = 1 - (np.sum(np.reshape(y_true_cls, [shape[0], -1]), axis=1) / (1.0 * shape[1] * shape[2]))
    # cross_entropy_loss = -beta * y_true_cls * np.log(y_pred_cls) - (1 - beta) * (1 - y_true_cls) * np.log(
    #     1 - y_pred_cls)
    # return np.mean(cross_entropy_loss)
    eps = 1e-5
    # y_pred_cls = y_pred_cls * training_mask + eps
    # y_true_cls = y_true_cls * training_mask + eps
    each_y_true_sample = tf.split(y_true_cls, num_or_size_splits=FLAGS.batch_size_per_gpu, axis=0)
    each_y_pred_sample = tf.split(y_pred_cls, num_or_size_splits=FLAGS.batch_size_per_gpu, axis=0)
    loss_maps = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_cls, logits=y_pred_cls)
    each_loss_sample = tf.split(loss_maps, num_or_size_splits=FLAGS.batch_size_per_gpu, axis=0)
    loss = None
    for i in range(FLAGS.batch_size_per_gpu):
        cur_true = each_y_true_sample[i]
        cur_pred = each_y_pred_sample[i]
        beta = 1 - (tf.reduce_sum(cur_true) / (FLAGS.input_size * FLAGS.input_size))
        # cur_loss = -beta * cur_true * tf.log(cur_pred) - (1-beta) * (1-cur_true) * tf.log((1-cur_pred))
        cur_loss = each_loss_sample[i]
        if loss is None:

            loss = cur_loss
            print 'cur_loss is ', loss
        else:
            loss = tf.concat([loss, cur_loss], axis=0)
    return loss


def calculate_OHEM_mask_py(y_true_cls, loss_map, only_negative):
    '''

    :param y_true_cls:　ground truth
    :param y_pred_cls: 预测值
    :param training_mask: mask文件
    :param weights_mask: 计算交叉熵时候每个ｐｉｘｅｌ的权重
    :return:
    '''
    DEBUG = False
    if DEBUG:
        print 'ok 0-1'
    weighted_cross_entropy_loss = loss_map
    if DEBUG:
        print 'ok 0-2'
    batch_size, h, w, c = np.shape(y_true_cls)
    if DEBUG:
        print 'ok 0-3'
    if DEBUG:
        print batch_size, h, w, c, np.shape(loss_map)
    if DEBUG:
        print 'ok 0-4'
    OHEM_mask = np.zeros([batch_size, h, w], np.uint8)
    for batch_id in range(batch_size):
        if DEBUG:
            print 'ok 1'
        cur_OHEM_mask = np.zeros([h, w], np.uint8)
        if DEBUG:
            print 'ok 2'
        cur_weighted_cross_loss = np.asarray(weighted_cross_entropy_loss[batch_id], np.float32)
        if DEBUG:
            print 'ok 3'
        cur_y_true_cls = y_true_cls[batch_id]
        if DEBUG:
            print 'ok 4'
        S = np.sum(cur_y_true_cls)
        if DEBUG:
            print 'ok 5'

        negative_num = min(3*S, np.sum(cur_y_true_cls == 0.0))
        if DEBUG:
            print 'ok 6'
        indices = np.transpose(np.where(cur_y_true_cls == 0))
        if DEBUG:
            print 'ok 7', np.shape(cur_y_true_cls), np.shape(cur_weighted_cross_loss)
        target_loss_value = cur_weighted_cross_loss[cur_y_true_cls == 0]
        if DEBUG:
            print 'ok 8'
        indices_sorted = np.asarray(np.argsort(target_loss_value), np.int32)
        if DEBUG:
            print 'ok 9'
        indices_sorted = indices_sorted[:int(negative_num)]
        if DEBUG:
            print 'ok 10'
        indices_sorted = indices[indices_sorted]
        if DEBUG:
            print 'ok 11'
        if not only_negative:
            # 是否只挑选正样本
            cur_OHEM_mask[np.squeeze(cur_y_true_cls) == 1] = 1
        if DEBUG:
            print 'ok 12'
        cur_OHEM_mask[indices_sorted] = 1
        if DEBUG:
            print 'ok 13'
        OHEM_mask[batch_id] = cur_OHEM_mask
        if DEBUG:
            print 'ok 14'
    if DEBUG:
        print 'ok 15', np.shape(OHEM_mask), np.shape(y_true_cls)
    return OHEM_mask


def cross_entropy_PixelLink(y_true_cls, y_pred_cls, training_mask, weights_mask):
    '''
    计算交叉熵　ＰｉｘｅｌＬｉｎｋ中的方法
    :param y_true_cls:　ground truth
    :param y_pred_cls: 预测值
    :param training_mask: mask文件
    :param weights_mask: 计算交叉熵时候每个ｐｉｘｅｌ的权重, instance_balanced mask
    :return:　数组形式的tensor，代表每个ｐｉｘｅｌ的交叉熵
    '''
    eps = 1e-5
    y_pred_cls = y_pred_cls
    y_true_cls = y_true_cls
    y_true_cls_tmp = tf.one_hot(tf.squeeze(tf.cast(y_true_cls, tf.uint8), axis=[3]), axis=3, depth=2)
    y_pred_cls_tmp = tf.concat([1 - y_pred_cls, y_pred_cls], axis=3)
    print 'y true cls tmp: ', y_true_cls_tmp
    print 'y_pred_cls_tmp: ', y_pred_cls_tmp
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_true_cls_tmp,
        logits=y_pred_cls_tmp)
    cross_entropy_loss = tf.expand_dims(cross_entropy_loss, 3)
    cross_entropy_loss = cross_entropy_loss * training_mask * weights_mask
    return tf.reduce_mean(cross_entropy_loss) * 10
    # print cross_entropy_loss
    # weighted_cross_entropy_loss = training_mask * weights_mask * tf.expand_dims(cross_entropy_loss, axis=3)

    [OHEM_mask_score] = tf.py_func(calculate_OHEM_mask_py, [y_true_cls, cross_entropy_loss, True], [tf.uint8])
    batch_size, h, w, _ = y_true_cls.get_shape().as_list()
    OHEM_mask_score.set_shape(y_true_cls.get_shape())
    OHEM_mask_score = tf.cast(OHEM_mask_score, tf.float32)
    # OHEM_mask_score = tf.expand_dims(OHEM_mask_score, axis=3)
    print 'OHEM mask score: ', OHEM_mask_score
    print 'y_true_cls: ', y_true_cls
    print 'weights mask: ', weights_mask
    print 'cross entropy loss: ', cross_entropy_loss
    each_y_true_sample = tf.split(y_true_cls, num_or_size_splits=FLAGS.batch_size_per_gpu, axis=0)
    each_OHEM_mask_score = tf.split(OHEM_mask_score, num_or_size_splits=FLAGS.batch_size_per_gpu, axis=0)
    each_weights_mask = tf.split(weights_mask, num_or_size_splits=FLAGS.batch_size_per_gpu, axis=0)
    each_entropy_loss = tf.split(cross_entropy_loss, num_or_size_splits=FLAGS.batch_size_per_gpu, axis=0)
    mean_entropy_loss = None
    for idx in range(FLAGS.batch_size_per_gpu):
        cur_OHEM_mask = each_OHEM_mask_score[idx]
        cur_y_true = each_y_true_sample[idx]
        cur_weights_mask = each_weights_mask[idx]
        cur_cross_entropy = each_entropy_loss[idx]
        print 'cur_OHEM_mask', cur_OHEM_mask
        print 'cur_y_true', cur_y_true
        print 'cur_weights_mask', cur_weights_mask
        print 'cur_cross_entropy', cur_cross_entropy
        cur_loss = tf.reduce_mean((cur_OHEM_mask * 2.0 + cur_weights_mask) * cur_cross_entropy)
        divided = ((1 + 3.0) * tf.reduce_sum(cur_y_true)) + eps
        cur_loss = cur_loss / divided
        # cur_loss = tf.reduce_sum(cur_cross_entropy)
        if mean_entropy_loss is None:
            mean_entropy_loss = cur_loss
        else:
            mean_entropy_loss = mean_entropy_loss + cur_loss
    return mean_entropy_loss
    # return tf.reduce_mean(weighted_cross_entropy_loss)

# 计算smooth L1 loss
def smooth_l1_dist(deltas, sigma2=9.0, name='smooth_l1_dist'):
    '''
    :param deltas: pred-GT
    :param sigma2:
    :param name:
    :return:
    '''
    with tf.name_scope(name=name) as scope:
        deltas_abs = tf.abs(deltas)
        smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
        return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                    (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)


def loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo, bbox_gt, pred_IoU,
         training_mask, weights_masks):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param bbox_gt: the point of ground truth bouning box, N, M, 4, 2
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    # loss = 1 - 2.0*(intersection / union)
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.05

    # classification_loss = py_func(cross_entropy_py, inp=[y_true_cls, y_pred_cls, training_mask], Tout=tf.float32)
    L_s_c = cross_entropy_PixelLink(y_true_cls, y_pred_cls, training_mask, weights_masks)
    # L_s_c *= 20
    # OHEM_mask_score = tf.cast(tf.py_func(calculate_OHEM_mask_py, [y_true_cls, classification_loss], tf.uint8), tf.float32)
    # classification_loss = tf.expand_dims(OHEM_mask_score, axis=3) * classification_loss
    # classification_loss = tf.reduce_mean(classification_loss) * 0.1

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
    OHEM_mask_geo = tf.cast(tf.py_func(calculate_OHEM_mask_py, [y_true_cls, L_g, False], tf.uint8), tf.float32)
    print 'OHEM_mask_geo: ', OHEM_mask_geo
    if FLAGS.OHEM_GEM:
        print 'using OHEM GEM'
        L_g = tf.reduce_mean(L_g * y_true_cls * training_mask * tf.expand_dims(OHEM_mask_geo, axis=3))
    else:
        print 'not using OHEM GEM'
        L_g = tf.reduce_mean(L_g * y_true_cls * training_mask)


    IoU_gt = tf.py_func(cal_IoU_gt_py, (y_pred_geo, y_pred_cls, bbox_gt, 0.7, ), tf.float32)
    L_I = tf.reduce_mean(smooth_l1_dist((pred_IoU-IoU_gt) * training_mask))


    L_s = classification_loss
    L_total = L_g + L_s + L_s_c + L_I
    return L_total, L_g, L_s, L_s_c, L_I


if __name__ == '__main__':
    # 测试model
    image_tensor = tf.placeholder(tf.float32, [None, 512, 512, 3])
    model_InceptionResNet(image_tensor)

    # 测试loss
    # y_pred_cls = tf.placeholder(tf.float32, [None, 512, 512, 1])
    # y_true_cls = tf.placeholder(tf.float32, [None, 512, 512, 1])
    # training_mask = tf.placeholder(tf.float32, [None, 512, 512, 1])
    # weights_mask = tf.placeholder(tf.float32, [None, 512, 512, 1])
    # y_pred_geo = tf.placeholder(tf.float32, [None, 512, 512, 5])
    # y_true_geo = tf.placeholder(tf.float32, [None, 512, 512, 5])
    # loss(y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask, weights_mask)


    # 测试OHEM
    # y_pred_cls = tf.placeholder(tf.float32, [None, 512, 512, 1])
    # y_true_cls = tf.placeholder(tf.float32, [None, 512, 512, 1])
    # weights_mask = tf.placeholder(tf.float32, [None, 512, 512, 1])
    # training_mask = tf.placeholder(tf.float32, [None, 512, 512, 1])
    # OHEM_mask = tf.py_func(calculate_OHEM_mask_py, [y_true_cls, y_pred_cls, weights_mask, training_mask], tf.uint8)
    # cross_entropy_PixelLink(y_true_cls, y_pred_cls, training_mask, weights_mask, OHEM_mask)
    # data_generator = icdar.get_batch(num_workers=1,
    #                                  input_size=512,
    #                                  batch_size=5)
    # data = next(data_generator)
    # y_score_map = data[2]
    # training_mask_value = data[4]
    # weights_mask_value = np.ones(np.shape(training_mask_value))
    # y_pred_value = np.random.random(np.shape(y_score_map))
    # print np.shape(y_pred_value)
    # print np.shape(y_score_map), np.max(y_score_map), np.min(y_score_map)
    # print np.shape(training_mask_value)
    # print np.shape(weights_mask_value)
    # # calculate_OHEM_mask_py(y_score_map, y_pred_value, training_mask_value, weights_mask_value)
    # sess = tf.Session()
    # feed_dict = {
    #     y_pred_cls: y_pred_value,
    #     y_true_cls: y_score_map,
    #     weights_mask: weights_mask_value,
    #     training_mask: traini