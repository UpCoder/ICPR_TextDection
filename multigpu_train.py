import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 10, '')
tf.app.flags.DEFINE_integer('num_readers', 16, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '1', '')
tf.app.flags.DEFINE_string('save_model_path', '/home/give/Game/OCR/Papers-code/EAST-Modify/tmp/east_icdar2015_resnet_v1_50_rbox_v7/', '')
tf.app.flags.DEFINE_boolean('restore', True, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_string('restore_path', '/home/give/Game/OCR/Papers-code/EAST-Modify/tmp/east_icdar2015_resnet_v1_50_rbox_v7/', '')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 100, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('summary_path', './log/', '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')
tf.app.flags.DEFINE_boolean('cross_loss', True, '')
tf.app.flags.DEFINE_boolean('using_instance_balanced', True, '')
from model import model_InceptionResNet
import model
from model import model_InceptionResNet_symmetry
import icdar


FLAGS = tf.app.flags.FLAGS

gpus = list(range(len(FLAGS.gpu_list.split(','))))


def tower_loss(images, score_maps, geo_maps, training_masks, weights_masks, reuse_variables=None):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        # f_score, f_geometry = model.model(images, is_training=True)
        f_score, f_geometry = model.model_InceptionResNet(images, is_training=True)
        # f_score, f_geometry = model.model_InceptionResNet_BLSTM(images, is_training=True)
        # f_score, f_geometry = model.model_InceptionResNet_symmetry(images, is_training=True)
    model_loss, L_g, L_s, L_s_c = model.loss(score_maps, f_score,
                            geo_maps, f_geometry,
                            training_masks, weights_masks)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    if reuse_variables is None:
        tf.summary.image('input', images)
        tf.summary.image('score_map', score_maps)
        tf.summary.image('score_map_pred', f_score * 255)
        tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:1])
        tf.summary.image('geo_map_0_pred', f_geometry[:, :, :, 0:1])
        tf.summary.image('training_masks', training_masks)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('geometry_loss', L_g)
        tf.summary.scalar('score_loss', L_s)

    return total_loss, model_loss, L_g, L_s, L_s_c


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        print grad_and_vars
        for g, _ in grad_and_vars:
            print 'g is ', g
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    # if not tf.gfile.Exists(FLAGS.checkpoint_path):
    #     tf.gfile.MakeDirs(FLAGS.checkpoint_path)
    # else:
    #     if not FLAGS.restore:
    #         tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
    #         tf.gfile.MkDir(FLAGS.checkpoint_path)

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
    if FLAGS.geometry == 'RBOX':
        input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
    else:
        input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 8], name='input_geo_maps')
    input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')
    input_weights_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=10000, decay_rate=0.90,
                                               staircase=True)
    # add summary
    # learning_rate = FLAGS.learning_rate
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)


    # split
    input_images_split = tf.split(input_images, len(gpus))
    input_score_maps_split = tf.split(input_score_maps, len(gpus))
    input_geo_maps_split = tf.split(input_geo_maps, len(gpus))
    input_training_masks_split = tf.split(input_training_masks, len(gpus))
    input_weights_masks_split = tf.split(input_weights_masks, len(gpus))

    tower_grads = []
    reuse_variables = None
    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                iis = input_images_split[i]
                isms = input_score_maps_split[i]
                igms = input_geo_maps_split[i]
                itms = input_training_masks_split[i]
                iwms = input_weights_masks_split[i]
                total_loss, model_loss, L_g, L_s, L_s_c = tower_loss(iis, isms, igms, itms, iwms, reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True

                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    tf.summary.scalar('average grads', tf.reduce_mean(grads[0][0]))
    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
    summary_writer = tf.summary.FileWriter(FLAGS.summary_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path, slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    start_step = 0
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if FLAGS.restore:
            sess.run(init)
            ckpt = tf.train.latest_checkpoint(FLAGS.restore_path)
            print('continue training from previous checkpoint from %s' % ckpt)
            start_step = int(os.path.basename(ckpt).split('-')[1])
            variable_restore_op = slim.assign_from_checkpoint_fn(ckpt,
                                                                 slim.get_trainable_variables(),
                                                                 ignore_missing_vars=True)
            variable_restore_op(sess)
            sess.run(tf.assign(global_step, start_step))
            # ckpt = tf.train.latest_checkpoint(FLAGS.restore_path)
            # print('continue training from previous checkpoint from %s' % ckpt)
            # start_step = int(os.path.basename(ckpt).split('-')[1])
            # saver.restore(sess, ckpt)
            # sess.run(tf.assign(global_step, start_step))
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                # print 'jump pretrained load'
                variable_restore_op(sess)

        data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                         input_size=FLAGS.input_size,
                                         batch_size=FLAGS.batch_size_per_gpu * len(gpus))

        start = time.time()
        for step in range(start_step, FLAGS.max_steps+start_step):
            data = next(data_generator)
            lr_value, ml, tl, geometry_loss_value, score_loss_value, score_cross_entropy_loss_value, _ = sess.run(
                [learning_rate, model_loss, total_loss, L_g, L_s, L_s_c, train_op], feed_dict={input_images: data[0],
                                                                                        input_score_maps: data[2],
                                                                                        input_geo_maps: data[3],
                                                                                        input_training_masks: data[4],
                                                                                        input_weights_masks: data[5]})
            if np.isnan(tl):
                print('Loss diverged, stop training')
                break

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start)/10
                avg_examples_per_second = (10 * FLAGS.batch_size_per_gpu * len(gpus))/(time.time() - start)
                start = time.time()
                print(
                    'Step {:06d}, model loss {:.4f}, total loss {:.4f}, geometry loss {:.4f}, score loss {:.4f}, score_cross_entropy_loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second, learning rate {:.7f}'.format(
                        step, ml, tl, geometry_loss_value, score_loss_value, score_cross_entropy_loss_value,
                        avg_time_per_step, avg_examples_per_second,
                        lr_value))

            if step % FLAGS.save_checkpoint_steps == 0 and step != 0:
                print 'save_model at ', FLAGS.save_model_path + 'model.ckpt'
                saver.save(sess, FLAGS.save_model_path + 'model.ckpt', global_step=global_step)

            if step % FLAGS.save_summary_steps == 0:
                print 'save summery at ', FLAGS.summary_path
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict={input_images: data[0],
                                                                                             input_score_maps: data[2],
                                                                                             input_geo_maps: data[3],
                                                                                             input_training_masks: data[4],
                                                                                             input_weights_masks: data[5]})
                summary_writer.add_summary(summary_str, global_step=step)

if __name__ == '__main__':
    tf.app.run()
