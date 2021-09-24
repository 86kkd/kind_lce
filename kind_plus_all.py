# coding: utf-8
from __future__ import print_function

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
import random
import argparse

from utils import *
from model import *
from glob import glob

import tensorflow._api.v2.compat.v1 as tf


def mutual_i_loss(input_I_low, input_I_high):
    low_gradient_x = gradient(input_I_low, "x")
    high_gradient_x = gradient(input_I_high, "x")
    x_loss = (low_gradient_x + high_gradient_x) * tf.exp(-10 * (low_gradient_x + high_gradient_x))
    low_gradient_y = gradient(input_I_low, "y")
    high_gradient_y = gradient(input_I_high, "y")
    y_loss = (low_gradient_y + high_gradient_y) * tf.exp(-10 * (low_gradient_y + high_gradient_y))
    mutual_loss = tf.reduce_mean(x_loss + y_loss)
    return mutual_loss


def mutual_i_input_loss(input_I_low, input_im):
    input_gray = tf.image.rgb_to_grayscale(input_im)
    low_gradient_x = gradient(input_I_low, "x")
    input_gradient_x = gradient(input_gray, "x")
    x_loss = tf.abs(tf.div(low_gradient_x, tf.maximum(input_gradient_x, 0.01)))
    low_gradient_y = gradient(input_I_low, "y")
    input_gradient_y = gradient(input_gray, "y")
    y_loss = tf.abs(tf.div(low_gradient_y, tf.maximum(input_gradient_y, 0.01)))
    mut_loss = tf.reduce_mean(x_loss + y_loss)
    return mut_loss


def run_decomposition_net(args_param: argparse.Namespace):
    os.environ['CUDA_VISIBLE_DEVICES'] = args_param.cuda
    learning_rate = args_param.decom_learning_rate
    epoch = args_param.decom_epoch
    eval_every_epoch = args_param.decom_eval_every_epoch
    batch_size = args_param.decom_batch_size
    patch_size = args_param.decom_patch_size
    data = args_param.train_data_dir
    sess = tf.Session()

    input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
    input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

    [R_low, I_low] = DecomNet(input_low)
    [R_high, I_high] = DecomNet(input_high)

    I_low_3 = tf.concat([I_low, I_low, I_low], axis=3)
    I_high_3 = tf.concat([I_high, I_high, I_high], axis=3)

    # network output
    output_R_low = R_low
    output_R_high = R_high
    output_I_low = I_low_3
    output_I_high = I_high_3

    # Loss
    recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3 - input_low))
    recon_loss_high = tf.reduce_mean(tf.abs(R_high * I_high_3 - input_high))

    equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_high))

    i_mutual_loss = mutual_i_loss(I_low, I_high)

    i_input_mutual_loss_high = mutual_i_input_loss(I_high, input_high)
    i_input_mutual_loss_low = mutual_i_input_loss(I_low, input_low)

    loss_Decom = 1 * recon_loss_high + 1 * recon_loss_low \
                 + 0.009 * equal_R_loss + 0.2 * i_mutual_loss \
                 + 0.15 * i_input_mutual_loss_high + 0.15 * i_input_mutual_loss_low

    lr = tf.placeholder(tf.float32, name='learning_rate')

    optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')
    var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
    train_op_Decom = optimizer.minimize(loss_Decom, var_list=var_Decom)
    sess.run(tf.global_variables_initializer())

    saver_Decom = tf.train.Saver(var_list=var_Decom)
    print("[*] Initialize model successfully...")
    # load data
    # train_data
    train_decom_low_data = []
    train_decom_high_data = []
    train_decom_low_data_names = glob(os.path.join(data, 'low/*.png'))
    train_decom_low_data_names.sort()
    train_decom_high_data_names = glob(os.path.join(data, 'high/*.png'))
    train_decom_high_data_names.sort()
    assert len(train_decom_low_data_names) == len(train_decom_high_data_names)
    print('[*] Number of training data: %d' % len(train_decom_low_data_names))
    for idx in range(len(train_decom_low_data_names)):
        low_im = load_images(train_decom_low_data_names[idx])
        train_decom_low_data.append(low_im)
        high_im = load_images(train_decom_high_data_names[idx])
        train_decom_high_data.append(high_im)

    # eval_data
    eval_decom_low_data = []
    eval_decom_high_data = []
    eval_decom_low_data_name = glob(os.path.join(data, 'eval15/low/*.png'))
    eval_decom_low_data_name.sort()
    eval_decom_high_data_name = glob(os.path.join(data, 'eval15/high/*.png'))
    eval_decom_high_data_name.sort()
    for idx in range(len(eval_decom_low_data_name)):
        eval_low_im = load_images(eval_decom_low_data_name[idx])
        eval_decom_low_data.append(eval_low_im)
        eval_high_im = load_images(eval_decom_high_data_name[idx])
        eval_decom_high_data.append(eval_high_im)
    sample_dir = os.path.join(args_param.sample_dir, 'decom_net_train_result')
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)

    train_phase = 'decomposition'
    numBatch = len(train_decom_low_data) // int(batch_size)
    train_op = train_op_Decom
    train_loss = loss_Decom
    saver = saver_Decom

    checkpoint_dir = os.path.join(args_param.checkpoint_dir, 'decom_net_retrain/')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('No decomnet pretrained model!')

    start_step = 0
    start_epoch = 0
    iter_num = 0

    print(
        "[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))
    start_time = time.time()
    image_id = 0

    for epoch in range(start_epoch, epoch):
        for batch_id in range(start_step, numBatch):
            batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
            batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
            for patch_id in range(batch_size):
                h, w, _ = train_decom_low_data[image_id].shape
                x = random.randint(0, h - patch_size)
                y = random.randint(0, w - patch_size)
                rand_mode = random.randint(0, 7)
                batch_input_low[patch_id, :, :, :] = data_augmentation(
                    train_decom_low_data[image_id][x: x + patch_size, y: y + patch_size, :], rand_mode)
                batch_input_high[patch_id, :, :, :] = data_augmentation(
                    train_decom_high_data[image_id][x: x + patch_size, y: y + patch_size, :], rand_mode)
                image_id = (image_id + 1) % len(train_decom_low_data)
                if image_id == 0:
                    tmp = list(zip(train_decom_low_data, train_decom_high_data))
                    random.shuffle(tmp)
                    train_decom_low_data, train_decom_high_data = zip(*tmp)

            _, loss = sess.run([train_op, train_loss], feed_dict={input_low: batch_input_low, \
                                                                  input_high: batch_input_high, \
                                                                  lr: learning_rate})
            print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                  % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
            iter_num += 1
        if (epoch + 1) % eval_every_epoch == 0:
            print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch + 1))
            for idx in range(len(eval_decom_low_data)):
                input_low_eval = np.expand_dims(eval_decom_low_data[idx], axis=0)
                result_1, result_2 = sess.run([output_R_low, output_I_low], feed_dict={input_low: input_low_eval})
                save_images(os.path.join(sample_dir, 'low_%d_%d.png' % (idx + 1, epoch + 1)), result_1, result_2)
            for idx in range(len(eval_decom_low_data)):
                input_low_eval = np.expand_dims(eval_decom_high_data[idx], axis=0)
                result_11, result_22 = sess.run([output_R_high, output_I_high], feed_dict={input_high: input_low_eval})
                save_images(os.path.join(sample_dir, 'high_%d_%d.png' % (idx + 1, epoch + 1)), result_11, result_22)

        saver.save(sess, checkpoint_dir + 'model_%d.ckpt' % (epoch + 1))

    print("[*] Finish training for phase %s." % train_phase)


def grad_loss(input_i_low, input_i_high):
    x_loss = tf.square(gradient_no_abs(input_i_low, 'x') - gradient_no_abs(input_i_high, 'x'))
    y_loss = tf.square(gradient_no_abs(input_i_low, 'y') - gradient_no_abs(input_i_high, 'y'))
    grad_loss_all = tf.reduce_mean(x_loss + y_loss)
    return grad_loss_all


def run_illumination_adjustment_net(args_param: argparse.Namespace):
    os.environ['CUDA_VISIBLE_DEVICES'] = args_param.cuda
    batch_size = args_param.illumin_batch_size
    patch_size = args_param.illumin_patch_size
    data = args_param.train_data_dir
    learning_rate = args_param.illumin_learning_rate
    epoch = args_param.illumin_epoch
    eval_every_epoch = args_param.illumin_eval_every_epoch

    sess = tf.Session()
    # the input of decomposition net
    input_decom = tf.placeholder(tf.float32, [None, None, None, 3], name='input_decom')
    # the input of illumination adjustment net
    input_low_i = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_i')
    input_low_i_ratio = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_i_ratio')
    input_high_i = tf.placeholder(tf.float32, [None, None, None, 1], name='input_high_i')

    [R_decom, I_decom] = DecomNet(input_decom)
    # the output of decomposition network
    decom_output_R = R_decom
    decom_output_I = I_decom
    # the output of illumination adjustment net
    output_i = Illumination_adjust_net(input_low_i, input_low_i_ratio)

    # define loss
    loss_grad = grad_loss(output_i, input_high_i)
    loss_square = tf.reduce_mean(tf.square(output_i - input_high_i))

    loss_adjust = loss_square + loss_grad

    lr = tf.placeholder(tf.float32, name='learning_rate')

    optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')

    var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
    var_adjust = [var for var in tf.trainable_variables() if 'I_enhance_Net' in var.name]

    saver_adjust = tf.train.Saver(var_list=var_adjust)
    saver_Decom = tf.train.Saver(var_list=var_Decom)
    train_op_adjust = optimizer.minimize(loss_adjust, var_list=var_adjust)
    sess.run(tf.global_variables_initializer())
    print("[*] Initialize model successfully...")

    ### load data
    ### Based on the decomposition net, we first get the decomposed reflectance maps
    ### and illumination maps, then train the adjust net.
    ### train_data
    train_illumin_low_data = []
    train_illumin_high_data = []
    train_illumin_low_data_names = glob(os.path.join(data, 'our485/low/*.png'))
    train_illumin_low_data_names.sort()
    train_illumin_high_data_names = glob(os.path.join(data, 'our485/high/*.png'))
    train_illumin_high_data_names.sort()
    assert len(train_illumin_low_data_names) == len(train_illumin_high_data_names)
    print('[*] Number of training data: %d' % len(train_illumin_low_data_names))
    for idx in range(len(train_illumin_low_data_names)):
        low_im = load_images(train_illumin_low_data_names[idx])
        train_illumin_low_data.append(low_im)
        high_im = load_images(train_illumin_high_data_names[idx])
        train_illumin_high_data.append(high_im)

    pre_decom_checkpoint_dir = './checkpoint/decom_model/'
    ckpt_pre = tf.train.get_checkpoint_state(pre_decom_checkpoint_dir)
    if ckpt_pre:
        print('loaded ' + ckpt_pre.model_checkpoint_path)
        saver_Decom.restore(sess, ckpt_pre.model_checkpoint_path)
    else:
        print('No pre_decom_net checkpoint!')

    decomposed_low_i_data_480 = []
    decomposed_high_i_data_480 = []
    for idx in range(len(train_illumin_low_data)):
        input_low = np.expand_dims(train_illumin_low_data[idx], axis=0)
        RR, II = sess.run([decom_output_R, decom_output_I], feed_dict={input_decom: input_low})
        RR0 = np.squeeze(RR)
        II0 = np.squeeze(II)
        print(RR0.shape, II0.shape)
        # decomposed_high_r_data_480.append(result_1_sq)
        decomposed_low_i_data_480.append(II0)
    for idx in range(len(train_illumin_high_data)):
        input_high = np.expand_dims(train_illumin_high_data[idx], axis=0)
        RR2, II2 = sess.run([decom_output_R, decom_output_I], feed_dict={input_decom: input_high})
        RR02 = np.squeeze(RR2)
        II02 = np.squeeze(II2)
        print(RR02.shape, II02.shape)
        # decomposed_high_r_data_480.append(result_1_sq)
        decomposed_high_i_data_480.append(II02)

    eval_adjust_low_i_data = decomposed_low_i_data_480[451:480]
    eval_adjust_high_i_data = decomposed_high_i_data_480[451:480]

    train_adjust_low_i_data = decomposed_low_i_data_480[0:450]
    train_adjust_high_i_data = decomposed_high_i_data_480[0:450]

    print('[*] Number of training data: %d' % len(train_adjust_high_i_data))

    train_phase = 'adjustment'
    numBatch = len(train_adjust_low_i_data) // int(batch_size)
    train_op = train_op_adjust
    train_loss = loss_adjust
    saver = saver_adjust

    checkpoint_dir = os.path.join(args_param.checkpoint_dir, 'illumination_adjust_net_retrain')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("No adjustment net pre model!")

    start_step = 0
    start_epoch = 0
    iter_num = 0
    print(
        "[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

    sample_dir = os.path.join(args_param.sample_dir, 'illumination_adjust_net_train/')
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)

    start_time = time.time()
    image_id = 0

    for epoch in range(start_epoch, epoch):
        for batch_id in range(start_step, numBatch):
            batch_input_low_i_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
            batch_input_high_i_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
            batch_input_low_i = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
            batch_input_high_i = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
            input_low_i_rand = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
            input_high_i_rand = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
            input_low_i_rand_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
            input_high_i_rand_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")

            for patch_id in range(batch_size):
                i_low_data = train_adjust_low_i_data[image_id]
                i_low_expand = np.expand_dims(i_low_data, axis=2)
                i_high_data = train_adjust_high_i_data[image_id]
                i_high_expand = np.expand_dims(i_high_data, axis=2)

                h, w = train_adjust_low_i_data[image_id].shape
                x = random.randint(0, h - patch_size)
                y = random.randint(0, w - patch_size)
                i_low_data_crop = i_low_expand[x: x + patch_size, y: y + patch_size, :]
                i_high_data_crop = i_high_expand[x: x + patch_size, y: y + patch_size, :]

                rand_mode = np.random.randint(0, 7)
                batch_input_low_i[patch_id, :, :, :] = data_augmentation(i_low_data_crop, rand_mode)
                batch_input_high_i[patch_id, :, :, :] = data_augmentation(i_high_data_crop, rand_mode)

                ratio = np.mean(i_low_data_crop / (i_high_data_crop + 0.0001))
                # print(ratio)
                i_low_data_ratio = np.ones([patch_size, patch_size]) * (1 / ratio + 0.0001)
                i_low_ratio_expand = np.expand_dims(i_low_data_ratio, axis=2)
                i_high_data_ratio = np.ones([patch_size, patch_size]) * (ratio)
                i_high_ratio_expand = np.expand_dims(i_high_data_ratio, axis=2)
                batch_input_low_i_ratio[patch_id, :, :, :] = i_low_ratio_expand
                batch_input_high_i_ratio[patch_id, :, :, :] = i_high_ratio_expand

                rand_mode = np.random.randint(0, 2)
                if rand_mode == 1:
                    input_low_i_rand[patch_id, :, :, :] = batch_input_low_i[patch_id, :, :, :]
                    input_high_i_rand[patch_id, :, :, :] = batch_input_high_i[patch_id, :, :, :]
                    input_low_i_rand_ratio[patch_id, :, :, :] = batch_input_low_i_ratio[patch_id, :, :, :]
                    input_high_i_rand_ratio[patch_id, :, :, :] = batch_input_high_i_ratio[patch_id, :, :, :]
                else:
                    input_low_i_rand[patch_id, :, :, :] = batch_input_high_i[patch_id, :, :, :]
                    input_high_i_rand[patch_id, :, :, :] = batch_input_low_i[patch_id, :, :, :]
                    input_low_i_rand_ratio[patch_id, :, :, :] = batch_input_high_i_ratio[patch_id, :, :, :]
                    input_high_i_rand_ratio[patch_id, :, :, :] = batch_input_low_i_ratio[patch_id, :, :, :]

                image_id = (image_id + 1) % len(train_adjust_low_i_data)

            _, loss = sess.run([train_op, train_loss],
                               feed_dict={input_low_i: input_low_i_rand, input_low_i_ratio: input_low_i_rand_ratio, \
                                          input_high_i: input_high_i_rand, \
                                          lr: learning_rate})
            print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                  % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
            iter_num += 1
        if (epoch + 1) % eval_every_epoch == 0:
            print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch + 1))

            for idx in range(10):
                rand_idx = idx  # np.random.randint(26)
                input_uu_i = eval_adjust_low_i_data[rand_idx]
                input_low_eval_i = np.expand_dims(input_uu_i, axis=0)
                input_low_eval_ii = np.expand_dims(input_low_eval_i, axis=3)
                h, w = eval_adjust_low_i_data[idx].shape
                rand_ratio = np.random.random(1) * 5
                input_uu_i_ratio = np.ones([h, w]) * rand_ratio
                input_low_eval_i_ratio = np.expand_dims(input_uu_i_ratio, axis=0)
                input_low_eval_ii_ratio = np.expand_dims(input_low_eval_i_ratio, axis=3)

                result_1 = sess.run(output_i,
                                    feed_dict={input_low_i: input_low_eval_ii,
                                               input_low_i_ratio: input_low_eval_ii_ratio})
                save_images(os.path.join(sample_dir, 'h_eval_%d_%d_%5f.png' % (epoch + 1, rand_idx + 1, rand_ratio)),
                            input_uu_i, result_1)

        saver.save(sess, checkpoint_dir + 'model_%d.ckpt' % (epoch + 1))

    print("[*] Finish training for phase %s." % train_phase)


def run_reflectance_restoration_net(args_param: argparse.Namespace):
    training = tf.placeholder_with_default(False, shape=(), name='training')
    data = args_param.train_data_dir
    epoch = args_param.reflect_epoch
    eval_every_epoch = args_param.reflect_eval_every_epoch
    batch_size = args_param.reflect_batch_size
    patch_size = args_param.reflect_patch_size
    os.environ['CUDA_VISIBLE_DEVICES'] = args_param.cuda

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # decomnet input
    input_all = tf.placeholder(tf.float32, [None, None, None, 3], name='input_all')

    # restoration net input
    input_low_r = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low_r')
    input_low_i = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_i')
    input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')
    input_high_i = tf.placeholder(tf.float32, [None, None, None, 1], name='input_high_i')

    [R_decom, I_decom] = DecomNet(input_all)
    output_r = Restoration_net(input_low_r, input_low_i, training)

    # network output
    output_R_all = R_decom
    output_I_all = I_decom

    # define loss
    # ssim loss
    output_r_1 = output_r[:, :, :, 0:1]
    input_high_1 = input_high[:, :, :, 0:1]
    ssim_r_1 = tf_ssim(output_r_1, input_high_1)
    output_r_2 = output_r[:, :, :, 1:2]
    input_high_2 = input_high[:, :, :, 1:2]
    ssim_r_2 = tf_ssim(output_r_2, input_high_2)
    output_r_3 = output_r[:, :, :, 2:3]
    input_high_3 = input_high[:, :, :, 2:3]
    ssim_r_3 = tf_ssim(output_r_3, input_high_3)
    ssim_r = (ssim_r_1 + ssim_r_2 + ssim_r_3) / 3.0
    loss_ssim = 1 - ssim_r
    # mse loss
    loss_square = tf.reduce_mean(
        tf.square(output_r - input_high))  # *(1-input_low_i))# * ( 1 - input_low_r ))#* (1- input_low_i)))
    # total loss
    loss_restoration = 1 * loss_square + 1 * loss_ssim

    lr = tf.placeholder(tf.float32, name='learning_rate')

    global_step = tf.get_variable('global_step', [], dtype=tf.int32, initializer=tf.constant_initializer(0),
                                  trainable=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')
    with tf.control_dependencies(update_ops):
        grads = optimizer.compute_gradients(loss_restoration)
        train_op_restoration = optimizer.apply_gradients(grads, global_step=global_step)

    var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
    var_restoration = [var for var in tf.trainable_variables() if 'Denoise_Net' in var.name]
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_restoration += bn_moving_vars

    saver_restoration = tf.train.Saver(var_list=var_restoration)
    saver_Decom = tf.train.Saver(var_list=var_Decom)
    sess.run(tf.global_variables_initializer())
    print("[*] Initialize model successfully...")

    eval_low_data = []
    eval_high_data = []
    eval_low_data_bmp = []

    eval_low_data_name = glob(os.path.join(data, 'our485/low/*.png')) + glob(
        os.path.join(data, 'add_sys/sys_low/*.png')) + glob(os.path.join(data, 'dark/low/*.png'))
    eval_low_data_name.sort()
    for idx in range(len(eval_low_data_name)):
        eval_low_im = load_images(eval_low_data_name[idx])
        eval_low_data.append(eval_low_im)

    eval_low_data_name_bmp = glob(os.path.join(data.replace('our485', 'eval15'), 'low/*.png'))
    eval_low_data_name_bmp.sort()
    for idx in range(len(eval_low_data_name_bmp)):
        eval_low_im = load_images(eval_low_data_name_bmp[idx])
        eval_low_data_bmp.append(eval_low_im)
        # print(eval_low_im.shape)

    eval_high_data_name = glob(os.path.join(data,
                                            'our485/high/*.png')) + glob(
        os.path.join(data, 'add_sys/sys_high/*.png')) + glob(os.path.join(data, 'dark/high/*.png'))
    eval_high_data_name.sort()
    for idx in range(len(eval_high_data_name)):
        eval_high_im = load_images(eval_high_data_name[idx])
        eval_high_data.append(eval_high_im)
        # print(eval_high_im.shape)

    pre_checkpoint_dir = './checkpoint/decom_model/'
    ckpt_pre = tf.train.get_checkpoint_state(pre_checkpoint_dir)
    if ckpt_pre:
        print('loaded ' + ckpt_pre.model_checkpoint_path)
        saver_Decom.restore(sess, ckpt_pre.model_checkpoint_path)
    else:
        print('No pre_checkpoint!')

    train_restoration_low_r_data_480 = []
    train_restoration_low_i_data_480 = []
    train_restoration_high_r_data_480 = []

    for idx in range(len(eval_high_data)):
        input_low_eval = np.expand_dims(eval_high_data[idx], axis=0)
        # print(idx)
        result_1, result_2 = sess.run([output_R_all, output_I_all], feed_dict={input_all: input_low_eval})
        result_1 = (result_1 * 0.99) ** 1.2
        result_1_sq = np.squeeze(result_1)
        result_2_sq = np.squeeze(result_2)
        # print(result_1.shape, result_2.shape)
        train_restoration_high_r_data_480.append(result_1_sq)

    for idx in range(len(eval_low_data)):
        input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
        # print(idx)
        result_11, result_12 = sess.run([output_R_all, output_I_all], feed_dict={input_all: input_low_eval})
        result_11_sq = np.squeeze(result_11)
        result_12_sq = np.squeeze(result_12)
        # print(result_11.shape, result_12.shape)
        train_restoration_low_r_data_480.append(result_11_sq)
        train_restoration_low_i_data_480.append(result_12_sq)

    eval_restoration_low_r_data_bmp = []
    eval_restoration_low_i_data_bmp = []
    for idx in range(len(eval_low_data_bmp)):
        input_low_eval = np.expand_dims(eval_low_data_bmp[idx], axis=0)
        # print(idx)
        result_11, result_12 = sess.run([output_R_all, output_I_all], feed_dict={input_all: input_low_eval})
        result_11_sq = np.squeeze(result_11)
        result_12_sq = np.squeeze(result_12)
        # print(result_11.shape, result_12.shape)
        eval_restoration_low_r_data_bmp.append(result_11_sq)
        eval_restoration_low_i_data_bmp.append(result_12_sq)

    eval_restoration_low_r_data = train_restoration_low_r_data_480[235:240]
    eval_restoration_low_i_data = train_restoration_low_i_data_480[235:240]

    train_restoration_low_r_data = train_restoration_low_r_data_480[0:234] + train_restoration_low_r_data_480[241:-1]
    train_restoration_low_i_data = train_restoration_low_i_data_480[0:234] + train_restoration_low_i_data_480[241:-1]
    train_restoration_high_r_data = train_restoration_high_r_data_480[0:234] + train_restoration_high_r_data_480[241:-1]
    print(len(train_restoration_high_r_data), len(train_restoration_low_r_data), len(train_restoration_low_i_data))
    print(len(eval_restoration_low_r_data), len(eval_restoration_low_i_data))
    assert len(train_restoration_high_r_data) == len(train_restoration_low_r_data)
    assert len(train_restoration_low_i_data) == len(train_restoration_low_r_data)

    print('[*] Number of training data: %d' % len(train_restoration_high_r_data))
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = 0.0001

    def lr_schedule(epoch):
        initial_lr = learning_rate
        if epoch <= 300:
            lr = initial_lr
        elif epoch <= 500:
            lr = initial_lr / 2
        elif epoch <= 1500:
            lr = initial_lr / 4
        else:
            lr = initial_lr / 8
        return lr

    sample_dir = os.path.join(args_param.sample_dir, 'new_restoration_train_results_3')
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)
    train_phase = 'restoration'
    numBatch = len(train_restoration_low_r_data) // int(batch_size)
    train_op = train_op_restoration
    train_loss = loss_restoration
    saver = saver_restoration

    checkpoint_dir = os.path.join(args_param.checkpoint_dir, 'new_restoration_retrain_3')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("No restoration pre model!")

    start_step = 0
    start_epoch = 0
    iter_num = 0
    print(
        "[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))
    start_time = time.time()
    image_id = 0

    for epoch in range(start_epoch, epoch):
        for batch_id in range(start_step, numBatch):
            batch_input_low_r = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
            batch_input_low_i = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")

            batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
            batch_input_high_i = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")

            for patch_id in range(batch_size):
                h, w, _ = train_restoration_low_r_data[image_id].shape
                x = random.randint(0, h - patch_size)
                y = random.randint(0, w - patch_size)
                i_low_expand = np.expand_dims(train_restoration_low_i_data[image_id], axis=2)
                rand_mode = random.randint(0, 7)
                batch_input_low_r[patch_id, :, :, :] = data_augmentation(
                    train_restoration_low_r_data[image_id][x: x + patch_size, y: y + patch_size, :],
                    rand_mode)  # + np.random.normal(0, 0.1, (patch_size,patch_size,3))  , rand_mode)
                batch_input_low_i[patch_id, :, :, :] = data_augmentation(
                    i_low_expand[x: x + patch_size, y: y + patch_size, :],
                    rand_mode)  # + np.random.normal(0, 0.1, (patch_size,patch_size,3))  , rand_mode)
                batch_input_high[patch_id, :, :, :] = data_augmentation(
                    train_restoration_high_r_data[image_id][x: x + patch_size, y: y + patch_size, :], rand_mode)

                image_id = (image_id + 1) % len(train_restoration_low_r_data)
                if image_id == 0:
                    tmp = list(
                        zip(train_restoration_low_r_data, train_restoration_low_i_data, train_restoration_high_r_data))
                    random.shuffle(list(tmp))
                    train_restoration_low_r_data, train_restoration_low_i_data, train_restoration_high_r_data = zip(
                        *tmp)

            _, loss = sess.run([train_op, train_loss],
                               feed_dict={input_low_r: batch_input_low_r, input_low_i: batch_input_low_i, \
                                          input_high: batch_input_high, \
                                          training: True, lr: lr_schedule(epoch)})
            print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                  % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
            iter_num += 1
        if (epoch + 1) % eval_every_epoch == 0:
            print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch + 1))

            for idx in range(len(eval_restoration_low_r_data)):
                input_uu_r = eval_restoration_low_r_data[idx]
                input_low_eval_r = np.expand_dims(input_uu_r, axis=0)
                input_uu_i = eval_restoration_low_i_data[idx]
                input_low_eval_i = np.expand_dims(input_uu_i, axis=0)
                input_low_eval_ii = np.expand_dims(input_low_eval_i, axis=3)
                result_1 = sess.run(output_r, feed_dict={input_low_r: input_low_eval_r, input_low_i: input_low_eval_ii,
                                                         training: False})
                save_images(os.path.join(sample_dir, 'eval_%d_%d.png' % (idx + 101, epoch + 1)), result_1)

            for idx in range(len(eval_restoration_low_r_data_bmp)):
                input_uu_r = eval_restoration_low_r_data_bmp[idx]
                input_low_eval_r = np.expand_dims(input_uu_r, axis=0)
                input_uu_i = eval_restoration_low_i_data_bmp[idx]
                input_low_eval_i = np.expand_dims(input_uu_i, axis=0)
                input_low_eval_ii = np.expand_dims(input_low_eval_i, axis=3)
                result_1 = sess.run(output_r, feed_dict={input_low_r: input_low_eval_r, training: False, \
                                                         input_low_i: input_low_eval_ii})
                save_images(os.path.join(sample_dir, 'eval_bmp_%d_%d.png' % (idx + 101, epoch + 1)), result_1)

            global_step = epoch
            saver.save(sess, checkpoint_dir + 'model_%d.ckpt' % (epoch + 1), global_step=global_step)

    print("[*] Finish training for phase %s." % train_phase)


def run_evaluate():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_data_dir', dest='train_data_dir', default='/home/ray/data/LOLdataset_KinD',
                        help='directory for training inputs')
    parser.add_argument('--sample_dir', dest='data', type=str, default='./experiment/exp1/simple', help='batch size')
    parser.add_argument('--checkpoint_dir', dest='data', type=str, default='./experiment/exp1/checkpoint',
                        help='batch size')
    parser.add_argument('--cuda', dest='cuda', type=str, default='1', help='cpu,0,1')

    parser.add_argument('--decom_batch_size', type=int, default=8, help='number of samples in one batch')
    parser.add_argument('--decom_patch_size', type=int, default=48, help='patch size')
    parser.add_argument('--decom_learning_rate', type=int, default=0.0001, help='learn rate')
    parser.add_argument('--decom_epoch', type=int, default=2500, help='epoch')
    parser.add_argument('--decom_eval_every_epoch', dest='eval_every_epoch', type=int, default=500,
                        help='eval_every-epoch')

    parser.add_argument('--illumin_batch_size', dest='batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--illumin_patch_size', dest='patch_size', type=int, default=48, help='batch size')
    parser.add_argument('--illumin_learning_rate', type=int, default=0.0001, help='learn rate')
    parser.add_argument('--illumin_epoch', type=int, default=2000, help='epoch')
    parser.add_argument('--illumin_eval_every_epoch', type=int, default=200, help='eval_every-epoch')

    parser.add_argument('--reflect_batch_size', dest='batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--reflect_patch_size', dest='patch_size', type=int, default=48, help='batch size')
    parser.add_argument('--reflect_learning_rate', dest='learning_rate', type=int, default=0.0001, help='learn rate')
    parser.add_argument('--reflect_epoch', dest='epoch', type=int, default=2000, help='epoch')
    parser.add_argument('--reflect_eval_every_epoch', dest='eval_every_epoch', type=int, default=200, help='eval_every-epoch')

    args_param = parser.parse_args()
    run_decomposition_net(args_param=args_param)
    run_illumination_adjustment_net(args_param=args_param)
    run_reflectance_restoration_net(args_param=args_param)
