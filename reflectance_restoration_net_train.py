# coding: utf-8
from __future__ import print_function

import os

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
import random
import argparse

from utils import *
from model import *
from glob import glob

import tensorflow.compat.v1 as tf
from tensorboardX import SummaryWriter


def load_model(sess, saver, ckpt_dir):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt:
        full_path = tf.train.latest_checkpoint(ckpt_dir)
        print('loaded ' + ckpt.model_checkpoint_path)
        try:
            global_step = int(full_path.split('/')[-1].split('-')[-1])
        except ValueError:
            global_step = None
        saver.restore(sess, full_path)
        return True, global_step
    else:
        print("[*] Failed to load model from %s" % ckpt_dir)
        return False, 0


parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
parser.add_argument('--epoch', dest='epoch', type=int, default=2400, help='epoch')
parser.add_argument('--data', dest='data', type=str,
                    default='/media/ray/dataset2/PublicDataset/data_2/ACDC-lowlight',
                    help='batch size')
parser.add_argument('--log_dir', type=str, default="./experiment/exp2/logs")
parser.add_argument('--sample_dir', type=str, default='./experiment/exp2/simple')
parser.add_argument('--checkpoint_dir', type=str, default='./experiment/exp2/checkpoint')
parser.add_argument('--cuda', dest='cuda', type=str, default='0', help='cpu,0,1')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', type=int, default=150, help='eval_every-epoch')
args = parser.parse_args()

training = tf.placeholder_with_default(False, shape=(), name='training')
data = args.data
epoch = args.epoch
eval_every_epoch = args.eval_every_epoch
batch_size = args.batch_size
patch_size = args.patch_size
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

os.makedirs(args.log_dir, exist_ok=True)
writer = SummaryWriter(logdir=args.log_dir)

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

# pre_checkpoint_dir = './checkpoint/decom_net_retrain'
pre_checkpoint_dir = os.path.join(args.checkpoint_dir, "decom_net_retrain_acdc")
ckpt_pre = tf.train.get_checkpoint_state(pre_checkpoint_dir)
if ckpt_pre:
    print('[*] loaded ' + ckpt_pre.model_checkpoint_path)
    saver_Decom.restore(sess, ckpt_pre.model_checkpoint_path)
else:
    print('[*] No decom pre_checkpoint!', pre_checkpoint_dir)

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
# print(len(train_restoration_high_r_data), len(train_restoration_low_r_data), len(train_restoration_low_i_data))
# print(len(eval_restoration_low_r_data), len(eval_restoration_low_i_data))
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


sample_dir = os.path.join(args.sample_dir, 'new_restoration_train_results_acdc')
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)
train_phase = 'restoration'
numBatch = len(train_restoration_low_r_data) // int(batch_size)
train_op = train_op_restoration
train_loss = loss_restoration
saver = saver_restoration

checkpoint_dir = os.path.join(args.checkpoint_dir, 'new_restoration_retrain_acdc')
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

start_epoch = None
iter_num = None
start_step = None

load_model_status, global_step = load_model(sess, saver, checkpoint_dir)
if load_model_status:
    iter_num = global_step
    start_epoch = global_step // numBatch
    start_step = global_step % numBatch
    print("[*] Model restore success!")
else:
    iter_num = 0
    start_epoch = 0
    start_step = 0
    print("[*] Not find pretrained model!")

print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))
start_time = time.time()
image_id = 0

for epoch in range(start_epoch, epoch):
    loss_list = []
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
                train_restoration_low_r_data, train_restoration_low_i_data, train_restoration_high_r_data = zip(*tmp)
        _learning_rate = lr_schedule(epoch)
        _, loss = sess.run([train_op, train_loss],
                           feed_dict={input_low_r: batch_input_low_r, input_low_i: batch_input_low_i, \
                                      input_high: batch_input_high, \
                                      training: True, lr:_learning_rate})
        loss_list.append(loss)
        print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
              % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
        iter_num += 1
    writer.add_scalar("reflectance/loss", np.array(loss_list).mean(), global_step=epoch + 1)
    writer.add_scalar("reflectance/learning_rate", _learning_rate, global_step=epoch + 1)

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
        saver.save(sess, os.path.join(checkpoint_dir, 'Reflectance_Restoration_Net.ckpt'), global_step=global_step)

print("[*] Finish training for phase %s." % train_phase)
