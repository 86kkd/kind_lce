# coding: utf-8
from __future__ import print_function

import argparse
import os
import random
import time
from glob import glob

import tensorflow.compat.v1 as tf
from tensorboardX import SummaryWriter

from model import *
from utils import *


# define loss

def grad_loss(input_i_low, input_i_high):
    x_loss = tf.square(gradient_no_abs(input_i_low, 'x') - gradient_no_abs(input_i_high, 'x'))
    y_loss = tf.square(gradient_no_abs(input_i_low, 'y') - gradient_no_abs(input_i_high, 'y'))
    grad_loss_all = tf.reduce_mean(x_loss + y_loss)
    return grad_loss_all


def color_loss(x, name=None):
    with tf.name_scope(name, "color_loss"):
        mean_rgb = tf.reduce_mean(x, [1, 2], keepdims=True)
        # mean_r, mean_g, mean_b = tf.split(mean_rgb, num_or_size_splits=1, axis=3)
        mean_r, mean_g, mean_b = mean_rgb[..., 0], mean_rgb[..., 1], mean_rgb[..., 2]

        d_rg = tf.square(mean_r - mean_g)
        d_rb = tf.square(mean_r - mean_b)
        d_gb = tf.square(mean_b - mean_g)
        d_sum = tf.square(d_rg) + tf.square(d_rb) + tf.square(d_gb)
        return tf.sqrt(d_sum)


def load_model(sess, saver, ckpt_dir):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        full_path = tf.train.latest_checkpoint(ckpt_dir)
        try:
            global_step = int(full_path.split('/')[-1].split('-')[-1])
        except ValueError:
            global_step = None
        saver.restore(sess, full_path)
        return True, global_step
    else:
        print("[*] Failed to load model from %s" % ckpt_dir)
        return False, 0


parser = argparse.ArgumentParser(description='illumination_adjustment_net need parameter')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=10, help='batch size')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='batch size')
parser.add_argument('--data', type=str, default='/media/ray/dataset2/PublicDataset/data_2/ACDC-lowlight',
                    help='batch size')
parser.add_argument('--sample_dir', type=str, default='./experiment/exp2/simple', help='batch size')
parser.add_argument('--checkpoint_dir', type=str, default='./experiment/exp2/checkpoint')
parser.add_argument('--log_dir', type=str, default="./experiment/exp2/logs")
parser.add_argument('--learning_rate', dest='learning_rate', type=int, default=0.0001, help='learn rate')
parser.add_argument('--epoch', dest='epoch', type=int, default=2000, help='epoch')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', type=int, default=200, help='eval_every-epoch')
parser.add_argument('--cuda', dest='cuda', type=str, default='0', help='cpu,0,1')
args = parser.parse_args()

name = "illumination_adjust_curve_net_global_rm_del_rotate_no_random_acdc"
os.makedirs(args.log_dir, exist_ok=True)
writer = SummaryWriter(logdir=os.path.join(args.log_dir, name))

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
batch_size = args.batch_size
patch_size = args.patch_size
data = args.data
learning_rate = args.learning_rate
epoch = args.epoch
eval_every_epoch = args.eval_every_epoch
decom_net_ckpt_path = os.path.join(args.checkpoint_dir, "decom_net_retrain_acdc")

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
output_i, A = Illumination_adjust_curve_net_ratio(input_low_i, input_low_i_ratio)

loss_grad = grad_loss(output_i, input_high_i)
loss_square = tf.reduce_mean(tf.square(output_i - input_high_i))
loss_tv = tf.reduce_mean(tf.image.total_variation(A))
loss_adjust = loss_square + 0.1 * loss_grad + loss_tv

lr = tf.placeholder(tf.float32, name='learning_rate')

optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')

var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
var_adjust = [var for var in tf.trainable_variables() if 'I_enhance_Net' in var.name]

saver_adjust = tf.train.Saver(var_list=var_adjust)
saver_Decom = tf.train.Saver(var_list=var_Decom)
train_op_adjust = optimizer.minimize(loss_adjust, var_list=var_adjust)
sess.run(tf.global_variables_initializer())
print("[*] Initialize model successfully...")

# load data
# Based on the decomposition net, we first get the decomposed reflectance maps
# and illumination maps, then train the adjust net.
# train_datatrain_illumin_low_data = []

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

pre_decom_checkpoint_dir = decom_net_ckpt_path
ckpt_pre = tf.train.get_checkpoint_state(pre_decom_checkpoint_dir)
if ckpt_pre:
    print('loaded ' + ckpt_pre.model_checkpoint_path)
    saver_Decom.restore(sess, ckpt_pre.model_checkpoint_path)
else:
    print('No pre_decom_net checkpoint!')

decomposed_low_i_data_480 = []
decomposed_high_i_data_480 = []
decomposed_low_r_data_480 = []
decomposed_high_r_data_480 = []
for idx in range(len(train_illumin_low_data)):
    input_low = np.expand_dims(train_illumin_low_data[idx], axis=0)
    RR, II = sess.run([decom_output_R, decom_output_I], feed_dict={input_decom: input_low})
    RR0 = np.squeeze(RR)
    II0 = np.squeeze(II)
    decomposed_low_i_data_480.append(II0)
    RR0 = np.resize(RR0, (400, 600))
    decomposed_low_r_data_480.append(RR0)
for idx in range(len(train_illumin_high_data)):
    input_high = np.expand_dims(train_illumin_high_data[idx], axis=0)
    RR2, II2 = sess.run([decom_output_R, decom_output_I], feed_dict={input_decom: input_high})
    RR02 = np.squeeze(RR2)
    II02 = np.squeeze(II2)
    decomposed_high_i_data_480.append(II02)
    RR02 = np.resize(RR02, (400, 600))
    decomposed_high_r_data_480.append(RR02)

train_eval_divider = 371
total_num = 400

eval_adjust_low_i_data = decomposed_low_i_data_480[train_eval_divider:total_num]
eval_adjust_high_i_data = decomposed_high_i_data_480[train_eval_divider:total_num]

train_adjust_low_i_data = decomposed_low_i_data_480[0:train_eval_divider - 1]
train_adjust_high_i_data = decomposed_high_i_data_480[0:train_eval_divider - 1]

train_adjust_low_r_data = decomposed_low_r_data_480[0:train_eval_divider - 1]
train_adjust_high_r_data = decomposed_high_r_data_480[0:train_eval_divider - 1]

print('[*] Number of training data: %d' % len(train_adjust_high_i_data))

train_phase = 'adjustment'
numBatch = len(train_adjust_low_i_data) // int(batch_size)
train_op = train_op_adjust
train_loss = loss_adjust
saver = saver_adjust

checkpoint_dir = os.path.join(args.checkpoint_dir, name)
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

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

sample_dir = os.path.join(args.sample_dir, name)
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)

start_time = time.time()
image_id = 0

size = 500

for epoch in range(start_epoch, epoch):
    loss_list = []
    for batch_id in range(start_step, numBatch):
        batch_input_low_i_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_input_high_i_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_input_low_i = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_input_high_i = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        input_low_i_rand = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        input_high_i_rand = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        input_low_i_rand_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        # input_high_i_rand_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        # m表示全图 还需要修改
        batch_input_low_m_ratio = np.zeros((batch_size, size, size, 1),
                                           dtype="float32")
        batch_input_high_m_ratio = np.zeros((batch_size, size, size, 1), dtype="float32")
        batch_input_low_m = np.zeros((batch_size, size, size, 1), dtype="float32")
        batch_input_high_m = np.zeros((batch_size, size, size, 1), dtype="float32")
        input_low_m_rand_ratio = np.zeros((batch_size, size, size, 1), dtype="float32")
        # input_high_m_rand_ratio = np.zeros((batch_size, size, size, 1), dtype="float32")
        input_low_m_rand = np.zeros((batch_size, size, size, 1), dtype="float32")
        input_high_m_rand = np.zeros((batch_size, size, size, 1), dtype="float32")
        # r表示反射图
        batch_input_low_rm = np.zeros((batch_size, size, size, 1), dtype="float32")
        batch_input_high_rm = np.zeros((batch_size, size, size, 1), dtype="float32")
        batch_input_low_ri = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_input_high_ri = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        for patch_id in range(batch_size):
            i_low_data = train_adjust_low_i_data[image_id]
            i_low_expand = np.expand_dims(i_low_data, axis=2)
            i_high_data = train_adjust_high_i_data[image_id]
            i_high_expand = np.expand_dims(i_high_data, axis=2)

            ri_low_data = train_adjust_low_r_data[image_id]
            ri_low_expand = np.expand_dims(ri_low_data, axis=2)
            ri_high_data = train_adjust_high_r_data[image_id]
            ri_high_expand = np.expand_dims(ri_high_data, axis=2)

            rm_low_expand = np.resize(ri_low_expand, (size, size, 1))
            rm_high_expand = np.resize(ri_high_expand, (size, size, 1))
            m_low_expand = np.resize(i_low_expand, (size, size, 1))
            m_high_expand = np.resize(i_high_expand, (size, size, 1))

            h, w = train_adjust_low_i_data[image_id].shape
            x = random.randint(0, h - patch_size)
            y = random.randint(0, w - patch_size)
            i_low_data_crop = i_low_expand[x: x + patch_size, y: y + patch_size, :]
            i_high_data_crop = i_high_expand[x: x + patch_size, y: y + patch_size, :]
            ri_low_data_crop = ri_low_expand[x: x + patch_size, y: y + patch_size, :]
            ri_high_data_crop = ri_high_expand[x: x + patch_size, y: y + patch_size, :]

            batch_input_low_i[patch_id, :, :, :] = i_low_data_crop
            batch_input_high_i[patch_id, :, :, :] = i_high_data_crop

            batch_input_low_m[patch_id, :, :, :] = m_low_expand
            batch_input_high_m[patch_id, :, :, :] = m_high_expand

            batch_input_low_ri[patch_id, :, :, :] = ri_low_data_crop
            batch_input_high_ri[patch_id, :, :, :] = ri_high_data_crop

            batch_input_low_rm[patch_id, :, :, :] = rm_low_expand
            batch_input_high_rm[patch_id, :, :, :] = rm_high_expand

            ratio = np.mean(i_low_data_crop / (i_high_data_crop + 0.0001))  # 对一张图片局部的高低照度求一次比值差
            ratio_m = np.mean(m_low_expand / (m_high_expand + 0.0001))
            # i_low_ratio_expand = batch_input_low_ri[patch_id, :, :, :] * (1 / ratio + 0.0001)
            # i_high_ratio_expand = batch_input_high_ri[patch_id, :, :, :] * (ratio)
            i_low_data_ratio = np.ones([patch_size, patch_size]) * (1 / ratio + 0.0001)  # 将np.ones可以换成R_grey
            i_low_ratio_expand = np.expand_dims(i_low_data_ratio, axis=2)
            i_high_data_ratio = np.ones([patch_size, patch_size]) * (ratio)
            i_high_ratio_expand = np.expand_dims(i_high_data_ratio, axis=2)
            # m_low_data_ratio = np.ones([size, size]) * (1 / ratio_m + 0.0001)
            # m_low_ratio_expand = np.expand_dims(m_low_data_ratio, axis=2)
            # m_high_data_ratio = np.ones([size, size]) * (ratio_m)
            # m_high_ratio_expand = np.expand_dims(m_high_data_ratio, axis=2)
            m_low_ratio_expand = batch_input_low_rm[patch_id, :, :, :] * (1 / ratio + 0.0001)
            m_high_ratio_expand = batch_input_high_rm[patch_id, :, :, :] * (ratio)
            batch_input_low_i_ratio[patch_id, :, :, :] = i_low_ratio_expand
            batch_input_high_i_ratio[patch_id, :, :, :] = i_high_ratio_expand
            batch_input_low_m_ratio[patch_id, :, :, :] = m_low_ratio_expand
            batch_input_high_m_ratio[patch_id, :, :, :] = m_high_ratio_expand

            rand_mode = np.random.randint(0, 2)
            if rand_mode == 1:
                input_low_i_rand[patch_id, :, :, :] = batch_input_low_i[patch_id, :, :, :]
                input_high_i_rand[patch_id, :, :, :] = batch_input_high_i[patch_id, :, :, :]
                input_low_i_rand_ratio[patch_id, :, :, :] = batch_input_low_i_ratio[patch_id, :, :, :]
                # input_high_i_rand_ratio[patch_id, :, :, :] = batch_input_high_i_ratio[patch_id, :, :, :]
            else:
                input_low_i_rand[patch_id, :, :, :] = batch_input_high_i[patch_id, :, :, :]
                input_high_i_rand[patch_id, :, :, :] = batch_input_low_i[patch_id, :, :, :]
                input_low_i_rand_ratio[patch_id, :, :, :] = batch_input_high_i_ratio[patch_id, :, :, :]
                # input_high_i_rand_ratio[patch_id, :, :, :] = batch_input_low_i_ratio[patch_id, :, :, :]

            image_id = (image_id + 1) % len(train_adjust_low_i_data)

        _, loss = sess.run([train_op, train_loss],
                           feed_dict={input_low_i: input_low_i_rand, \
                                      input_high_i: input_high_i_rand, \
                                      lr: learning_rate})
        _, loss_m = sess.run((train_op, train_loss),  # 通过全局图片进行优化
                             feed_dict={input_low_i: input_low_m_rand, input_low_i_ratio: input_low_m_rand_ratio, \
                                        input_high_i: input_high_m_rand, \
                                        lr: learning_rate})
        loss_list.append(loss)

        print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
              % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
        iter_num += 1
    writer.add_scalar("illumin/loss", np.array(loss_list).mean(), global_step=epoch + 1)
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
                                feed_dict={input_low_i: input_low_eval_ii})
            save_images(os.path.join(sample_dir, 'h_eval_%d_%d_%5f.png' % (epoch + 1, rand_idx + 1, rand_ratio)),
                        input_uu_i, result_1)

    saver.save(sess, os.path.join(checkpoint_dir, 'Illumination_Adjustment_Net.ckpt'), global_step=iter_num)

print("[*] Finish training for phase %s." % train_phase)
