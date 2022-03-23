# coding: utf-8
from __future__ import print_function

import argparse
import os
import random
import time
from glob import glob

import numpy as np
import tensorflow.compat.v1 as tf
# import tensorflow as tf
from tensorboardX import SummaryWriter
from tqdm import tqdm

from model import *
from utils import *


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


parser = argparse.ArgumentParser(description='illumination_adjustment_net need parameter')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='batch size')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='batch size')
parser.add_argument('--data', type=str, default='/home/ray/data/LOLdataset_KinD', help='batch size')
parser.add_argument('--sample_dir', type=str, default='./experiment/exp2/simple', help='batch size')
parser.add_argument('--checkpoint_dir', type=str, default='./experiment/exp2/checkpoint')
parser.add_argument('--log_dir', type=str, default="./experiment/exp2/logs")
parser.add_argument('--learning_rate', dest='learning_rate', type=int, default=0.0001, help='learn rate')
parser.add_argument('--epoch', dest='epoch', type=int, default=2000, help='epoch')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', type=int, default=200, help='eval_every-epoch')
parser.add_argument('--cuda', dest='cuda', type=str, default='1', help='cpu,0,1')
args = parser.parse_args()

reinforcement_name = "reinforcement_net_concat_R_I_all_ch128"
illmin_name = "illumination_adjust_curve_net_add_sigmoid_tvlossweight5"
os.makedirs(args.log_dir, exist_ok=True)
writer = SummaryWriter(logdir=os.path.join(args.log_dir, reinforcement_name))

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
batch_size = args.batch_size
patch_size = args.patch_size
data = args.data
learning_rate = args.learning_rate
epoch = args.epoch
eval_every_epoch = args.eval_every_epoch

sess = tf.Session()
training = tf.placeholder_with_default(False, shape=(), name='training')
# the input of decomposition net
input_decom = tf.placeholder(tf.float32, [None, None, None, 3], name='input_decom')
input_low_data = input_decom
input_high_data = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high_data')
input_low_r = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low_r')
input_low_i = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_i')
input_high_r = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high_r')
input_high_i = tf.placeholder(tf.float32, [None, None, None, 1], name='input_high_i')

[R_decom, I_decom] = DecomNet(input_decom)
# the output of decomposition network
decom_output_R = R_decom
decom_output_I = I_decom

output_i, A = Illumination_adjust_curve_net(input_low_i)
# the output of illumination adjustment net
input_illumin_i = output_i

output_r = Restoration_net(input_low_r, input_low_i, training)
# the output of restoration net
input_restoration_r = output_r

# the output of reinforcement net
output = Reinforcement_Net(input_low_r, input_low_i, input_decom)
# output = Reinforcement_UNet(input_low_r, input_low_i, input_decom)

# define loss
loss_mae = tf.reduce_mean(tf.abs(input_high_data - output))
loss_ssim = tf.reduce_mean(tf.image.ssim_multiscale(input_high_data, output, max_val=255))
loss_reinforcement = 5 * loss_mae + 1 - loss_ssim

lr = tf.placeholder(tf.float32, name='learning_rate')
optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')

var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
var_adjust = [var for var in tf.trainable_variables() if 'I_enhance_Net' in var.name]
var_restoration = [var for var in tf.trainable_variables() if 'Denoise_Net' in var.name]
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_restoration += bn_moving_vars
var_reinforcement = [var for var in tf.trainable_variables() if 'Reinforcement_Net' in var.name]

saver_adjust = tf.train.Saver(var_list=var_adjust)
saver_Decom = tf.train.Saver(var_list=var_Decom)
saver_restoration = tf.train.Saver(var_list=var_restoration)
saver_reinforcement = tf.train.Saver(var_list=var_reinforcement)

train_op_reinforcement = optimizer.minimize(loss_reinforcement, var_list=var_reinforcement)
sess.run(tf.global_variables_initializer())
print("[*] Initialize model successfully...")

# load data
# Based on the decomposition net, we first get the decomposed reflectance maps
# and illumination maps, then train the adjust net.
# train_datatrain_illumin_low_data = []

train_low_data = []
train_high_data = []
train_low_data_names = glob(os.path.join(data, 'our485/low/*.png'))
train_low_data_names.sort()
train_high_data_names = glob(os.path.join(data, 'our485/high/*.png'))
train_high_data_names.sort()
assert len(train_low_data_names) == len(train_high_data_names)
print('[*] Number of training data: %d' % len(train_low_data_names))
for idx in range(len(train_low_data_names)):
    low_im = load_images(train_low_data_names[idx])
    train_low_data.append(low_im)
    high_im = load_images(train_high_data_names[idx])
    train_high_data.append(high_im)

pre_decom_checkpoint_dir = os.path.join(args.checkpoint_dir, "decom_net_retrain")
ckpt_pre = tf.train.get_checkpoint_state(pre_decom_checkpoint_dir)
if ckpt_pre:
    print('loaded ' + ckpt_pre.model_checkpoint_path)
    saver_Decom.restore(sess, ckpt_pre.model_checkpoint_path)
else:
    print('No pre_decom_net checkpoint!')

checkpoint_dir_adjust = os.path.join(args.checkpoint_dir, illmin_name)
ckpt_adjust = tf.train.get_checkpoint_state(checkpoint_dir_adjust)
if ckpt_adjust:
    print('[*] loaded ' + ckpt_adjust.model_checkpoint_path)
    saver_adjust.restore(sess, ckpt_adjust.model_checkpoint_path)
else:
    print("[*] No adjust net pretrained model!")

checkpoint_dir_restoration = os.path.join(args.checkpoint_dir, 'new_restoration_retrain')
ckpt_restoration = tf.train.get_checkpoint_state(checkpoint_dir_restoration)
if ckpt_restoration:
    print('[*] loaded ' + ckpt_restoration.model_checkpoint_path)
    saver_restoration.restore(sess, ckpt_restoration.model_checkpoint_path)
else:
    print("[*] No restoration net pretrained model!")

decomposed_low_i_data_480 = []
decomposed_low_r_data_480 = []
decomposed_high_i_data_480 = []
decomposed_high_r_data_480 = []
print("Decomposing image into R and I...")
for idx in tqdm(range(len(train_low_data))):
    input_low = np.expand_dims(train_low_data[idx], axis=0)
    RR, II = sess.run([decom_output_R, decom_output_I], feed_dict={input_decom: input_low})
    RR0 = np.squeeze(RR)
    II0 = np.squeeze(II)
    # print(RR0.shape, II0.shape)
    decomposed_low_r_data_480.append(RR0)
    decomposed_low_i_data_480.append(II0)
for idx in tqdm(range(len(train_high_data))):
    input_high = np.expand_dims(train_high_data[idx], axis=0)
    RR2, II2 = sess.run([decom_output_R, decom_output_I], feed_dict={input_decom: input_high})
    RR02 = np.squeeze(RR2)
    II02 = np.squeeze(II2)
    # print(RR02.shape, II02.shape)
    decomposed_high_r_data_480.append(RR02)
    decomposed_high_i_data_480.append(II02)
print("Finishing decomposition...")

eval_adjust_low_i_data = decomposed_low_i_data_480[451:480]
eval_adjust_high_i_data = decomposed_high_i_data_480[451:480]

eval_adjust_low_r_data = decomposed_low_r_data_480[451:480]
eval_adjust_high_r_data = decomposed_high_r_data_480[451:480]

train_adjust_low_i_data = decomposed_low_i_data_480[0:450]
train_adjust_high_i_data = decomposed_high_i_data_480[0:450]

train_adjust_low_r_data = decomposed_low_r_data_480[0:450]
train_adjust_high_r_data = decomposed_high_r_data_480[0:450]

eval_low_data_30 = train_low_data[451:480]
eval_high_data_30 = train_high_data[451:480]

train_low_data_450 = train_low_data[0:450]
train_high_data_450 = train_high_data[0:450]

print('[*] Number of training data: %d' % len(train_adjust_high_i_data))

train_phase = 'reinforcement'
numBatch = len(train_adjust_low_i_data) // int(batch_size)
train_op = train_op_reinforcement
train_loss = loss_reinforcement
saver = saver_reinforcement

checkpoint_dir = os.path.join(args.checkpoint_dir, reinforcement_name)
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

sample_dir = os.path.join(args.sample_dir, reinforcement_name)
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)

start_time = time.time()
image_id = 0

for epoch in range(start_epoch, epoch):
    loss_list = []
    for batch_id in range(start_step, numBatch):
        reinforcement_decom_i_low = train_adjust_low_i_data[batch_id]
        reinforcement_decom_r_low = train_adjust_low_r_data[batch_id]
        reinforcement_low_data = train_low_data_450[batch_id]
        reinforcement_high_data = train_high_data_450[batch_id]

        reinforcement_decom_i_low = np.expand_dims(reinforcement_decom_i_low, axis=0)
        reinforcement_decom_i_low = np.expand_dims(reinforcement_decom_i_low, axis=3)
        reinforcement_decom_r_low = np.expand_dims(reinforcement_decom_r_low, axis=0)
        reinforcement_low_data = np.expand_dims(reinforcement_low_data, axis=0)
        reinforcement_high_data = np.expand_dims(reinforcement_high_data, axis=0)

        restoration_r = sess.run(output_r, feed_dict={input_low_r: reinforcement_decom_r_low,
                                                      input_low_i: reinforcement_decom_i_low})
        adjust_i = sess.run(output_i, feed_dict={input_low_i: reinforcement_decom_i_low})

        _, loss = sess.run([train_op, train_loss],
                           feed_dict={input_low_i: adjust_i, input_low_r: restoration_r,
                                      input_decom: reinforcement_low_data, input_high_data: reinforcement_high_data
                               , lr: learning_rate})

        loss_list.append(loss)

        print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
              % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
        iter_num += 1
    writer.add_scalar("reinforcement/loss", np.array(loss_list).mean(), global_step=epoch + 1)
    if (epoch + 1) % eval_every_epoch == 0:
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch + 1))

        for idx in range(10):
            eval_reinforcement_decom_i_low = eval_adjust_low_i_data[idx]
            eval_reinforcement_decom_r_low = eval_adjust_low_r_data[idx]
            eval_reinforcement_low_data = eval_low_data_30[idx]
            eval_reinforcement_high_data = eval_high_data_30[idx]

            eval_reinforcement_decom_i_low = np.expand_dims(eval_reinforcement_decom_i_low, axis=0)
            eval_reinforcement_decom_i_low = np.expand_dims(eval_reinforcement_decom_i_low, axis=3)
            eval_reinforcement_decom_r_low = np.expand_dims(eval_reinforcement_decom_r_low, axis=0)
            eval_reinforcement_low_data = np.expand_dims(eval_reinforcement_low_data, axis=0)
            eval_reinforcement_high_data = np.expand_dims(eval_reinforcement_high_data, axis=0)

            restoration_r = sess.run(output_r, feed_dict={input_low_r: eval_reinforcement_decom_r_low,
                                                          input_low_i: eval_reinforcement_decom_i_low})
            adjust_i = sess.run(output_i, feed_dict={input_low_i: eval_reinforcement_decom_i_low})

            res = sess.run(output,
                           feed_dict={input_low_i: adjust_i, input_low_r: restoration_r,
                                      input_decom: eval_reinforcement_low_data, input_high_data: eval_reinforcement_high_data
                               , lr: learning_rate})
            save_images(os.path.join(sample_dir, 'reinforcement_eval_%d_%d.png' % (epoch + 1, idx + 1)), res)

    saver.save(sess, os.path.join(checkpoint_dir, 'Reinforcement_Net.ckpt'), global_step=iter_num)

print("[*] Finish training for phase %s." % train_phase)
