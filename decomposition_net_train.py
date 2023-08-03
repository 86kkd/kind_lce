# coding: utf-8
from __future__ import print_function

import argparse
import os
import random
import time
from glob import glob

import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
from tensorboardX import SummaryWriter

from model import *
from utils import *


def load_model(sess, saver, ckpt_dir):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt:
        print('[*] loaded ' + ckpt.model_checkpoint_path)
        full_path = tf.train.latest_checkpoint(ckpt_dir)
        try:
            _global_step = int(full_path.split('/')[-1].split('-')[-1])
        except ValueError:
            _global_step = None
        saver.restore(sess, full_path)
        return True, _global_step
    else:
        print("[*] Failed to load model from %s" % ckpt_dir)
        return False, 0


parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
parser.add_argument('--train_data_dir', dest='train_data_dir',
                    default='/media/ray/dataset2/PublicDataset/data_2/ACDC-lowlight',
                    help='directory for training inputs')
parser.add_argument('--sample_dir', type=str, default='./experiment/exp2/simple', help='batch size')
parser.add_argument('--checkpoint_dir', type=str, default='./experiment/exp2/checkpoint',
                    help='batch size')
parser.add_argument('--log_dir', type=str, default="./experiment/exp2/logs")
parser.add_argument('--learning_rate', dest='learning_rate', type=int, default=0.0001, help='learn rate')
parser.add_argument('--epoch', dest='epoch', type=int, default=2500, help='epoch')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', type=int, default=500, help='eval_every-epoch')
parser.add_argument('--cuda', dest='cuda', type=str, default='0', help='cpu,0,1')
args = parser.parse_args()

os.makedirs(args.log_dir, exist_ok=True)
writer = SummaryWriter(logdir=args.log_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
learning_rate = args.learning_rate
epoch = args.epoch
eval_every_epoch = args.eval_every_epoch
batch_size = args.batch_size
patch_size = args.patch_size
data = args.train_data_dir
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


# define loss

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
train_decom_low_data_names = glob(os.path.join(data, 'our485/low/*.png'))
train_decom_low_data_names.sort()
train_decom_high_data_names = glob(os.path.join(data, 'our485/high/*.png'))
train_decom_high_data_names.sort()
assert len(train_decom_low_data_names) == len(train_decom_high_data_names) and len(train_decom_low_data_names) != 0
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

sample_dir = os.path.join(args.sample_dir, 'decom_net_train_result_acdc')
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)

train_phase = 'decomposition'
numBatch = len(train_decom_low_data) // int(batch_size)
train_op = train_op_Decom
train_loss = loss_Decom
saver = saver_Decom

checkpoint_dir = os.path.join(args.checkpoint_dir, 'decom_net_retrain_acdc')
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
start_time = time.time()
image_id = 0
for epoch in range(start_epoch, epoch):
    loss_list = []
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
        loss_list.append(loss)
        print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
              % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
        iter_num += 1
    writer.add_scalar("decom/loss", np.array(loss_list).mean(), global_step=epoch + 1)
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

    saver.save(sess, os.path.join(checkpoint_dir, 'Decomposition_Net.ckpt'), global_step=iter_num)

print("[*] Finish training for phase %s." % train_phase)
