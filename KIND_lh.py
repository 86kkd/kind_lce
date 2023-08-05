import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
import os
import numpy as np
from skimage import filters,color
from glob import glob
import scipy.ndimage
from PIL import Image
import time

tf.disable_v2_behavior()

# Assume the model functions are already defined
from model import DecomNet, Illumination_adjust_curve_net_ratio, Restoration_net

def load_images(file):
    im = Image.open(file)
    im = im.resize((600, 400), Image.ANTIALIAS)
    img = np.array(im, dtype="float32") / 255.0
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    return img_norm

def load_data(test_dir):
    eval_low_data = []
    eval_img_name = []
    eval_low_data_name = glob(os.path.join(test_dir, 'low/*.png'))
    eval_low_data_name.sort()
    for idx in range(len(eval_low_data_name)):
        [_, name] = os.path.split(eval_low_data_name[idx])
        suffix = name[name.find('.') + 1:]
        name = name[:name.find('.')]
        eval_img_name.append(name)
        eval_low_im = load_images(eval_low_data_name[idx])
        eval_low_data.append(eval_low_im)
    return eval_low_data, eval_img_name

def gaussian_filter(input_img, sigma):
    output_img = scipy.ndimage.gaussian_filter(input_img, sigma=sigma)
    return output_img.astype(np.float32)

def tf_gaussian_filter(input_img, sigma):
    output_img = tf.py_func(
        gaussian_filter,
        [input_img, sigma],
        tf.float32,
        name='gaussian_filter'
    )
    return output_img
# 定义新计算图
graph = tf.Graph()
with graph.as_default():
    # 占位符
    input_decom = tf.placeholder(tf.float32, [None, None, None, 3], name='input_decom')
    input_low_i_ratio = tf.placeholder(tf.float32, name='ratio')

    # 模型结构
    decom_r_low, decom_i_low = DecomNet(input_decom)
    restoration_r = Restoration_net(decom_r_low, decom_i_low, training=False)
    
    h, w, _ = tf.shape(input_decom)[1], tf.shape(input_decom)[2], tf.shape(input_decom)[3]
    i_low_data_ratio = tf.ones([h, w]) * input_low_i_ratio
    i_low_ratio_expand = tf.expand_dims(i_low_data_ratio, axis=2)
    i_low_ratio_expand2 = tf.expand_dims(i_low_ratio_expand, axis=0)
    
    adjust_i, A = Illumination_adjust_curve_net_ratio(decom_i_low, i_low_ratio_expand2)
    
    decom_r_sq = tf.squeeze(decom_r_low)
    r_gray = tf.image.rgb_to_grayscale(decom_r_sq)
    r_gray_gaussian = tf_gaussian_filter(r_gray, 3)
    low_i = tf.minimum(tf.sqrt(r_gray_gaussian * 2), 1)
    low_i_expand_3 = tf.expand_dims(low_i, axis=3)
    print("Shape of restoration_r: ", restoration_r.get_shape())
    print("Shape of low_i_expand_3: ", low_i_expand_3.get_shape())
    result_denoise = restoration_r * low_i_expand_3
    fusion4 = result_denoise * adjust_i
    
    saver = tf.train.Saver()
    # 启动sessrion运行图
    with tf.Session() as sess:
        # savemode
        checkpoint_dir = './experiment/exp2/checkpoint'
        illmin_name = "illumination_adjust_curve_net_global_rm_del_rotate"
        test_dit = "./simple"

        # Create a session and make it the default session
        sess = tf.Session()
        tf.keras.backend.set_session(sess)
        var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
        var_adjust = [var for var in tf.trainable_variables() if 'I_enhance_Net' in var.name]
        var_restoration = [var for var in tf.trainable_variables() if 'Denoise_Net' in var.name]
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_restoration += bn_moving_vars
        saver_Decom = tf.train.Saver(var_list=var_Decom)
        saver_adjust = tf.train.Saver(var_list=var_adjust)
        saver_restoration = tf.train.Saver(var_list=var_restoration)
        decom_checkpoint_dir = os.path.join(checkpoint_dir, 'decom_net_retrain')
        ckpt_pre = tf.train.get_checkpoint_state(decom_checkpoint_dir)
        if ckpt_pre:
            print('[*] loaded ' + ckpt_pre.model_checkpoint_path)
            saver_Decom.restore(sess, ckpt_pre.model_checkpoint_path)
        else:
            print('[*] No decomnet pretrained model!')

        checkpoint_dir_adjust = os.path.join(checkpoint_dir, illmin_name)
        ckpt_adjust = tf.train.get_checkpoint_state(checkpoint_dir_adjust)
        if ckpt_adjust:
            print('[*] loaded ' + ckpt_adjust.model_checkpoint_path)
            saver_adjust.restore(sess, ckpt_adjust.model_checkpoint_path)
        else:
            print("[*] No adjust net pretrained model!")

        checkpoint_dir_restoration = os.path.join(checkpoint_dir, 'new_restoration_retrain')
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir_restoration)
        if ckpt:
            print('[*] loaded ' + ckpt.model_checkpoint_path)
            saver_restoration.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("[*] No restoration net pretrained model!")
        
        eval_low_data, eval_img_name = load_data(test_dit)   
        for idx in range(len(eval_low_data)):
            name = eval_img_name[idx]
            input_low = eval_low_data[idx]
            input_low_eval = np.expand_dims(input_low, axis=0)
            ratio = float(10.0)
            
            feed_dict = {
                input_decom: input_low_eval,
                input_low_i_ratio: ratio,
                # ... 添加其他的 feed_dict 值
            }
            t1 = time.time()
            output = sess.run(fusion4, feed_dict=feed_dict)
            t2 = time.time()
            print(f"\033[94minfer time:{t2-t1:.3f}s\033[0m")
        # saver.save(sess, "model/model.ckpt")
