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
from utils import save_images
import argparse

tf.disable_v2_behavior()

parser = argparse.ArgumentParser(description='')
parser.add_argument("--save_samples", action = "store_true")
args = parser.parse_args()



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

def gaussian_kernel(kernel_size, sigma):
    """Creates a 2D Gaussian kernel."""
    x = tf.linspace(-3.0, 3.0, kernel_size)
    z = (1.0 / (sigma * tf.sqrt(2.0 * np.pi))) * tf.exp(tf.negative(tf.pow(x, 2.0) / (2.0 * tf.pow(sigma, 2.0))))
    z_2d = tf.matmul(tf.reshape(z, [kernel_size, 1]), tf.reshape(z, [1, kernel_size]))
    return z_2d / tf.reduce_sum(z_2d)

def gaussian_blur(img, kernel_size, sigma):
    """Applies Gaussian blur to an image tensor."""
    gaussian_filter = gaussian_kernel(kernel_size, sigma)
    gaussian_filter = gaussian_filter[:, :, tf.newaxis, tf.newaxis]
    return tf.nn.depthwise_conv2d(img, gaussian_filter, [1, 1, 1, 1], padding='SAME')

# 定义新计算图
graph = tf.Graph()
with graph.as_default():
    # 占位符
    input_decom = tf.placeholder(tf.float32, [1, 400, 600, 3], name='input_decom')
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
    r_gray = r_gray[tf.newaxis, :, :, :]
    r_gray_gaussian = gaussian_blur(r_gray, kernel_size=7, sigma=3.0)
    low_i = tf.minimum(tf.sqrt(r_gray_gaussian * 2), 1)
    low_i_expand_0 = tf.expand_dims(low_i, axis=0)
    result_denoise = restoration_r * low_i_expand_0
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
        
        if args.save_samples:
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
                save_images(os.path.join("sample_results", '%s_KinD_plus.png' % name), output)
        # Convert to SavedModel
        builder = tf.saved_model.builder.SavedModelBuilder("./model/saved_model")
        builder.add_meta_graph_and_variables(sess,
                                            [tf.saved_model.tag_constants.SERVING],
                                            signature_def_map= {
                                                "serving_default": tf.saved_model.signature_def_utils.predict_signature_def(
                                                    inputs= {"input_decom": input_decom, "input_low_i_ratio": input_low_i_ratio},
                                                    outputs= {"fusion4": fusion4})
                                            })
        builder.save() 
        saver.save(sess, "model/model.ckpt")
        print("\033[92msave success\033[0m")
