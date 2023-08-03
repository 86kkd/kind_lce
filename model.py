import tensorflow.compat.v1 as tf

# import tensorflow as tf

tf.disable_v2_behavior()
from msia_BN_3_M import *


def lrelu(x, trainbable=None):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels, scope_name, trainable=True):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool_size = 2
        deconv_filter = tf.get_variable('weights', [pool_size, pool_size, output_channels, in_channels], trainable=True)
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1],
                                        name=scope_name)

        deconv_output = tf.concat([deconv, x2], 3)
        deconv_output.set_shape([None, None, None, output_channels * 2])

        return deconv_output


def DecomNet(input):
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
        pool1 = slim.max_pool2d(conv1, [2, 2], stride=2, padding='SAME')
        conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
        pool2 = slim.max_pool2d(conv2, [2, 2], stride=2, padding='SAME')
        conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
        up8 = upsample_and_concat(conv3, conv2, 64, 128, 'g_up_1')
        conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
        up9 = upsample_and_concat(conv8, conv1, 32, 64, 'g_up_2')
        conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
        # Here, we use 1*1 kernel to replace the 3*3 ones in the paper to get better results.
        conv10 = slim.conv2d(conv9, 3, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
        R_out = tf.sigmoid(conv10)

        l_conv2 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='l_conv1_2')
        l_conv3 = tf.concat([l_conv2, conv9], 3)
        # Here, we use 1*1 kernel to replace the 3*3 ones in the paper to get better results.
        l_conv4 = slim.conv2d(l_conv3, 1, [1, 1], rate=1, activation_fn=None, scope='l_conv1_4')
        L_out = tf.sigmoid(l_conv4)

    return R_out, L_out


def Restoration_net(input_r, input_i, training=True):
    with tf.variable_scope('Denoise_Net', reuse=tf.AUTO_REUSE):
        conv1 = slim.conv2d(input_r, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv1_1')
        conv1 = slim.conv2d(conv1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv1_2')
        msia_1 = msia_3_M(conv1, input_i, name='de_conv1', training=training)  # , name='de_conv1_22')

        conv2 = slim.conv2d(msia_1, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv2_1')
        conv2 = slim.conv2d(conv2, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv2_2')
        msia_2 = msia_3_M(conv2, input_i, name='de_conv2', training=training)

        conv3 = slim.conv2d(msia_2, 512, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv3_1')
        conv3 = slim.conv2d(conv3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv3_2')
        msia_3 = msia_3_M(conv3, input_i, name='de_conv3', training=training)

        conv4 = slim.conv2d(msia_3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv4_1')
        conv4 = slim.conv2d(conv4, 64, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv4_2')
        msia_4 = msia_3_M(conv4, input_i, name='de_conv4', training=training)

        conv5 = slim.conv2d(msia_4, 32, [3, 3], rate=1, activation_fn=lrelu, scope='de_conv5_1')
        conv10 = slim.conv2d(conv5, 3, [3, 3], rate=1, activation_fn=None, scope='de_conv10')
        out = tf.sigmoid(conv10)
        return out


def Illumination_adjust_net(input_i, input_ratio, isInferenced=False):
    with tf.variable_scope('I_enhance_Net', reuse=tf.AUTO_REUSE):
        if not isInferenced:
            input_all = tf.concat([input_i, input_ratio], 3)
        else:
            input_all = input_i

        conv1 = slim.conv2d(input_all, 32, [3, 3], rate=1, activation_fn=lrelu, scope='conv_1')
        conv2 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='conv_2')
        conv3 = slim.conv2d(conv2, 32, [3, 3], rate=1, activation_fn=lrelu, scope='conv_3')
        conv4 = slim.conv2d(conv3, 1, [3, 3], rate=1, activation_fn=lrelu, scope='conv_4')

        L_enhance = tf.sigmoid(conv4)
    return L_enhance


def Illumination_adjust_curve_net(x):
    with tf.variable_scope('I_enhance_Net', reuse=tf.AUTO_REUSE):
        # todo: 128 channel
        x1 = slim.conv2d(x, 32, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1, scope='conv_1')
        x2 = slim.conv2d(x1, 32, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1, scope='conv_2')
        x3 = slim.conv2d(x2, 32, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1, scope='conv_3')
        x4 = slim.conv2d(x3, 32, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1, scope='conv_4')

        x5 = slim.conv2d(tf.concat([x3, x4], 3), 32, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1,
                         scope='conv_5')
        x6 = slim.conv2d(tf.concat([x2, x5], 3), 32, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1,
                         scope='conv_6')
        xr = slim.conv2d(tf.concat([x1, x6], 3), 8, [3, 3], rate=1, activation_fn=tf.nn.tanh, padding='same', stride=1,
                         scope='conv_7')

        # print(xr.shape, len(tf.split(xr, 8, axis=3)))
        r1, r2, r3, r4, r5, r6, r7, r8 = tf.split(xr, 8, axis=3)

        x = x + r1 * (tf.pow(x, 2) - x)
        x = x + r2 * (tf.pow(x, 2) - x)
        x = x + r3 * (tf.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (tf.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (tf.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (tf.pow(x, 2) - x)
        x = x + r7 * (tf.pow(x, 2) - x)
        enhance_image = x + r8 * (tf.pow(x, 2) - x)
        r = tf.concat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        L_enhance = tf.sigmoid(enhance_image)
    return L_enhance, r


def Illumination_adjust_curve_net_ratio(x, ratio):
    with tf.variable_scope('I_enhance_Net_ratio', reuse=tf.AUTO_REUSE):
        in_data = tf.concat([x, ratio], 3)
        # todo: 128 channel
        x1 = slim.conv2d(x, 32, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1, scope='conv_1')
        x2 = slim.conv2d(x1, 32, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1, scope='conv_2')
        x3 = slim.conv2d(x2, 32, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1, scope='conv_3')
        x4 = slim.conv2d(x3, 32, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1, scope='conv_4')

        x5 = slim.conv2d(tf.concat([x3, x4], 3), 32, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1,
                         scope='conv_5')
        x6 = slim.conv2d(tf.concat([x2, x5], 3), 32, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1,
                         scope='conv_6')
        xr = slim.conv2d(tf.concat([x1, x6], 3), 8, [3, 3], rate=1, activation_fn=tf.nn.tanh, padding='same', stride=1,
                         scope='conv_7')

        # print(xr.shape, len(tf.split(xr, 8, axis=3)))
        r1, r2, r3, r4, r5, r6, r7, r8 = tf.split(xr, 8, axis=3)

        x = x + r1 * (tf.pow(x, 2) - x)
        x = x + r2 * (tf.pow(x, 2) - x)
        x = x + r3 * (tf.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (tf.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (tf.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (tf.pow(x, 2) - x)
        x = x + r7 * (tf.pow(x, 2) - x)
        enhance_image = x + r8 * (tf.pow(x, 2) - x)
        r = tf.concat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        L_enhance = tf.sigmoid(enhance_image)
    return L_enhance, r


def Reinforcement_Net(r, i, origin_image):
    with tf.variable_scope('Reinforcement_Net', reuse=tf.AUTO_REUSE):
        # res = r * i
        # input = tf.concat([res, origin_image], 3)
        input = tf.concat([r, i, origin_image], 3)
        # enhancement net
        x1 = slim.conv2d(input, 128, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1, scope='conv_1')
        x2 = slim.conv2d(x1, 128, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1, scope='conv_2')
        x3 = slim.conv2d(x2, 128, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1, scope='conv_3')
        x4 = slim.conv2d(x3, 128, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1, scope='conv_4')
        x5 = slim.conv2d(x4, 3, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1, scope='conv_5')
        # origin_x5 = tf.concat([x5, origin_image], 3)
        # x6 = slim.conv2d(origin_x5, 32, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1, scope='conv_6')
        # x7 = slim.conv2d(x6, 32, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1, scope='conv_7')
        # x8 = slim.conv2d(x7, 32, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1, scope='conv_8')
        # x9 = slim.conv2d(x8, 32, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1, scope='conv_9')
        # x10 = slim.conv2d(x9, 3, [3, 3], rate=1, activation_fn=lrelu, padding='same', stride=1, scope='conv_10')
        L_enhance = tf.sigmoid(x5)
    return L_enhance


def Reinforcement_UNet(r, i, origin, reuse=False):
    with tf.variable_scope("Reinforcement_UNet", reuse=reuse):
        res = r * i
        in_data = tf.concat([res, origin], 3)

        in_data = tf.concat([r, i], 3)
        # Conv1 + Crop1
        conv1_1 = tf.layers.conv2d(res, 64, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))  # Use Xavier init.
        conv1_2 = tf.layers.conv2d(conv1_1, 64, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        crop1 = tf.keras.layers.Cropping2D(cropping=((90, 90), (90, 90)))(conv1_2)

        # MaxPooling1 + Conv2 + Crop2
        pool1 = tf.layers.max_pooling2d(conv1_2, 2, 2)
        # Arguments: inputs, pool_size(integer or tuple), strides(integer or tuple),
        # padding='valid'.
        conv2_1 = tf.layers.conv2d(pool1, 128, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv2_2 = tf.layers.conv2d(conv2_1, 128, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        crop2 = tf.keras.layers.Cropping2D(cropping=((41, 41), (41, 41)))(conv2_2)

        # MaxPooling2 + Conv3 + Crop3
        pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2)
        # Arguments: inputs, pool_size(integer or tuple), strides(integer or tuple),
        # padding='valid'.
        conv3_1 = tf.layers.conv2d(pool2, 256, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv3_2 = tf.layers.conv2d(conv3_1, 256, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        crop3 = tf.keras.layers.Cropping2D(cropping=((16, 17), (16, 17)))(conv3_2)

        # MaxPooling3 + Conv4 + Drop4 + Crop4
        pool3 = tf.layers.max_pooling2d(conv3_2, 2, 2)
        # Arguments: inputs, pool_size(integer or tuple), strides(integer or tuple),
        # padding='valid'.
        conv4_1 = tf.layers.conv2d(pool3, 512, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv4_2 = tf.layers.conv2d(conv4_1, 512, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        drop4 = tf.layers.dropout(conv4_2)
        # Arguments: inputs, rate=0.5.
        crop4 = tf.keras.layers.Cropping2D(cropping=((4, 4), (4, 4)))(drop4)

        # MaxPooling4 + Conv5 + Crop5
        pool4 = tf.layers.max_pooling2d(drop4, 2, 2)
        # Arguments: inputs, pool_size(integer or tuple), strides(integer or tuple),
        # padding='valid'.
        conv5_1 = tf.layers.conv2d(pool4, 1024, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv5_2 = tf.layers.conv2d(conv5_1, 1024, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        drop5 = tf.layers.dropout(conv5_2)

        # Upsampling6 + Conv + Merge6
        up6_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(drop5)
        up6 = tf.layers.conv2d(up6_1, 512, 2, padding="SAME", activation=tf.nn.relu,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        merge6 = tf.concat([crop4, up6], axis=3)  # concat channel
        # values: A list of Tensor objects or a single Tensor.

        # Conv6 + Upsampling7 + Conv + Merge7
        conv6_1 = tf.layers.conv2d(merge6, 512, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv6_2 = tf.layers.conv2d(conv6_1, 512, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

        up7_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6_2)
        up7 = tf.layers.conv2d(up7_1, 256, 2, padding="SAME", activation=tf.nn.relu,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        merge7 = tf.concat([crop3, up7], axis=3)  # concat channel

        # Conv7 + Upsampling8 + Conv + Merge8
        conv7_1 = tf.layers.conv2d(merge7, 256, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv7_2 = tf.layers.conv2d(conv7_1, 256, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

        up8_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7_2)
        up8 = tf.layers.conv2d(up8_1, 128, 2, padding="SAME", activation=tf.nn.relu,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        merge8 = tf.concat([crop2, up8], axis=3)  # concat channel

        # Conv8 + Upsampling9 + Conv + Merge9
        conv8_1 = tf.layers.conv2d(merge8, 128, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv8_2 = tf.layers.conv2d(conv8_1, 128, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

        up9_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8_2)
        up9 = tf.layers.conv2d(up9_1, 64, 2, padding="SAME", activation=tf.nn.relu,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        merge9 = tf.concat([crop1, up9], axis=3)  # concat channel

        # Conv9
        conv9_1 = tf.layers.conv2d(merge9, 64, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv9_2 = tf.layers.conv2d(conv9_1, 64, 3, activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv9_3 = tf.layers.conv2d(conv9_2, 3, 3, padding="SAME", activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

        # Conv10
        conv10 = tf.layers.conv2d(conv9_3, 3, 1,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        L_enhance = tf.sigmoid(conv10)
    return L_enhance
