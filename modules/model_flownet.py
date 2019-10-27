from math import ceil
import tensorflow as tf

slim = tf.contrib.slim

from modules.videosr_ops import *
from modules.utils import *


class FLOWNETS(object):
    def __init__(self, device='/gpu:0'):
        self.device = device

    def load_flownets(self, sess, model_file='./flownets_model'):
        t_vars = tf.trainable_variables()
        flownets_var = [var for var in t_vars if 'flownets' in var.name]
        saver = tf.train.Saver(var_list=flownets_var)
        saver.restore(sess, model_file)
        print('Loading Flownet-s model... OK!')

    def forward(self, inputs, scope='flownets', reuse=False):
        with tf.variable_scope(scope, reuse=reuse) as sc:
            _, height, width, _ = inputs.get_shape().as_list()
            divisor = 64
            adapted_width = ceil(1.0 * width / divisor) * divisor
            adapted_height = ceil(1.0 * height / divisor) * divisor
            rescale_coeff_x = 1.0 * width / adapted_width
            rescale_coeff_y = 1.0 * height / adapted_height

            inputs = inputs - tf.reduce_mean(inputs, reduction_indices=[1, 2], keep_dims=True)
            inputs = tf.image.resize_images(inputs, [int(adapted_height), int(adapted_width)], align_corners=True)

            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=leaky_relu, stride=2,
                                padding='SAME',
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                biases_initializer=tf.constant_initializer(0.0)), \
                 tf.device(self.device):
                conv1 = slim.conv2d(inputs, 64, [7, 7], stride=1, scope='conv1')[:, 0::2, 0::2, :]
                conv2 = slim.conv2d(conv1, 128, [5, 5], stride=1, scope='conv2')[:, 0::2, 0::2, :]
                conv3 = slim.conv2d(conv2, 256, [5, 5], stride=1, scope='conv3')[:, 0::2, 0::2, :]
                conv3_1 = slim.conv2d(conv3, 256, [3, 3], stride=1, scope='conv3_1')

                conv4 = slim.conv2d(conv3_1, 512, [3, 3], stride=1, scope='conv4')[:, 0::2, 0::2, :]
                conv4_1 = slim.conv2d(conv4, 512, [3, 3], stride=1, scope='conv4_1')
                conv5 = slim.conv2d(conv4_1, 512, [3, 3], stride=1, scope='conv5')[:, 0::2, 0::2, :]
                conv5_1 = slim.conv2d(conv5, 512, [3, 3], stride=1, scope='conv5_1')
                conv6 = slim.conv2d(conv5_1, 1024, [3, 3], stride=1, scope='conv6')[:, 0::2, 0::2, :]
                conv6_1 = slim.conv2d(conv6, 1024, [3, 3], stride=1, scope='conv6_1')

                deconv5 = slim.conv2d_transpose(conv6_1, 512, [4, 4], scope='deconv5')
                predict_flow6 = slim.conv2d(conv6_1, 2, [3, 3], stride=1, activation_fn=None, scope='Convolution1')
                upsampled_flow6_to_5 = slim.conv2d_transpose(predict_flow6, 2, [4, 4], activation_fn=None,
                                                             scope='upsample_flow6to5')

                concat5 = tf.concat([conv5_1, deconv5, upsampled_flow6_to_5], 3, name="Concat2")

                deconv4 = slim.conv2d_transpose(concat5, 256, [4, 4], scope='deconv4')

                predict_flow5 = slim.conv2d(concat5, 2, [3, 3], stride=1, activation_fn=None, scope='Convolution2')
                upsampled_flow5_to_4 = slim.conv2d_transpose(predict_flow5, 2, [4, 4], activation_fn=None,
                                                             scope='upsample_flow5to4')

                concat4 = tf.concat([conv4_1, deconv4, upsampled_flow5_to_4], 3, name='Concat3')
                deconv3 = slim.conv2d_transpose(concat4, 128, [4, 4], scope='deconv3')
                predict_flow4 = slim.conv2d(concat4, 2, [3, 3], stride=1, activation_fn=None, scope='Convolution3')
                upsampled_flow4_to_3 = slim.conv2d_transpose(predict_flow4, 2, [4, 4], activation_fn=None,
                                                             scope='upsample_flow4to3')

                concat3 = tf.concat([conv3_1, deconv3, upsampled_flow4_to_3], 3, name='Concat4')
                deconv2 = slim.conv2d_transpose(concat3, 64, [4, 4], scope='deconv2')
                predict_flow3 = slim.conv2d(concat3, 2, [3, 3], stride=1, activation_fn=None, scope='Convolution4')
                upsampled_flow3_to_2 = slim.conv2d_transpose(predict_flow3, 2, [4, 4], activation_fn=None,
                                                             scope='upsample_flow3to2')

                concat2 = tf.concat([conv2, deconv2, upsampled_flow3_to_2], 3, name="Concat5")
                predict_flow2 = slim.conv2d(concat2, 2, [3, 3], stride=1, activation_fn=None, scope='Convolution5')

                blob44 = predict_flow2 * 20.0
            predict_flow_resize = tf.image.resize_images(blob44, [int(height), int(width)], align_corners=True)
            scale = tf.reshape([rescale_coeff_x, rescale_coeff_y], [1, 1, 1, 2])
            predict_flow_final = predict_flow_resize * scale
        return predict_flow_final

    def uv_conf(self, input_a, input_b):
        [num_batch, height, width, channels] = input_a.get_shape().as_list()
        assert channels == 1 or channels == 3, 'ERROR: uv_conf need input with channel==1 or 3'

        with tf.variable_scope('fusion/uv_conf'):
            input = tf.concat([input_a, input_b], 0)
            input = input - tf.reduce_mean(input, reduction_indices=[1, 2], keep_dims=True)
            input = tf.div(input, tf.reduce_mean(input * input, reduction_indices=[1, 2], keep_dims=True))
            if channels == 1:
                input = tf.tile(input, [1, 1, 1, 3])

            with slim.arg_scope([slim.conv2d], activation_fn=None, padding='SAME',
                                weights_initializer=tf.constant_initializer(0.0),
                                biases_initializer=tf.constant_initializer(0.0)):
                conv_nOutput = [32, 32, 64, 64, 64, 64, 64, 64, 64]
                x = input
                for i in range(8):
                    x = slim.conv2d(x, conv_nOutput[i], [5, 5], scope='conv' + str(i + 1))
                    x = slim.batch_norm(x, epsilon=1e-3, decay=0.9, activation_fn=tf.nn.relu, scope='bn' + str(i + 1))
                x = slim.conv2d(x, conv_nOutput[8], [5, 5], scope='conv9')
                x = slim.batch_norm(x, epsilon=1e-3, decay=0.9, activation_fn=None, scope='bn9')
                output = tf.reduce_sum(
                    tf.nn.l2_normalize(x[:num_batch, :, :, :], dim=3) * tf.nn.l2_normalize(x[num_batch:, :, :, :],
                                                                                           dim=3), reduction_indices=3,
                    keep_dims=True)
                # output = tf.clip_by_value(output, 0.0, 1.0)
                # output = tf.exp(output) / tf.reduce_sum(tf.exp(output), reduction_indices=0, keep_dims=True)
                # output = tf.exp(output) / tf.reduce_sum(tf.exp(output), reduction_indices=0, keep_dims=True)
                output = (output + 1) / 2
        return output

    def load_uv_conf(self, sess, model_file='uv_conf_model'):
        t_vars = tf.trainable_variables()
        uv_conf_var = [var for var in t_vars if 'uv_conf' in var.name]
        saver = tf.train.Saver(var_list=uv_conf_var)
        saver.restore(sess, model_file)
        print('Loading uv_conf model... OK!')

    def test(self, dataPath):
        import os
        import glob
        import numpy as np
        import scipy.misc

        sess = tf.Session()
        inList = sorted(glob.glob(os.path.join(dataPath, 'input/*.png')))
        num_frame = len(inList)
        inputs = [scipy.misc.imread(name).astype(np.float32) / 255.0 for name in inList]
        inputs = tf.stack(map(lambda x: tf.convert_to_tensor(x), inputs))
        n, h, w, c = map(lambda x: x.value, inputs.get_shape())
        out_height = h * 4
        out_width = w * 4
        idx0 = num_frame / 2

        inputs_idx0 = tf.tile(tf.expand_dims(inputs[idx0, :, :, ::-1], 0), [num_frame, 1, 1, 1])
        inputs_uv = tf.concat([inputs[:, :, :, ::-1], inputs_idx0], 3)
        uv = self.forward(inputs_uv)
        # uv = tf.zeros_like(uv)

        inputs_idx0 = tf.tile(tf.expand_dims(inputs[idx0, :, :, :], 0), [num_frame, 1, 1, 1])
        warped_ref = imwarp_backward(uv, inputs_idx0, [h, w])
        # input_weight = tf.concat(3, [inputs, output_warp])
        fusion_w = self.uv_conf(inputs, warped_ref)
        print('fusion_w: ', fusion_w.get_shape())
        # fusion_w = tf.expand_dims(tf.ones_like(inputs[:, :, :, 0]), 3)

        input_forwarp = tf.concat([inputs * fusion_w, fusion_w], 3)
        output_base = imwarp_forward(uv, input_forwarp, [out_height, out_width])
        output_base = tf.reduce_sum(tf.reshape(output_base, [num_frame, out_height, out_width, 4]),
                                    reduction_indices=0)
        output_base = tf.div(output_base[:, :, 0:3], tf.expand_dims(output_base[:, :, 3] + 1e-10, 2))
        output_base = tf.clip_by_value(output_base, 0.0, 1.0)
        inputs_bic = tf.image.resize_bicubic(tf.expand_dims(inputs[idx0, :, :, :], 0), [out_height, out_width])

        vars_all = tf.trainable_variables()
        display_tf_variables(vars_all)
        sess.run(tf.initialize_all_variables())
        self.load_flownets(sess)
        # self.load_uv_conf(sess)

        [res, res_bic, res_w, res_warped_ref] = sess.run([output_base, inputs_bic, fusion_w, warped_ref])
        scipy.misc.imshow(res)
        scipy.misc.imshow(res_w[0, :, :, 0])
        print(res_w[:, 40, 40, 0])
        scipy.misc.imsave('flownets.png', res)
        scipy.misc.imsave('flownets-bic.png', res_bic[0, :, :, :])
        for i in range(31):
            scipy.misc.imsave('flownets-res_w-%02d.png' % i, (res_w[i, :, :, 0] * 20 * 255.0).astype('uint8'))
            # scipy.misc.imsave('flownets-warp-00.png', (res_warped_ref[0, :, :, :]*255.0).astype('uint8'))
            # scipy.misc.imsave('flownets-warp-15.png', (res_warped_ref[idx0, :, :, :]*255.0).astype('uint8'))
    
    def test_uv(self, dataPath):
        import os
        import glob
        import numpy as np
        import scipy.misc
        import flowTools 

        sess = tf.Session()
        
        inList = sorted(glob.glob(os.path.join(dataPath, 'input/*.png')))
        num_frame = len(inList)
        inputs = [scipy.misc.imread(name).astype(np.float32) / 255.0 for name in inList]
        inputs = tf.stack(map(lambda x: tf.convert_to_tensor(x), inputs))
        n, h, w, c = map(lambda x: x.value, inputs.get_shape())
        out_height = h * 4
        out_width = w * 4
        idx0 = num_frame / 2

        inputs_idx0 = tf.tile(tf.expand_dims(inputs[idx0, :, :, ::-1], 0), [num_frame, 1, 1, 1])
        inputs_uv = tf.concat([inputs[:, :, :, ::-1], inputs_idx0], 3)
        uv = self.forward(inputs_uv)
        sess.run(tf.initialize_all_variables())
        self.load_flownets(sess)
        
        uv = sess.run(uv)

        for i in xrange(31):
            scipy.misc.imsave('flownet_{}.png'.format(i), flowTools.flowToColor(uv[i, :, :, :]))


class FLOWNETC(object):
    def __init__(self, device='/gpu:0', caffenet=None):
        self.device = device
        self.caffenet = caffenet

    def load_flownetc(self, sess, model_file='flownetc_model'):
        t_vars = tf.trainable_variables()
        flownets_var = [var for var in t_vars if 'flownetc' in var.name]
        saver = tf.train.Saver(var_list=flownets_var)
        saver.restore(sess, model_file)
        print('Loading Flownet-c model... OK!')

    def layer_corr(self, input_a, input_b, kernel_size=1, max_disp=20, pad=20, stride=[1, 2], name='corr'):
        [n, height, width, c] = map(lambda x: x.value, input_a.get_shape())

        kernel_r = (kernel_size - 1) / 2
        border_size = max_disp + kernel_r

        paddedbottomwidth = width + 2 * pad
        paddedbottomheight = height + 2 * pad
        top_width = ceil((paddedbottomwidth - border_size * 2) / stride[0])
        top_height = ceil((paddedbottomheight - border_size * 2) / stride[0])

        neighborhood_grid_radius_ = max_disp / stride[1]
        neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1

        top_channels_ = neighborhood_grid_width_ * neighborhood_grid_width_

        input_b = tf.pad(input_b, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='CONSTANT')
        output = [tf.reduce_sum(input_a * input_b[:, pad + i:pad + i + height, pad + j:pad + j + width, :],
                                reduction_indices=3, keep_dims=False)
                  for i in range(-max_disp, max_disp + stride[1], stride[1]) for j in
                  range(-max_disp, max_disp + stride[1], stride[1])]

        output = tf.stack(output, axis=3) / 441
        return output

    def forward(self, inputs, scope='flownetc', reuse=False):
        with tf.variable_scope(scope, reuse=reuse) as sc:
            num_batch, height, width, num_channels = map(lambda x: x.value, inputs.get_shape())
            divisor = 64
            adapted_width = ceil(1.0 * width / divisor) * divisor
            adapted_height = ceil(1.0 * height / divisor) * divisor
            rescale_coeff_x = 1.0 * width / adapted_width
            rescale_coeff_y = 1.0 * height / adapted_height

            inputs = inputs - tf.reduce_mean(inputs, reduction_indices=[1, 2], keep_dims=True)
            inputs = tf.image.resize_images(inputs, int(adapted_height), int(adapted_width), align_corners=True)

            inputs = tf.concat([inputs[:, :, :, :num_channels / 2], inputs[:, :, :, num_channels / 2:]], 3)
            if num_channels == 1:
                inputs = tf.expand_dims(input, 3)
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=leaky_relu, stride=2,
                                padding='SAME',
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                biases_initializer=tf.constant_initializer(0.0)), \
                 tf.device(self.device):
                conv1 = slim.conv2d(inputs, 64, [7, 7], stride=1, scope='conv1')[:, 0::2, 0::2, :]
                conv2 = slim.conv2d(conv1, 128, [5, 5], stride=1, scope='conv2')[:, 0::2, 0::2, :]
                conv3 = slim.conv2d(conv2, 256, [5, 5], stride=1, scope='conv3')[:, 0::2, 0::2, :]
                conv3a = conv3[:num_batch, :, :, :]
                conv3b = conv3[num_batch:, :, :, :]

                corr = self.layer_corr(conv3a, conv3b, max_disp=20, stride=[1, 2])

                conv_redir = slim.conv2d(conv3a, 32, [1, 1], stride=1, scope='conv_redir')
                blob20 = tf.concat([conv_redir, corr], 3, name='Concat1')
                conv3_1 = slim.conv2d(blob20, 256, [3, 3], stride=1, scope='conv3_1')

                conv4 = slim.conv2d(conv3_1, 512, [3, 3], stride=1, scope='conv4')[:, 0::2, 0::2, :]
                conv4_1 = slim.conv2d(conv4, 512, [3, 3], stride=1, scope='conv4_1')
                conv5 = slim.conv2d(conv4_1, 512, [3, 3], stride=1, scope='conv5')[:, 0::2, 0::2, :]
                conv5_1 = slim.conv2d(conv5, 512, [3, 3], stride=1, scope='conv5_1')
                conv6 = slim.conv2d(conv5_1, 1024, [3, 3], stride=1, scope='conv6')[:, 0::2, 0::2, :]
                conv6_1 = slim.conv2d(conv6, 1024, [3, 3], stride=1, scope='conv6_1')

                deconv5 = slim.conv2d_transpose(conv6_1, 512, [4, 4], scope='deconv5')
                predict_flow6 = slim.conv2d(conv6_1, 2, [3, 3], stride=1, activation_fn=None, scope='Convolution1')
                upsampled_flow6_to_5 = slim.conv2d_transpose(predict_flow6, 2, [4, 4], activation_fn=None,
                                                             scope='upsample_flow6to5')

                concat5 = tf.concat([conv5_1, deconv5, upsampled_flow6_to_5], 3, name="Concat2")

                deconv4 = slim.conv2d_transpose(concat5, 256, [4, 4], scope='deconv4')

                predict_flow5 = slim.conv2d(concat5, 2, [3, 3], stride=1, activation_fn=None, scope='Convolution2')
                upsampled_flow5_to_4 = slim.conv2d_transpose(predict_flow5, 2, [4, 4], activation_fn=None,
                                                             scope='upsample_flow5to4')

                concat4 = tf.concat([conv4_1, deconv4, upsampled_flow5_to_4], 3, name='Concat3')
                deconv3 = slim.conv2d_transpose(concat4, 128, [4, 4], scope='deconv3')
                predict_flow4 = slim.conv2d(concat4, 2, [3, 3], stride=1, activation_fn=None, scope='Convolution3')
                upsampled_flow4_to_3 = slim.conv2d_transpose(predict_flow4, 2, [4, 4], activation_fn=None,
                                                             scope='upsample_flow4to3')

                concat3 = tf.concat([conv3_1, deconv3, upsampled_flow4_to_3], 3, name='Concat4')
                deconv2 = slim.conv2d_transpose(concat3, 64, [4, 4], scope='deconv2')
                predict_flow3 = slim.conv2d(concat3, 2, [3, 3], stride=1, activation_fn=None, scope='Convolution4')
                upsampled_flow3_to_2 = slim.conv2d_transpose(predict_flow3, 2, [4, 4], activation_fn=None,
                                                             scope='upsample_flow3to2')

                concat2 = tf.concat([conv2[:num_batch, :, :, :], 3, deconv2, upsampled_flow3_to_2])
                predict_flow2 = slim.conv2d(concat2, 2, [3, 3], stride=1, activation_fn=None, scope='Convolution5')

                blob44 = predict_flow2 * 20.0
            predict_flow_resize = tf.image.resize_images(blob44, height, width, align_corners=True)
            scale = tf.reshape([rescale_coeff_x, rescale_coeff_y], [1, 1, 1, 2])
            predict_flow_final = tf.mul(predict_flow_resize, scale)
        return predict_flow_final

    def uv_conf(self, input_a, input_b):
        [num_batch, height, width, channels] = map(lambda x: x.value, input_a.get_shape())
        assert channels == 1 or channels == 3, 'ERROR: uv_conf need input with channel==1 or 3'

        with tf.variable_scope('fusion/uv_conf'):
            input = tf.concat([input_a, input_b], 0)
            input = input - tf.reduce_mean(input, reduction_indices=[1, 2], keep_dims=True)
            input = tf.div(input, tf.reduce_mean(input * input, reduction_indices=[1, 2], keep_dims=True))
            if channels == 1:
                input = tf.tile(input, [1, 1, 1, 3])

            with slim.arg_scope([slim.conv2d], activation_fn=None, padding='SAME',
                                weights_initializer=tf.constant_initializer(0.0),
                                biases_initializer=tf.constant_initializer(0.0)):
                conv_nOutput = [32, 32, 64, 64, 64, 64, 64, 64, 64]
                x = input
                for i in range(8):
                    x = slim.conv2d(x, conv_nOutput[i], [5, 5], scope='conv' + str(i + 1))
                    x = slim.batch_norm(x, epsilon=1e-3, decay=0.9, activation_fn=tf.nn.relu, scope='bn' + str(i + 1))
                x = slim.conv2d(x, conv_nOutput[8], [5, 5], scope='conv9')
                x = slim.batch_norm(x, epsilon=1e-3, decay=0.9, activation_fn=None, scope='bn9')
                output = tf.reduce_sum(
                    tf.nn.l2_normalize(x[:num_batch, :, :, :], dim=3) * tf.nn.l2_normalize(x[num_batch:, :, :, :],
                                                                                           dim=3), reduction_indices=3,
                    keep_dims=True)
                # output = tf.clip_by_value(output, 0.0, 1.0)
                # output = tf.exp(output) / tf.reduce_sum(tf.exp(output), reduction_indices=0, keep_dims=True)
                # output = tf.exp(output) / tf.reduce_sum(tf.exp(output), reduction_indices=0, keep_dims=True)
                output = (output + 1) / 2
        return output

    def load_uv_conf(self, sess, model_file='uv_conf_model'):
        t_vars = tf.trainable_variables()
        uv_conf_var = [var for var in t_vars if 'uv_conf' in var.name]
        saver = tf.train.Saver(var_list=uv_conf_var)
        saver.restore(sess, model_file)
        print('Loading uv_conf model... OK!')

    def test(self, dataPath):
        import os
        import glob
        import numpy as np
        import scipy.misc

        sess = tf.Session()
        inList = sorted(glob.glob(os.path.join(dataPath, 'input/*.png')))
        num_frame = len(inList)
        inputs = [scipy.misc.imread(name).astype(np.float32) / 255.0 for name in inList]
        inputs = tf.pack(map(lambda x: tf.convert_to_tensor(x), inputs))
        n, h, w, c = map(lambda x: x.value, inputs.get_shape())
        out_height = h * 4
        out_width = w * 4
        idx0 = num_frame / 2

        inputs_idx0 = tf.tile(tf.expand_dims(inputs[idx0, :, :, ::-1], 0), [num_frame, 1, 1, 1])
        inputs_uv = tf.concat([inputs[:, :, :, ::-1], inputs_idx0], 3)
        uv = self.forward(inputs_uv)
        # uv = tf.zeros_like(uv)

        inputs_idx0 = tf.tile(tf.expand_dims(inputs[idx0, :, :, :], 0), [num_frame, 1, 1, 1])
        warped_ref = imwarp_backward(uv, inputs_idx0, [h, w])
        # input_weight = tf.concat(3, [inputs, output_warp])
        fusion_w = self.uv_conf(inputs, warped_ref)
        print('fusion_w: ', fusion_w.get_shape())
        # fusion_w = tf.expand_dims(tf.ones_like(inputs[:, :, :, 0]), 3)

        input_forwarp = tf.concat([inputs * fusion_w, fusion_w], 3)
        output_base = imwarp_forward(uv, input_forwarp, [out_height, out_width])
        output_base = tf.reduce_sum(tf.reshape(output_base, [num_frame, out_height, out_width, 4]),
                                    reduction_indices=0)
        output_base = tf.div(output_base[:, :, 0:3], tf.expand_dims(output_base[:, :, 3] + 1e-10, 2))
        output_base = tf.clip_by_value(output_base, 0.0, 1.0)
        inputs_bic = tf.image.resize_bicubic(tf.expand_dims(inputs[idx0, :, :, :], 0), [out_height, out_width])

        vars_all = tf.trainable_variables()
        display_tf_variables(vars_all)
        sess.run(tf.initialize_all_variables())
        self.load_flownetc(sess)
        self.load_uv_conf(sess)

        [res, res_bic, res_w, res_warped_ref] = sess.run([output_base, inputs_bic, fusion_w, warped_ref])
        scipy.misc.imshow(res)
        scipy.misc.imshow(res_w[0, :, :, 0])
        print(res_w[:, 40, 40, 0])
        scipy.misc.imsave('flownets.png', res)
        scipy.misc.imsave('flownets-bic.png', res_bic[0, :, :, :])
        for i in range(31):
            scipy.misc.imsave('flownets-res_w-%02d.png' % i, (res_w[i, :, :, 0] * 20 * 255.0).astype('uint8'))
            # scipy.misc.imsave('flownets-warp-00.png', (res_warped_ref[0, :, :, :]*255.0).astype('uint8'))
            # scipy.misc.imsave('flownets-warp-15.png', (res_warped_ref[idx0, :, :, :]*255.0).astype('uint8'))

            # def main(_):

            # flownets = FLOWNETC()
            # # flownets.test(dataPath='/mnt/sda1/xtao/projects/video-sr/data/calendar')
            # # flownets.test(dataPath='/mnt/sda1/xtao/projects/video-sr/data_video/car07_008')
            # flownets.test(dataPath='/mnt/sda1/xtao/projects/video-sr/data_video/grass_002')


# # import sys
# # pycaffe_root = '/mnt/sda1/xtao/projects/dispflownet-release/python/'
# # sys.path.insert(0, pycaffe_root)
# # import caffe
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# sess = tf.Session()
# # # load caffe model and weight
# # caffe_files = ['/mnt/sda1/xtao/projects/dispflownet-release/models/FlowNetC/model/deploy_matlab.prototxt',
# #                '/mnt/sda1/xtao/projects/dispflownet-release/models/FlowNetC/model/flownet_official.caffemodel']
# # caffe.set_mode_cpu()
# # caffe.set_device(6)
# # caffenet = caffe.Net(caffe_files[0], caffe_files[1], caffe.TEST)
# 
# flownetc = FLOWNETS()
# flownetc.test(dataPath='/mnt/sda1/xtao/projects/video-sr/data/calendar')
if __name__ == '__main__':
    model = FLOWNETS()
    model.test_uv(dataPath='/mnt/sda1/xtao/projects/video-sr/data/calendar')