import os
import time
import glob
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
import random
from datetime import datetime

from modules import BasicConvLSTMCell
from modules.model_flownet import *
from modules.model_easyflow import *
from modules.videosr_ops import *
from modules.utils import *
from modules.SSIM_Index import *
from utils import get_num_params,cv2_imsave,cv2_imread
from tqdm import tqdm,trange
from model.base_model import VSR

'''This work tries to rebuild DRVSR (Detail-revealing Deep Video Super-resolution).
This code is based on the original code of DRVSR https://github.com/jiangsutx/SPMC_VideoSR.
'''

class DRVSR(VSR):
    def __init__(self):
        self.num_frames = 3
        self.num_block = 1
        self.crop_size = 100
        self.scale_factor = 4
        self.gt_size=self.crop_size*self.scale_factor
        self.eval_in_size=[128,240]
        self.batch_size = 10
        self.eval_batch_size=4
        self.end_lr=1e-4
        self.reload=True
        self.max_steps = int(1.5e5+1)
        self.lstm_loss_weight = np.linspace(0.5, 1.0, self.num_frames)
        self.lstm_loss_weight = self.lstm_loss_weight / np.sum(self.lstm_loss_weight)
        self.learning_rate = 1e-3
        self.beta1 = 0.9
        self.decay_step=1.2e5

        self.pathlist = open('./data/filelist_train.txt', 'rt').read().splitlines()
        random.shuffle(self.pathlist)
        self.vallist = open('./data/filelist_val.txt', 'rt').read().splitlines()
        self.save_dir='./checkpoint/drvsr'
        self.log_dir='./drvsr.txt'

        # flownets is a object of class FLOWNETS
        self.flownets = EASYFLOW()

    def input_producer(self, batch_size=10):
        def read_data():
            idx0 = self.num_frames // 2
            data_seq = tf.random_crop(self.data_queue, [2, self.num_frames])
            input = tf.stack(
                [tf.image.decode_png(tf.read_file(data_seq[0][i]), channels=3) for i in range(self.num_frames)])
            gt = tf.stack([tf.image.decode_png(tf.read_file(data_seq[1][idx0]), channels=3)])
            # gt = tf.stack([tf.image.decode_png(tf.read_file(data_seq[1][i]), channels=3) for i in range(self.num_frames)])
            input, gt = prepprocessing(input, gt)
            print('Input producer shape: ', input.get_shape(), gt.get_shape())
            return input, gt

        def prepprocessing(input, gt=None):
            input = tf.cast(input, tf.float32) / 255.0
            gt = tf.cast(gt, tf.float32) / 255.0

            shape = tf.shape(input)[1:]
            size = tf.convert_to_tensor([self.crop_size, self.crop_size, 3], dtype=tf.int32, name="size")
            check = tf.Assert(tf.reduce_all(shape >= size), ["Need value.shape >= size, got ", shape, size])
            shape = control_flow_ops.with_dependencies([check], shape)

            limit = shape - size + 1
            offset = tf.random_uniform(tf.shape(shape), dtype=size.dtype, maxval=size.dtype.max, seed=None) % limit

            offset_in = tf.concat([[0], offset], axis=-1)
            size_in = tf.concat([[self.num_frames], size], axis=-1)
            input = tf.slice(input, offset_in, size_in)
            offset_gt = tf.concat([[0], offset[:2] * self.scale_factor, [0]], axis=-1)
            size_gt = tf.concat([[1], size[:2] * self.scale_factor, [3]], axis=-1)
            gt = tf.slice(gt, offset_gt, size_gt)

            input.set_shape([self.num_frames, self.crop_size, self.crop_size, 3])
            gt.set_shape([1, self.crop_size * self.scale_factor, self.crop_size * self.scale_factor, 3])
            return input, gt

        with tf.variable_scope('input'):
            inList_all = []
            gtList_all = []
            for dataPath in self.pathlist:
                inList = sorted(glob.glob(os.path.join(dataPath, 'blur{}/*.png'.format(self.scale_factor))))
                gtList = sorted(glob.glob(os.path.join(dataPath, 'truth/*.png')))
                inList_all.append(inList)
                gtList_all.append(gtList)
            inList_all = tf.convert_to_tensor(inList_all, dtype=tf.string)
            gtList_all = tf.convert_to_tensor(gtList_all, dtype=tf.string)

            self.data_queue = tf.train.slice_input_producer([inList_all, gtList_all], capacity=20)
            input, gt = read_data()
            batch_in, batch_gt = tf.train.batch([input, gt], batch_size=batch_size, num_threads=3, capacity=20)
        return batch_in, batch_gt

    def forward(self, frames_lr, is_training=True, reuse=False):
        num_batch, num_frame, height, width, num_channels = frames_lr.get_shape().as_list()
        out_height = height * self.scale_factor
        out_width = width * self.scale_factor
        # calculate flow
        idx0 = num_frame // 2
        frames_y = rgb2y(frames_lr)
        frame_ref_y = frames_y[:, int(idx0), :, :, :]
        self.frames_y = frames_y
        self.frame_ref_y = frame_ref_y

        #frame_0up_ref = zero_upsampling(frame_ref_y, scale_factor=self.scale_factor)
        frame_bic_ref = tf.image.resize_images(frame_ref_y, [out_height, out_width], method=2)


        x_unwrap = []
        with tf.variable_scope('LSTM'):
            cell = BasicConvLSTMCell.BasicConvLSTMCell([out_height // 4, out_width // 4], [3, 3], 128)
            rnn_state = cell.zero_state(batch_size=num_batch, dtype=tf.float32)

        self.uv = []
        frame_i_fw_all = []
        max_feature=64

        for i in range(num_frame):
            if i >0 and not reuse:
                reuse=True
            frame_i = frames_y[:, i, :, :, :]
            if i == 0:
                #                with tf.variable_scope('easyflow',reuse=False):
                #                    uv = self.flownets.forward(frame_i, frame_ref_y)
                uv = self.flownets.forward(frame_i, frame_ref_y, reuse=reuse)
            else:
                #                with tf.variable_scope('easyflow',reuse=True):
                #                    uv = self.flownets.forward(frame_i, frame_ref_y,reuse=True)
                uv = self.flownets.forward(frame_i, frame_ref_y, reuse=True)
            self.uv.append(uv)
            self.frames_y = frames_y
            self.frame_ref_y = frame_ref_y
            print('Build model - frame_{}'.format(i), frame_i.get_shape(), uv.get_shape())

            frame_i_fw = imwarp_forward(uv, tf.concat([frame_i], -1), [out_height, out_width])
            if i==0:
                tem=imwarp_forward(uv, tf.concat([frame_i], -1), [height, width])
                frame_i_fw_all=tem
            else:
                tem = imwarp_forward(uv, tf.concat([frame_i], -1), [height, width])
                frame_i_fw_all=tf.concat([frame_i_fw_all,tem],axis=0)

            with tf.variable_scope('srmodel', reuse=reuse):
                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, stride=1,
                                    weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                    biases_initializer=tf.constant_initializer(0.0)), \
                     slim.arg_scope([slim.batch_norm], center=True, scale=False, updates_collections=None,
                                    activation_fn=tf.nn.relu, epsilon=1e-5, is_training=is_training):
                    # conv2d_bn = lambda inputs, num_outputs, kernel_size, scope: \
                    #     slim.batch_norm(slim.conv2d(inputs, num_outputs, kernel_size, scope='conv' + scope), scope='bn' + scope)
                    # rnn_input = tf.concat([frame_bic_ref, frame_i_fw], 3)
                    rnn_input = tf.concat([frame_i_fw], 3)

                    conv1 = slim.conv2d(rnn_input, 32, [5, 5], scope='enc1')
                    conv2 = slim.conv2d(conv1, 64, [3, 3], stride=2, scope='enc2')
                    conv2_1 = slim.conv2d(conv2, 64, [3, 3], scope='enc2_1')

                    conv3 = slim.conv2d(conv2_1, 128, [3, 3], stride=2, scope='enc3')
                    y_1, rnn_state = cell(conv3, rnn_state)
                    conv3_1 = slim.conv2d(y_1, 128, [3, 3], scope='enc3_1')

                    dec1 = slim.conv2d_transpose(conv3_1, 64, [4, 4], stride=2, scope='dec1')

                    cat3 = dec1 + conv2_1
                    dec1_1 = slim.conv2d(cat3, 64, [3, 3], scope='dec1_1')

                    dec2 = slim.conv2d_transpose(dec1_1, 32, [4, 4], stride=2, scope='dec2')
                    cat2 = dec2 + conv1
                    dec2_1 = slim.conv2d(cat2, 32, [3, 3], scope='dec2_1')
                    rnn_out = slim.conv2d(dec2_1, 1, [5, 5], activation_fn=None, scope='dec3')
                    rnn_out = rnn_out + frame_bic_ref
                if i >= 0:
                    x_unwrap.append(rnn_out)
                if i == 0:
                    tf.get_variable_scope().reuse_variables()

        x_unwrap = tf.stack(x_unwrap, 1)
        self.uv = tf.stack(self.uv, 1)
        return x_unwrap, frame_i_fw_all

    def build_model(self):
        frames_lr, frame_gt = self.input_producer(batch_size=self.batch_size)
        n, t, h, w, c = frames_lr.get_shape().as_list()
        output, frame_i_fw = self.forward(frames_lr)

        # calculate loss
        # reconstruction loss
        frame_gt_y = rgb2y(frame_gt)
        mse = tf.reduce_mean((output - frame_gt_y) ** 2, axis=[0, 2, 3, 4])
        self.mse = mse
        # for i in range(self.num_frames):
        #     tf.summary.scalar('mse_%d' % i, mse[i])
        #tf.summary.image('out_0', im2uint8(output[0, :, :, :, :]), max_outputs=3)
        #tf.summary.image('res', im2uint8(output[:, -1, :, :, :]), max_outputs=3)
        #tf.summary.image('gt', im2uint8(frame_gt_y[:, 0, :, :, :]), max_outputs=3)

        self.loss_mse = tf.reduce_sum(mse * self.lstm_loss_weight)
        # tf.summary.scalar('loss_mse', self.loss_mse)

        # flow loss
        frames_ref_warp = imwarp_backward(self.uv,
                                          tf.tile(tf.expand_dims(self.frame_ref_y, 1), [1, self.num_frames, 1, 1, 1]),
                                          [h, w])
        self.loss_flow_data = tf.reduce_mean(tf.abs(self.frames_y - frames_ref_warp))
        uv4d = tf.reshape(self.uv, [self.batch_size * self.num_frames, h, w, 2])
        self.loss_flow_tv = tf.reduce_sum(tf.image.total_variation(uv4d)) / uv4d.shape.num_elements()
        self.loss_flow = self.loss_flow_data + 0.01 * self.loss_flow_tv
        # tf.summary.scalar('loss_flow', self.loss_flow)
        #tf.summary.image('uv', flowToColor(self.uv[0, :, :, :, :], maxflow=3.0), max_outputs=3)

        # total loss
        self.loss = self.loss_mse + self.loss_flow * 0.01
        # tf.summary.scalar('loss_all', self.loss)

    def evaluation(self):
        print('Evaluating ...')
        inList_all = []
        gtList_all = []
        for dataPath in self.vallist:
            inList = sorted(glob.glob(os.path.join(dataPath, 'blur{}/*.png'.format(self.scale_factor))))
            gtList = sorted(glob.glob(os.path.join(dataPath, 'truth/*.png')))
            inList_all.append(inList)
            gtList_all.append(gtList)

        if not hasattr(self, 'sess'):
            sess = tf.Session()
            # self.flownets.load_easyflow(sess, os.path.join('./easyflow_log/model1', 'checkpoints'))
        else:
            sess = self.sess

        border=8
        in_h,in_w=self.eval_in_size
        out_h = in_h*self.scale_factor #512
        out_w = in_w*self.scale_factor #960
        bd=border//self.scale_factor

        if not hasattr(self, 'eval_input'):
            self.eval_input = tf.placeholder(tf.float32, [self.eval_batch_size, self.num_frames, in_h, in_w, 3])
            self.eval_gt = tf.placeholder(tf.float32, [self.eval_batch_size, 1, out_h, out_w, 3])
            self.eval_output, frame_i_fw = self.forward(self.eval_input, is_training=False, reuse=True)

            # calculate loss
            frame_gt_y = rgb2y(self.eval_gt)
            self.eval_mse = tf.reduce_mean((self.eval_output[:, :, :, :, :] - frame_gt_y) ** 2, axis=[2, 3, 4])

        batch_in = []
        batch_gt = []
        radius = self.num_frames // 2
        mse_acc = None
        ssim_acc = None
        batch_cnt = 0
        for inList, gtList in zip(inList_all, gtList_all):
            for idx0 in range(15, len(inList), 32):
                inp = [cv2_imread(inList[0]) for i in range(idx0 - radius, 0)]
                inp.extend([cv2_imread(inList[i]) for i in range(max(0, idx0 - radius), idx0)])
                inp.extend([cv2_imread(inList[i]) for i in range(idx0, min(len(inList), idx0 + radius + 1))])
                inp.extend([cv2_imread(inList[-1]) for i in range(idx0 + radius, len(inList) - 1, -1)])
                inp = [i[bd:in_h+bd, bd:in_w+bd].astype(np.float32) / 255.0 for i in inp]
                gt = [cv2_imread(gtList[idx0])]
                gt = [i[border:out_h+border, border:out_w+border, :].astype(np.float32) / 255.0 for i in gt]

                batch_in.append(np.stack(inp, axis=0))
                batch_gt.append(np.stack(gt, axis=0))

                if len(batch_in) == self.eval_batch_size:
                    batch_cnt += self.eval_batch_size
                    batch_in = np.stack(batch_in, 0)
                    batch_gt = np.stack(batch_gt, 0)
                    mse_val, eval_output_val = sess.run([self.eval_mse, self.eval_output],
                                                        feed_dict={self.eval_input: batch_in, self.eval_gt: batch_gt})
                    ssim_val = np.array(
                        [[compute_ssim(eval_output_val[ib, it, :, :, 0], batch_gt[ib, 0, :, :, 0], l=1.0)
                          for it in range(self.num_frames)] for ib in range(self.eval_batch_size)])
                    if mse_acc is None:
                        mse_acc = mse_val
                        ssim_acc = ssim_val
                    else:
                        mse_acc = np.concatenate([mse_acc, mse_val], axis=0)
                        ssim_acc = np.concatenate([ssim_acc, ssim_val], axis=0)

                    batch_in = []
                    batch_gt = []
                    print('\tEval batch {} - {} ...'.format(batch_cnt, batch_cnt + self.eval_batch_size))

        psnr_acc = 10 * np.log10(1.0 / mse_acc)
        mse_avg = np.mean(mse_acc, axis=0)
        psnr_avg = np.mean(psnr_acc, axis=0)
        ssim_avg = np.mean(ssim_acc, axis=0)
        # for i in range(mse_avg.shape[0]):
        #     tf.summary.scalar('val_mse{}'.format(i), tf.convert_to_tensor(mse_avg[i], dtype=tf.float32))
        print('Eval MSE: {}, PSNR: {}'.format(mse_avg, psnr_avg))
        # write to log file
        with open(self.log_dir, 'a+') as f:
            mse_avg=(mse_avg*1e8).astype(np.int64)/(1e8)
            psnr_avg=(psnr_avg*1e8).astype(np.int64)/(1e8)
            ssim_avg=(ssim_avg*1e8).astype(np.int64)/(1e8)
            f.write('{'+'"Iter": {} , "MSE": {}, "PSNR": {}, "SSIM": {}'.format(sess.run(self.global_step), mse_avg.tolist(), psnr_avg.tolist(),
                                                                     ssim_avg.tolist())+'}\n')
        '''np.save(os.path.join(self.train_dir, 'eval_iter_{}'.format(sess.run(self.global_step))),
                {'mse': mse_acc, 'psnr': psnr_acc, 'ssim': ssim_acc})'''

    def train(self):
        def train_op_func(loss, var_list, is_gradient_clip=False):
            if is_gradient_clip:
                train_op = tf.train.AdamOptimizer(lr, self.beta1)
                grads_and_vars = train_op.compute_gradients(loss, var_list=var_list)
                unchanged_gvs = [(grad, var) for grad, var in grads_and_vars if not 'LSTM' in var.name]
                rnn_grad = [grad for grad, var in grads_and_vars if 'LSTM' in var.name]
                rnn_var = [var for grad, var in grads_and_vars if 'LSTM' in var.name]
                capped_grad, _ = tf.clip_by_global_norm(rnn_grad, clip_norm=3)
                capped_gvs = list(zip(capped_grad, rnn_var))
                train_op = train_op.apply_gradients(grads_and_vars=capped_gvs + unchanged_gvs, global_step=global_step)
            else:
                # train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss, var_list=var_list, global_step=global_step)
                train_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=var_list, global_step=global_step)
            return train_op

        """Train video sr network"""
        global_step = tf.Variable(initial_value=0, trainable=False)
        self.global_step = global_step

        # Create folder for logs
        if not tf.gfile.Exists(self.save_dir):
            tf.gfile.MakeDirs(self.save_dir)

        self.build_model()
        lr = tf.train.polynomial_decay(self.learning_rate, global_step, self.decay_step, end_learning_rate=self.end_lr, power=0.9)
        # tf.summary.scalar('learning_rate', lr)
        vars_all = tf.trainable_variables()
        vars_sr = [v for v in vars_all if 'srmodel' in v.name]
        vars_flow = [v for v in vars_all if 'flow' in v.name]
        train_all = train_op_func(self.loss, vars_all, is_gradient_clip=True)
        train_flow = train_op_func(self.loss_flow, vars_flow, is_gradient_clip=True)
        train_sr = train_op_func(self.loss_mse, vars_sr, is_gradient_clip=True)

        print('params num of flow:',get_num_params(vars_flow))
        print('params num of sr:',get_num_params(vars_sr))
        print('params num of all:',get_num_params(vars_all))

        config = tf.ConfigProto() 
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        #sess=tf.Session()
        self.sess=sess
        sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)
        #self.flownets.load_easyflow(sess, os.path.join('./easyflow_log/model1', 'checkpoints'))
        if self.reload:
            self.load(sess, self.save_dir)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # summary_op = tf.summary.merge_all()
        # summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph, flush_secs=30)
        

        cost_time=0
        start_time=time.time()
        gs=sess.run(global_step)
        for step in range(sess.run(global_step), self.max_steps):
            if step < 10000:
                train_op = train_sr #train_flow
            else:
                train_op = train_all

            if step>gs and step % 20 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),'Step:{}, loss:({:.3f},{:.3f},{:.3f}), mse:{}'.format(step,
                                    loss_value,loss_mse_value,loss_flow_value * 100,str(mse_value)))

            # if step % 50 == 0:
            #     # summary_str = sess.run(summary_op, feed_dict={inputs:batch_input, gt:batch_gt})
            #     summary_str = sess.run(summary_op)
            #     summary_writer.add_summary(summary_str, global_step=step)

            if step % 500 == 0:
                if step>gs:
                    self.save(sess, self.save_dir, step)
                cost_time=time.time()-start_time
                print('cost {}s.'.format(cost_time))
                self.evaluation()
                cost_time=time.time()-start_time
                start_time=time.time()
                print('cost {}s.'.format(cost_time))

            _, loss_value, mse_value, loss_mse_value, loss_flow_value = sess.run(
                [train_op, self.loss, self.mse, self.loss_mse, self.loss_flow])
            # print (loss_value)
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'


    def save(self, sess, checkpoint_dir, step):
        model_name = "videoSR.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None):
        print(" [*] Reading SR checkpoints...")
        model_name = "videoSR.model"

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading checkpoints...{} Success".format(ckpt_name))
            return True
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False
                
    def testvideo(self, dataPath=None, savename='result', reuse=False, scale_factor=4, num_frames=3):
        inList = sorted(glob.glob(os.path.join(dataPath, 'blur{}/*.png').format(scale_factor)))
        inp = [cv2_imread(i).astype(np.float32) / 255.0 for i in inList]

        print('Testing path: {}'.format(dataPath))
        print('# of testing frames: {}'.format(len(inList)))
        max_frame=len(inList)

        savepath = os.path.join(dataPath,savename)
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        self.scale_factor = scale_factor

        all_time=[]
        for idx0 in trange(len(inList)):
            T = num_frames // 2

            imgs = [inp[0] for i in np.arange(idx0 - T, 0)]
            imgs.extend([inp[i] for i in np.arange(max(0, idx0 - T), idx0)])
            imgs.extend([inp[i] for i in np.arange(idx0, min(len(inList), idx0 + T + 1))])
            imgs.extend([inp[-1] for i in np.arange(idx0 + T, len(inList) - 1, -1)])

            dims = imgs[0].shape
            if len(dims) == 2:
                imgs = [np.expand_dims(i, -1) for i in imgs]
            h, w, c = imgs[0].shape
            out_h = h * scale_factor
            out_w = w * scale_factor
            padh = int(ceil(h / 4.0) * 4.0 - h)
            padw = int(ceil(w / 4.0) * 4.0 - w)
            imgs = [np.pad(i, [[0, padh], [0, padw], [0, 0]], 'edge') for i in imgs]
            imgs = np.expand_dims(np.stack(imgs, axis=0), 0)

            if idx0 == 0:
                # frames_lr = tf.convert_to_tensor(imgs, tf.float32)
                frames_lr = tf.placeholder(dtype=tf.float32, shape=imgs.shape)
                frames_ref_ycbcr = rgb2ycbcr(frames_lr[:, T:T + 1, :, :, :])
                frames_ref_ycbcr = tf.tile(frames_ref_ycbcr, [1, num_frames, 1, 1, 1])
                output, frame_i_fw = self.forward(frames_lr, is_training=False, reuse=reuse)
                # print (frames_lr_ycbcr.get_shape(), h, w, padh, padw)
                if len(dims) == 3:
                    output_rgb = ycbcr2rgb(tf.concat([output, resize_images(frames_ref_ycbcr
                                                                            , [(h + padh) * scale_factor
                                                                                , (w + padw) * scale_factor]
                                                                            , method=2)[:, :, :, :, 1:3]], -1))
                else:
                    output_rgb = output
                output = output[:, :, :out_h, :out_w, :]
                output_rgb = output_rgb[:, :, :out_h, :out_w, :]

            if reuse == False:
                config = tf.ConfigProto() 
                config.gpu_options.allow_growth = True
                sess = tf.Session(config=config)
                #sess=tf.Session()
                self.sess=sess
                # summary_op = tf.summary.merge_all()
                # summary_writer = tf.summary.FileWriter('E:/PyProject/out0', sess.graph, flush_secs=30)
                self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
                self.load(self.sess, self.save_dir)
                #self.flownets.load_easyflow(self.sess, os.path.join('./easyflow_log/model1', 'checkpoints'))
                reuse=True
            # frames_ref_warp = imwarp_backward(self.uv, tf.tile(tf.expand_dims(self.frame_ref_y, 1), [1, num_frames, 1, 1, 1]),
            #                                   [200, 244])
            # warp_err = tf.reduce_mean((self.frames_y - frames_ref_warp) ** 2, axis=4)
            case_path = dataPath.split('/')[-1]
            # print('Testing - ', case_path, len(imgs))
            st_time=time.time()
            [imgs_hr, imgs_hr_rgb, uv, frame_fw] = self.sess.run([output, output_rgb, self.uv, frame_i_fw],
                                                            feed_dict={frames_lr: imgs})
            all_time.append(time.time()-st_time)
            # print('spent {} s'.format(ed_time-st_time))

            '''cv2_imsave(os.path.join(savepath, 'y_%03d.png' % (idx0)),
                              im2uint8(imgs_hr[0, -1, :, :, 0]))'''
            '''for q in range(0, 3):
                cv2_imsave(os.path.join(savepath, 'y_ff%03d_%01d.png' % (idx0, q)),
                                  im2uint8(frame_fw[q, :, :, 0]))'''
            if len(dims) == 3:
                cv2_imsave(os.path.join(savepath, '%04d.png' % (idx0)),
                                  im2uint8(imgs_hr_rgb[0, -1, :, :, :]))

                # summary_str = self.sess.run(summary_op)
                # summary_writer.add_summary(summary_str, idx0)
        print('SR results path: {}'.format(savepath))
        if max_frame>0:
            all_time=np.array(all_time)
            print('spent {} s in total and {} s in average'.format(np.sum(all_time),np.mean(all_time[1:])))

    def testvideos(self,datapath='/dev/f/data/video/test2/vid4',start=0,savename='drvsr'):
        kind=sorted(glob.glob(os.path.join(datapath,'*')))
        kind=[k for k in kind if os.path.isdir(k)]
        reuse=False
        for i in kind:
            idx=kind.index(i)
            if idx>=start:
                if idx>start:
                    reuse=True
                self.testvideo(i, savename=savename,reuse=reuse)


if __name__ == '__main__':
    model = DRVSR()
    #model.train()
    model.testvideos('/dev/f/data/video/test2/vid4', 0, savename='drvsr')

