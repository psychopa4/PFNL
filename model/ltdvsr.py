import os
import time
import glob
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
import random
import subprocess
import cv2
from datetime import datetime
from os.path import join,exists

from modules import BasicConvLSTMCell
from modules.model_flownet import *
from modules.model_easyflow import *
from modules.videosr_ops import *
from modules.utils import *
from modules.SSIM_Index import *
import modules.ps
import random
from tensorflow.python.layers.convolutional import Conv2D,conv2d
from utils import cv2_imsave,cv2_imread,automkdir,get_num_params
from tqdm import tqdm,trange
from model.base_model import VSR

'''This work tries to rebuild RVSR-LTD (Robust Video Super-Resolution with Learned Temporal Dynamics).
The code is mainly based on https://github.com/psychopa4/MMCNN and https://github.com/jiangsutx/SPMC_VideoSR.
'''

class LTDVSR(VSR):
    def __init__(self):
        self.num_frames = 5
        self.in_size = 30
        self.scale = 4
        self.gt_size=self.in_size*self.scale
        self.eval_in_size=[128,240]
        self.batch_size = 16
        self.eval_batch_size=4
        self.learning_rate = 1e-3
        self.end_lr=1e-4
        self.reload=True
        self.max_step=int(1.5e5+1)
        self.decay_step=1.2e5
        self.beta1 = 0.9
        self.train_dir='./data/filelist_train.txt'
        self.eval_dir='./data/filelist_val.txt'
        self.save_dir='./checkpoint/ltdvsr'
        self.log_dir='./ltdvsr.txt'

    def forward(self, frames_lr, is_training=True, reuse=False):
        num_batch, num_frame, height, width, num_channels = frames_lr.get_shape().as_list()
        out_height = height * self.scale
        out_width = width * self.scale
        # calculate flow
        idx0 = num_frame // 2
        frames_y = rgb2y(frames_lr)
        frame_ref_y = frames_y[:, int(idx0), :, :, :]
        self.frames_y = frames_y
        self.frame_ref_y = frame_ref_y

        frame_bic_ref = tf.image.resize_images(frame_ref_y, [out_height, out_width], method=2)

        x_unwrap = []

        self.uv = []
        max_feature=64

        frame_i_fw_all = []
        for i in range(num_frame):
            if i > 0 and not reuse:
                reuse = True
            frame_i = frames_y[:, i, :, :, :]
            uv = self.flow(frame_i, frame_ref_y)
            self.uv.append(uv)
            print('Build model - frame_{}'.format(i), frame_i.get_shape(), uv.get_shape())
            # frame_i_bw = imwarp_backward(uv, frame_ref_y, [height, width])
            # frame_i_W = tf.exp(-tf.abs(frame_i_bw - frame_i) * 5)
            frame_i_fw = imwarp_forward(uv, tf.concat([frame_i], -1), [height , width ])
            frame_i_fw_all.append(frame_i_fw)
        # rnn_input=tf.concat([frame_i_fw_all[i] for i in range(num_frame)], 3)

        x=tf.stack(frame_i_fw_all,1)
        activate=tf.nn.relu#tf.nn.leaky_relu
        n,f1,w,h,c=x.shape
        ki=tf.contrib.layers.xavier_initializer()
        ds=1
        with tf.variable_scope('ltdvsr',reuse=tf.AUTO_REUSE) as scope:
            inp0=tf.concat([x[:,f1//2,:,:,:]],-1)
            inp1=tf.concat([x[:,i,:,:,:] for i in range(f1//2-1,f1//2+1+1)],-1)
            inp2=tf.concat([x[:,i,:,:,:] for i in range(f1//2-2,f1//2+2+1)],-1)

            conv0_0=conv2d(inp0,64,5,strides=ds, padding='same', activation=activate, kernel_initializer=ki,name='conv0_0')
            conv0_1=conv2d(conv0_0,64,3,strides=ds, padding='same', activation=activate, kernel_initializer=ki,name='conv0_1')
            conv0_3=conv2d(conv0_1,64,3,strides=ds, padding='same', activation=activate, kernel_initializer=ki,name='conv0_3')
            conv0_2=conv2d(conv0_3,self.scale**2,3,strides=ds, padding='same', activation=None, kernel_initializer=ki,name='conv0_2')

            conv1_0=conv2d(inp1,64,5,strides=ds, padding='same', activation=activate, kernel_initializer=ki,name='conv1_0')
            conv1_1=conv2d(conv1_0,64,3,strides=ds, padding='same', activation=activate, kernel_initializer=ki,name='conv1_1')
            conv1_3=conv2d(conv1_1,64,3,strides=ds, padding='same', activation=activate, kernel_initializer=ki,name='conv1_3')
            conv1_2=conv2d(conv1_3,self.scale**2,3,strides=ds, padding='same', activation=None, kernel_initializer=ki,name='conv1_2')

            conv2_0=conv2d(inp2,64,5,strides=ds, padding='same', activation=activate, kernel_initializer=ki,name='conv2_0')
            conv2_1=conv2d(conv2_0,64,3,strides=ds, padding='same', activation=activate, kernel_initializer=ki,name='conv2_1')
            conv2_3=conv2d(conv2_1,64,3,strides=ds, padding='same', activation=activate, kernel_initializer=ki,name='conv2_3')
            conv2_2=conv2d(conv2_3,self.scale**2,3,strides=ds, padding='same', activation=None, kernel_initializer=ki,name='conv2_2')

            est0=tf.depth_to_space(conv0_2,self.scale)
            est1=tf.depth_to_space(conv1_2,self.scale)
            est2=tf.depth_to_space(conv2_2,self.scale)

            bilinear=[tf.image.resize_images(frames_y[:,i,:,:,:],[out_height,out_width],method=0) for i in range(self.num_frames)]
            # tem=tf.concat([y[:,i,:,:,:] for i in range(self.num_frames)],-1)
            tem=tf.concat(bilinear,-1)
            tem0=conv2d(tem,32,5,strides=ds, padding='same', activation=None, kernel_initializer=ki,name='tem0')
            #tem0_0=tf.layers.batch_normalization(tem0,training=trainable,reuse=reuse,name='tem0_0')
            tem0_0=activate(tem0)

            tem1=conv2d(tem0_0,16,5,strides=ds, padding='same', activation=None, kernel_initializer=ki,name='tem1')
            #tem1_0=tf.layers.batch_normalization(tem1,training=trainable,reuse=reuse,name='tem1_0')
            tem1_0=activate(tem1)

            tem2=conv2d(tem1_0,3,5,strides=ds, padding='same', activation=None, kernel_initializer=ki,name='tem2')
            #tem2_0=tf.layers.batch_normalization(tem2,training=trainable,reuse=reuse,name='tem2_0')
            temlast=tf.nn.softmax(tem2,axis=-1)

            out0=tf.multiply(est0,temlast[:,:,:,:1])
            out1=tf.multiply(est1,temlast[:,:,:,1:2])
            out2=tf.multiply(est2,temlast[:,:,:,-1:])

            out=tf.add_n([out0,out1,out2])
        self.uv = tf.stack(self.uv, 1)

        return tf.stack([out], axis=1,name='out')#out

    def flow(self, source, reference):
        ds=1
        activate=tf.nn.relu#tf.nn.leaky_relu
        ki=tf.contrib.layers.xavier_initializer()
        n,h,w,c=reference.shape
        with tf.variable_scope('flow',reuse=tf.AUTO_REUSE) as scope:
            x=tf.concat([reference,source],-1)
            x=conv2d(x,32,9,strides=ds, padding='same', activation=activate, kernel_initializer=ki,name='conv0')
            x=tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], padding='SAME', name='pool0')
            x=conv2d(x,32,9,strides=ds, padding='same', activation=activate, kernel_initializer=ki,name='conv1')
            x=tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], padding='SAME', name='pool1')
            x=tf.image.resize_images(x,[h,w],method=0)
            uv=conv2d(x,2,3,strides=ds, padding='same', activation=tf.nn.tanh, kernel_initializer=ki,name='conv2')
        return uv

    def build_model(self):
        frames_lr, frame_gt = self.double_input_producer()
        n, t, h, w, c = frames_lr.get_shape().as_list()
        output = self.forward(frames_lr)

        # calculate loss
        # reconstruction loss
        frame_gt_y = rgb2y(frame_gt)
        mse = tf.reduce_mean(tf.sqrt((output - frame_gt_y) ** 2+1e-6), axis=[0, 1, 2, 3, 4])
        self.mse = mse
        self.loss_mse = tf.reduce_sum(mse)

        # flow loss
        frames_ref_warp = imwarp_backward(self.uv,
                                          tf.tile(tf.expand_dims(self.frame_ref_y, 1), [1, self.num_frames, 1, 1, 1]),
                                          [h, w])
        self.loss_flow_data = tf.reduce_mean(tf.abs(self.frames_y - frames_ref_warp))
        uv4d = tf.reshape(self.uv, [self.batch_size * self.num_frames, h, w, 2])
        self.loss_flow_tv = tf.reduce_sum(tf.image.total_variation(uv4d)) / uv4d.shape.num_elements()
        self.loss_flow = self.loss_flow_data + 0.01 * self.loss_flow_tv

        # total loss
        self.loss = self.loss_mse + self.loss_flow * 0.01

    def evaluation(self):
        print('Evaluating ...')

        filenames=open(self.eval_dir, 'rt').read().splitlines()
        gtList_all=[sorted(glob.glob(join(f,'truth','*.png'))) for f in filenames]
        inList_all=[sorted(glob.glob(join(f,'blur{}'.format(self.scale),'*.png'))) for f in filenames]

        if not hasattr(self, 'sess'):
            sess = tf.Session()
        else:
            sess = self.sess

        border=8
        in_h,in_w=self.eval_in_size
        out_h = in_h*self.scale #512
        out_w = in_w*self.scale #960
        bd=border//self.scale
        if not hasattr(self, 'eval_input'):
            self.eval_input = tf.placeholder(tf.float32, [self.eval_batch_size, self.num_frames, in_h, in_w, 3])
            self.eval_gt = tf.placeholder(tf.float32, [self.eval_batch_size, 1, out_h, out_w, 3])
            self.eval_output = self.forward(self.eval_input, is_training=False, reuse=True)

            # calculate loss
            frame_gt_y = rgb2y(self.eval_gt)
            self.eval_mse = tf.reduce_mean((self.eval_output[:, :, :, :, :] - frame_gt_y) ** 2, axis=[2, 3, 4])

        batch_in = []
        batch_gt = []
        radius = self.num_frames // 2
        mse_acc = None
        ssim_acc = None
        batch_cnt = 0
        #batch_name=[]
        for inList, gtList in zip(inList_all, gtList_all):
            for idx0 in range(15, len(inList), 32):
                #batch_name.append(gtList[idx0])
                inp = [cv2_imread(inList[0]) for i in range(idx0 - radius, 0)]
                inp.extend([cv2_imread(inList[i]) for i in range(max(0, idx0 - radius), idx0)])
                inp.extend([cv2_imread(inList[i]) for i in range(idx0, min(len(inList), idx0 + radius + 1))])
                inp.extend([cv2_imread(inList[-1]) for i in range(idx0 + radius, len(inList) - 1, -1)])
                inp = [i[bd:in_h+bd, bd:in_w+bd, :].astype(np.float32) / 255.0 for i in inp]
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
                          for it in range(1)] for ib in range(self.eval_batch_size)])
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
        for i in range(mse_avg.shape[0]):
            tf.summary.scalar('val_mse{}'.format(i), tf.convert_to_tensor(mse_avg[i], dtype=tf.float32))
        print('Eval MSE: {}, PSNR: {}'.format(mse_avg, psnr_avg))
        # write to log file
        with open(self.log_dir, 'a+') as f:
            mse_avg=(mse_avg*1e8).astype(np.int64)/(1e8)
            psnr_avg=(psnr_avg*1e8).astype(np.int64)/(1e8)
            ssim_avg=(ssim_avg*1e8).astype(np.int64)/(1e8)
            f.write('{'+'"Iter": {} , "MSE": {}, "PSNR": {}, "SSIM": {}'.format(sess.run(self.global_step), mse_avg.tolist(), psnr_avg.tolist(),
                                                                     ssim_avg.tolist())+'}\n')

    def train(self):
        """Train video sr network"""
        global_step = tf.Variable(initial_value=0, trainable=False)
        self.global_step = global_step

        # Create folder for logs
        if not tf.gfile.Exists(self.save_dir):
            tf.gfile.MakeDirs(self.save_dir)

        self.build_model()
        lr = tf.train.polynomial_decay(self.learning_rate, global_step, self.decay_step, end_learning_rate=self.end_lr, power=1.)
        tf.summary.scalar('learning_rate', lr)
        vars_all = tf.trainable_variables()
        vars_sr = [v for v in vars_all if 'ltdvsr' in v.name]
        vars_flow = [v for v in vars_all if 'flow' in v.name]
        train_all = tf.train.AdamOptimizer(lr).minimize(self.loss, var_list=vars_all, global_step=global_step)
        train_flow = tf.train.AdamOptimizer(lr).minimize(self.loss_flow, var_list=vars_flow, global_step=global_step)
        train_sr = tf.train.AdamOptimizer(lr).minimize(self.loss_mse, var_list=vars_sr, global_step=global_step)

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
        if self.reload:
            self.load(sess, self.save_dir)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        cost_time=0
        start_time=time.time()
        gs=sess.run(global_step)
        for step in range(sess.run(global_step), self.max_step):
            if step < 10000:
                train_op = train_sr
            else:
                train_op = train_all

            if step>gs and step % 20 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),'Step:{}, loss:({:.3f},{:.3f},{:.3f}), mse:{}'.format(step,
                                    loss_value,loss_mse_value,loss_flow_value * 100,str(mse_value)))

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
        # model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        # checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
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

    def testvideo(self, dataPath=None, savename='result', reuse=False, scale=4, num_frames=3):
        scale = self.scale
        num_frames= self.num_frames
        inList = sorted(glob.glob(os.path.join(dataPath, 'blur{}/*.png').format(scale)))
        inp = [cv2_imread(i).astype(np.float32) / 255.0 for i in inList]
        max_frame=len(inList)

        print('Testing path: {}'.format(dataPath))
        print('# of testing frames: {}'.format(len(inList)))

        savepath = os.path.join(dataPath,savename)
        automkdir(savepath)

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
            out_h = h * scale
            out_w = w * scale
            padh = int(ceil(h / 4.0) * 4.0 - h)
            padw = int(ceil(w / 4.0) * 4.0 - w)
            imgs = [np.pad(i, [[0, padh], [0, padw], [0, 0]], 'edge') for i in imgs]
            imgs = np.expand_dims(np.stack(imgs, axis=0), 0)

            if idx0 == 0:
                # frames_lr = tf.convert_to_tensor(imgs, tf.float32)
                frames_lr = tf.placeholder(dtype=tf.float32, shape=imgs.shape)
                frames_ref_ycbcr = rgb2ycbcr(frames_lr[:, T:T + 1, :, :, :])
                # frames_ref_ycbcr = tf.tile(frames_ref_ycbcr, [1, num_frames, 1, 1, 1])
                output = self.forward(frames_lr, is_training=False, reuse=reuse)
                # print (frames_lr_ycbcr.get_shape(), h, w, padh, padw)
                if len(dims) == 3:
                    output_rgb = ycbcr2rgb(tf.concat([output, resize_images(frames_ref_ycbcr
                                                                            , [(h + padh) * scale
                                                                                , (w + padw) * scale]
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
                self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
                self.load(self.sess, self.save_dir)
                reuse=True

            st_time=time.time()
            [imgs_hr, imgs_hr_rgb, uv] = self.sess.run([output, output_rgb, self.uv],
                                                            feed_dict={frames_lr: imgs})
            all_time.append(time.time()-st_time)

            if len(dims) == 3:
                cv2_imsave(os.path.join(savepath, 'rgb_%03d.png' % (idx0)),
                                  im2uint8(imgs_hr_rgb[0, -1, :, :, :]))

                # summary_str = self.sess.run(summary_op)
                # summary_writer.add_summary(summary_str, idx0)
        print('SR results path: {}'.format(savepath))
        if max_frame>0:
            all_time=np.array(all_time)
            print('spent {} s in total and {} s in average'.format(np.sum(all_time),np.mean(all_time[1:])))

    def testvideos(self,datapath='/dev/f/data/video/test2/udm10',start=0,savename='ltdvsr'):
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
    model = LTDVSR()
    #model.train()
    #model.testvideos('/dev/f/data/video/test2/vid4', 0, savename='ltdvsr1')