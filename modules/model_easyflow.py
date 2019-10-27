import os
import glob
import time
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import subprocess
from datetime import datetime
from tensorflow.python.ops import control_flow_ops
from modules.videosr_ops import *

#os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax( [int(x.split()[2]) for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))

class EASYFLOW(object):
    def __init__(self):
        self.num_frames = 7
        self.crop_size = 100

        self.max_steps = int(1e6)
        self.batch_size = 20
        self.learning_rate = 1e-4
        self.train_dir = './easyflow_log/model1'
        self.pathlist = open('./data/filelist_train.txt', 'rt').read().splitlines()


    def input_producer(self, batch_size=10):
        def read_data():
            data_seq = tf.random_crop(self.data_queue, [1, self.num_frames])
            input = tf.stack([tf.image.decode_png(tf.read_file(data_seq[0][i]), channels=3) for i in range(self.num_frames)])
            input = preprocessing(input)
            print('Input producer shape: ', input.get_shape())
            return input

        def preprocessing(input):
            input = tf.cast(input, tf.float32) / 255.0

            shape = tf.shape(input)[1:]
            size = tf.convert_to_tensor([self.crop_size, self.crop_size, 3], dtype=tf.int32, name="size")
            check = tf.Assert(tf.reduce_all(shape >= size), ["Need value.shape >= size, got ", shape, size])
            shape = control_flow_ops.with_dependencies([check], shape)

            limit = shape - size + 1
            offset = tf.random_uniform(tf.shape(shape), dtype=size.dtype, maxval=size.dtype.max, seed=None) % limit

            offset_in = tf.concat([[0], offset], axis=-1)
            size_in = tf.concat([[self.num_frames], size], axis=-1)
            input = tf.slice(input, offset_in, size_in)

            input.set_shape([self.num_frames, self.crop_size, self.crop_size, 3])
            return input

        with tf.variable_scope('input'):
            inList_all = []
            for dataPath in self.pathlist:
                inList = sorted(glob.glob(os.path.join(dataPath, 'input/*.png')))
                inList_all.append(inList)
            inList_all = tf.convert_to_tensor(inList_all, dtype=tf.string)

            self.data_queue = tf.train.slice_input_producer([inList_all], capacity=40)
            input = read_data()
            batch_in = tf.train.batch([input], batch_size=batch_size, num_threads=3, capacity=40)
        return batch_in
#
    def forward(self, imga, imgb, scope='easyflow', reuse=False):
        dims = len(imga.get_shape())
        if dims == 5:
            n, num_frame, height, width, num_channels = imga.get_shape().as_list()
            imga = tf.reshape(imga, [n * num_frame, height, width, num_channels])
            imgb = tf.reshape(imgb, [n * num_frame, height, width, num_channels])

        n, h, w, c = imga.get_shape().as_list()
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                biases_initializer=tf.constant_initializer(0.0)), \
                 slim.arg_scope([slim.conv2d_transpose], activation_fn=tf.nn.relu,
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                biases_initializer=tf.constant_initializer(0.0)):

                inputs = tf.concat([imga, imgb], 3, name='flow_inp')
                c1 = slim.conv2d(inputs, 24, [5, 5], stride=2, scope='c1')
                c2 = slim.conv2d(c1, 24, [3, 3], scope='c2')
                c3 = slim.conv2d(c2, 24, [5, 5], stride=2, scope='c3')
                c4 = slim.conv2d(c3, 24, [3, 3], scope='c4')
                c5 = slim.conv2d(c4, 32, [3, 3], activation_fn=tf.nn.tanh, scope='c5')

                c5_hr = tf.reshape(c5, [n, h//4, w//4, 2, 4, 4])
                c5_hr = tf.transpose(c5_hr, [0, 1, 4, 2, 5, 3])
                c5_hr = tf.reshape(c5_hr, [n, h, w, 2])

                img_warp = imwarp_backward(c5_hr, imgb, [h, w])
                c5_pack = tf.concat([inputs, c5_hr, img_warp], 3, name='cat')

                s1 = slim.conv2d(c5_pack, 24, [5, 5], stride=2, scope='s1')
                s2 = slim.conv2d(s1, 24, [3, 3], scope='s2')
                s3 = slim.conv2d(s2, 24, [3, 3], scope='s3')
                s4 = slim.conv2d(s3, 24, [3, 3], scope='s4')
                s5 = slim.conv2d(s4, 8, [3, 3], activation_fn=tf.nn.tanh, scope='s5')

                s5_hr = tf.reshape(s5, [n, h // 2, w //2, 2, 2, 2])
                s5_hr = tf.transpose(s5_hr, [0, 1, 4, 2, 5, 3])
                s5_hr = tf.reshape(s5_hr, [n, h, w, 2])
                uv = c5_hr + s5_hr
        if dims == 5:
            uv = tf.reshape(uv, [self.batch_size, num_frame, height, width, 2])
        return uv

    def build_model(self):
        frames_lr = self.input_producer(batch_size=self.batch_size)
        n, t, h, w, c = frames_lr.get_shape().as_list()

        idx0 = self.num_frames // 2
        frames_y = rgb2y(frames_lr)
        frames_ref_y = frames_y[:, idx0:idx0 + 1, :, :, :]
        frames_ref_y = tf.tile(frames_ref_y, [1, self.num_frames, 1, 1, 1])

        uv = self.forward(frames_y, frames_ref_y)
        frames_ref_warp = imwarp_backward(uv, frames_ref_y, [h, w])
        tf.summary.image('inp', im2uint8(frames_y[0, :, :, :, :]), max_outputs=3)
        tf.summary.image('uv', flowToColor(uv[0, :, :, :, :], maxflow=3.0), max_outputs=3)
        tf.summary.image('warp', im2uint8(frames_ref_warp[0, :, :, :, :]), max_outputs=3)

        loss_data = tf.reduce_mean(tf.abs(frames_y - frames_ref_warp))
        loss_tv = tf.reduce_sum(tf.image.total_variation(uv)) / uv.shape.num_elements()
        self.loss = loss_data + 0.01 * loss_tv
        #self.loss = loss_data
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('loss_data', loss_data)
        tf.summary.scalar('loss_tv', loss_tv)

    def train(self):
        def train_op_func(loss, var_list, is_gradient_clip=False):
            if is_gradient_clip:
                train_op = tf.train.AdamOptimizer(lr)
                grads_and_vars = train_op.compute_gradients(loss, var_list=var_list)
                unchanged_gvs = [(grad, var) for grad, var in grads_and_vars if not 'LSTM' in var.name]
                rnn_grad = [grad for grad, var in grads_and_vars if 'LSTM' in var.name]
                rnn_var = [var for grad, var in grads_and_vars if 'LSTM' in var.name]
                capped_grad, _ = tf.clip_by_global_norm(rnn_grad, clip_norm=3)
                capped_gvs = list(zip(capped_grad, rnn_var))
                train_op = train_op.apply_gradients(grads_and_vars=capped_gvs + unchanged_gvs, global_step=global_step)
            else:
                train_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=var_list, global_step=global_step)
            return train_op

        """Train easyflow network"""
        global_step = tf.Variable(initial_value=0, trainable=False)

        # Create folder for logs
        if not tf.gfile.Exists(self.train_dir):
            tf.gfile.MakeDirs(self.train_dir)

        self.build_model()
        decay_steps = 3e5
        lr = tf.train.polynomial_decay(self.learning_rate, global_step, decay_steps, end_learning_rate=1e-6, power=0.9)
        tf.summary.scalar('learning_rate', lr)
        vars_all = tf.trainable_variables()
        vars_sr = [v for v in vars_all if 'srmodel' in v.name]
        vars_srcnn = [v for v in vars_all if 'srcnn' in v.name]
        vars_flownet = [v for v in vars_all if 'flownet' in v.name]
        train_all = train_op_func(self.loss, vars_all, is_gradient_clip=True)
        # train_sr = train_op_func(self.loss, vars_sr, is_gradient_clip=True)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        # self.load(sess, os.path.join(self.train_dir, 'checkpoints'))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph, flush_secs=30)

        for step in range(sess.run(global_step), self.max_steps):

            start_time = time.time()
            _, loss_value = sess.run([train_all, self.loss])
            duration = time.time() - start_time
            # print loss_value
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 5 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.4f (%.1f data/s; %.3f s/batch)')
                print((format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, loss_value * 100,
                                     examples_per_sec, sec_per_batch)))
            if step % 10 == 0:
                # summary_str = sess.run(summary_op, feed_dict={inputs:batch_input, gt:batch_gt})
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

            # Save the model checkpoint periodically.
            if step % 500 == 499 or (step + 1) == self.max_steps:
                checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
                self.save(sess, checkpoint_path, step)

    def save(self, sess, checkpoint_dir, step):
        if not hasattr(self,'saver'):
            self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        model_name = "easyflow.model"
        # model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        # checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir='./easyflow_log/model1/checkpoints', step=None):
        print(" [*] Reading checkpoints...")
        model_name = "easyflow.model"

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading checkpoints... Success{}".format(ckpt_name))
            return True
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False

    def load_easyflow(self, sess, checkpoint_dir='./easyflow_log/model1/checkpoints'):
        print(" [*] Reading EasyFlow checkpoints...")
        model_name = "easyflow.model"

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            flownets_var = [var for var in tf.trainable_variables() if 'easyflow' in var.name]
            saver = tf.train.Saver(var_list=flownets_var)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading checkpoints...{} Success".format(ckpt_name))
            return True
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False


def main(_):
    model = EASYFLOW()
    model.train()

if __name__ == '__main__':
    tf.app.run()
