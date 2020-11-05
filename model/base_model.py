import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from os.path import join,exists
import glob
import random
import numpy as np
from PIL import Image
import scipy
from utils import LoadImage, DownSample, DownSample_4D, BLUR, AVG_PSNR, depth_to_space_3D, DynFilter3D, LoadParams, cv2_imread, cv2_imsave, get_num_params, automkdir
from model.nets import FR_16L, FR_28L, FR_52L
from modules.videosr_ops import imwarp_forward
import time
import os
from tqdm import trange,tqdm

class VSR(object):
    def __init__(self):
        """
        Initialize training data.

        Args:
            self: (todo): write your description
        """
        self.num_frames=7
        self.scale=4
        self.in_size=32
        self.gt_size=self.in_size*self.scale
        self.eval_in_size=[128,240]
        self.batch_size=16
        self.eval_basz=4
        self.learning_rate=1e-3
        self.end_lr=1e-4
        self.reload=True
        self.max_step=int(1.5e5+1)
        self.decay_step=1.2e5
        self.train_dir='./data/filelist_train.txt'
        self.eval_dir='./data/filelist_val.txt'
        self.save_dir='./checkpoint'
        self.log_dir='./eval_log.txt'
    

    def frvsr_input_producer(self):
        """
        Reads : py : func.

        Args:
            self: (todo): write your description
        """
        def read_data():
            """
            Reads a png from the image.

            Args:
            """
            idx0 = self.num_frames // 2
            data_seq = tf.random_crop(self.data_queue, [2, self.num_frames])
            input = tf.stack([tf.image.decode_png(tf.read_file(data_seq[0][i]), channels=3) for i in range(self.num_frames)])
            #gt = tf.stack([tf.image.decode_png(tf.read_file(data_seq[1][idx0]), channels=3)])
            gt = tf.stack([tf.image.decode_png(tf.read_file(data_seq[1][i]), channels=3) for i in range(self.num_frames)])
            input, gt = prepprocessing(input, gt)
            print('Input producer shape: ', input.get_shape(), gt.get_shape())
            return input, gt

        def prepprocessing(input, gt=None):
            """
            Prepprocessing.

            Args:
                input: (todo): write your description
                gt: (todo): write your description
            """
            input = tf.cast(input, tf.float32) / 255.0
            gt = tf.cast(gt, tf.float32) / 255.0

            shape = tf.shape(input)[1:]
            size = tf.convert_to_tensor([self.in_size, self.in_size, 3], dtype=tf.int32, name="size")
            check = tf.Assert(tf.reduce_all(shape >= size), ["Need value.shape >= size, got ", shape, size])
            shape = control_flow_ops.with_dependencies([check], shape)

            limit = shape - size + 1
            offset = tf.random_uniform(tf.shape(shape), dtype=size.dtype, maxval=size.dtype.max, seed=None) % limit

            offset_in = tf.concat([[0], offset], axis=-1)
            size_in = tf.concat([[self.num_frames], size], axis=-1)
            input = tf.slice(input, offset_in, size_in)
            offset_gt = tf.concat([[0], offset[:2] * self.scale, [0]], axis=-1)
            size_gt = tf.concat([[self.num_frames], size[:2] * self.scale, [3]], axis=-1)
            gt = tf.slice(gt, offset_gt, size_gt)

            input.set_shape([self.num_frames, self.in_size, self.in_size, 3])
            gt.set_shape([self.num_frames, self.in_size * self.scale, self.in_size * self.scale, 3])
            return input, gt
            

        pathlist=open(self.train_dir, 'rt').read().splitlines()
        random.shuffle(pathlist)
        with tf.variable_scope('input'):
            inList_all = []
            gtList_all = []
            for dataPath in pathlist:
                inList = sorted(glob.glob(os.path.join(dataPath, 'blur{}/*.png'.format(self.scale))))
                gtList = sorted(glob.glob(os.path.join(dataPath, 'truth/*.png')))
                inList_all.append(inList)
                gtList_all.append(gtList)
            inList_all = tf.convert_to_tensor(inList_all, dtype=tf.string)
            gtList_all = tf.convert_to_tensor(gtList_all, dtype=tf.string)

            self.data_queue = tf.train.slice_input_producer([inList_all, gtList_all], capacity=self.batch_size*2)
            input, gt = read_data()
            batch_in, batch_gt = tf.train.batch([input, gt], batch_size=self.batch_size, num_threads=3, capacity=self.batch_size*2)
        return batch_in, batch_gt

    def double_input_producer(self):
        """
        Reads images from the image.

        Args:
            self: (todo): write your description
        """
        def read_data():
            """
            Reads the image from the image.

            Args:
            """
            idx0 = self.num_frames // 2
            data_seq = tf.random_crop(self.data_queue, [2, self.num_frames])
            input = tf.stack([tf.image.decode_png(tf.read_file(data_seq[0][i]), channels=3) for i in range(self.num_frames)])
            gt=tf.stack([tf.image.decode_png(tf.read_file(data_seq[1][idx0]), channels=3)])
            input, gt = prepprocessing(input, gt)

            flip=tf.random_uniform((1,3),minval=0.0,maxval=1.0,dtype=tf.float32,seed=None,name=None)    #if training gets worse, comment the data flip part out
            input=tf.where(flip[0][0]<0.5,input,input[:,::-1])
            input=tf.where(flip[0][1]<0.5,input,input[:,:,::-1])
            input=tf.where(flip[0][2]<0.5,input,tf.transpose(input,perm=(0,2,1,3)))
            gt=tf.where(flip[0][0]<0.5,gt,gt[:,::-1])
            gt=tf.where(flip[0][1]<0.5,gt,gt[:,:,::-1])
            gt=tf.where(flip[0][2]<0.5,gt,tf.transpose(gt,perm=(0,2,1,3)))
            print('Input producer shape: ', input.get_shape(), gt.get_shape())
            return input, gt
        

        def prepprocessing(input, gt=None):
            """
            Prepprocessing.

            Args:
                input: (todo): write your description
                gt: (todo): write your description
            """
            input = tf.cast(input, tf.float32) / 255.0
            gt = tf.cast(gt, tf.float32) / 255.0

            shape = tf.shape(input)[1:]
            size = tf.convert_to_tensor([self.in_size, self.in_size, 3], dtype=tf.int32, name="size")
            check = tf.Assert(tf.reduce_all(shape >= size), ["Need value.shape >= size, got ", shape, size])
            shape = control_flow_ops.with_dependencies([check], shape)

            limit = shape - size + 1
            offset = tf.random_uniform(tf.shape(shape), dtype=size.dtype, maxval=size.dtype.max, seed=None) % limit

            offset_in = tf.concat([[0], offset], axis=-1)
            size_in = tf.concat([[self.num_frames], size], axis=-1)
            input = tf.slice(input, offset_in, size_in)
            offset_gt = tf.concat([[0], offset[:2] * self.scale, [0]], axis=-1)
            size_gt = tf.concat([[1], size[:2] * self.scale, [3]], axis=-1)
            gt = tf.slice(gt, offset_gt, size_gt)

            input.set_shape([self.num_frames, self.in_size, self.in_size, 3])
            gt.set_shape([1, self.in_size * self.scale, self.in_size * self.scale, 3])
            return input, gt
            

        pathlist=open(self.train_dir, 'rt').read().splitlines()
        random.shuffle(pathlist)
        with tf.variable_scope('input'):
            inList_all = []
            gtList_all = []
            for dataPath in pathlist:
                inList = sorted(glob.glob(os.path.join(dataPath, 'blur{}/*.png'.format(self.scale))))
                gtList = sorted(glob.glob(os.path.join(dataPath, 'truth/*.png')))
                inList_all.append(inList)
                gtList_all.append(gtList)
            inList_all = tf.convert_to_tensor(inList_all, dtype=tf.string)
            gtList_all = tf.convert_to_tensor(gtList_all, dtype=tf.string)

            self.data_queue = tf.train.slice_input_producer([inList_all, gtList_all], capacity=self.batch_size*2)
            input, gt = read_data()
            batch_in, batch_gt = tf.train.batch([input, gt], batch_size=self.batch_size, num_threads=3, capacity=self.batch_size*2)
        return batch_in, batch_gt

    def single_input_producer(self):
        """
        Read a single tf.

        Args:
            self: (todo): write your description
        """
        def read_data():
            """
            Reads a png from disk.

            Args:
            """
            data_seq = tf.random_crop(self.data_queue, [1, self.num_frames])
            #input = tf.stack([tf.image.decode_png(tf.read_file(data_seq[0][i]), channels=3) for i in range(self.num_frames)])
            gt = tf.stack([tf.image.decode_png(tf.read_file(data_seq[0][i]), channels=3) for i in range(self.num_frames)])
            #gt = tf.stack([tf.image.decode_png(tf.read_file(data_seq[1][i]), channels=3) for i in range(self.num_frames)])
            
            input, gt = prepprocessing(gt)

            return input, gt

        def prepprocessing(gt=None):
            """
            Prepprocessing.

            Args:
                gt: (todo): write your description
            """
            n,w,h,c=gt.shape
            sp=tf.shape(gt)[1:]
            size=tf.convert_to_tensor([self.gt_size,self.gt_size,c], dtype=tf.int32)

            limit=sp-size+1
            offset= tf.random_uniform(sp.shape, dtype=size.dtype, maxval=size.dtype.max, seed=None) % limit
            offset_gt=tf.concat([[0],offset[:2],[0]], axis=-1)
            size_gt=tf.concat([[n],size], axis=-1)

            gt=tf.slice(gt, offset_gt, size_gt)
            gt=tf.cast(gt, tf.float32)/255.

            flip=tf.random_uniform((1,3),minval=0.0,maxval=1.0,dtype=tf.float32,seed=None,name=None)
            gt=tf.where(flip[0][0]<0.5,gt,gt[:,::-1])
            gt=tf.where(flip[0][1]<0.5,gt,gt[:,:,::-1])
            gt=tf.where(flip[0][2]<0.5,gt,tf.transpose(gt,perm=(0,2,1,3)))
            inp=DownSample_4D(gt, BLUR, scale=self.scale)
            gt=gt[n//2:n//2+1,:,:,:]

            inp.set_shape([self.num_frames, self.in_size, self.in_size, 3])
            gt.set_shape([1, self.in_size * self.scale, self.in_size * self.scale, 3])
            print('Input producer shape: ', inp.get_shape(), gt.get_shape())
            
            return inp, gt

        pathlist=open(self.train_dir, 'rt').read().splitlines()
        random.shuffle(pathlist)
        with tf.variable_scope('trainin'):
            gtList_all = []
            for dataPath in pathlist:
                gtList = sorted(glob.glob(os.path.join(dataPath, 'truth/*.png')))
                gtList_all.append(gtList)
            gtList_all = tf.convert_to_tensor(gtList_all, dtype=tf.string)

            self.data_queue = tf.train.slice_input_producer([gtList_all], capacity=self.batch_size*2)
            input, gt = read_data()
            batch_in, batch_gt = tf.train.batch([input, gt], batch_size=self.batch_size, num_threads=3, capacity=self.batch_size*2)
        return batch_in, batch_gt
            
        
    def forward(self, x):
        """
        Implement forward on x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        pass

                    
    def build(self):
        """
        Build a build.

        Args:
            self: (todo): write your description
        """
        pass
        
    def eval(self):
        """
        Evaluate the evaluation.

        Args:
            self: (todo): write your description
        """
        pass
    
    def train(self):
        """
        Train a taver.

        Args:
            self: (todo): write your description
        """
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config) 
        #sess=tf.Session()
        self.sess=sess
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)

        return
                
    def save(self, sess, checkpoint_dir, step):
        """
        Save the model to disk.

        Args:
            self: (todo): write your description
            sess: (todo): write your description
            checkpoint_dir: (str): write your description
            step: (int): write your description
        """
        model_name = "VSR"
        # model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        # checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None):
        """
        Load the model.

        Args:
            self: (todo): write your description
            sess: (todo): write your description
            checkpoint_dir: (str): write your description
            step: (todo): write your description
        """
        print(" [*] Reading SR checkpoints...")
        model_name = "VSR"

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading checkpoints...{} Success".format(ckpt_name))
            return True
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False
            
    def test_video(self, path, name='result', reuse=False):
        """
        Test if a video.

        Args:
            self: (todo): write your description
            path: (str): write your description
            name: (str): write your description
            reuse: (todo): write your description
        """
        pass

    def testvideos(self):
        """
        Test if a list of the test.

        Args:
            self: (todo): write your description
        """
        pass
    
        
if __name__=='__main__':
    model=VSR()
    model.train()
    #model.testvideos()