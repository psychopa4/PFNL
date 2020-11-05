import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from os.path import join,exists
import glob
import random
import numpy as np
from PIL import Image
import scipy
import time
import os
from tensorflow.python.layers.convolutional import Conv2D,conv2d
from utils import NonLocalBlock, DownSample, DownSample_4D, BLUR, get_num_params, cv2_imread, cv2_imsave, automkdir
from tqdm import tqdm,trange
from model.base_model import VSR

'''This is the official code of PFNL (Progressive Fusion Video Super-Resolution Network via Exploiting Non-Local Spatio-Temporal Correlations).
The code is mainly based on https://github.com/psychopa4/MMCNN and https://github.com/jiangsutx/SPMC_VideoSR.
'''

class PFNL(VSR):
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
        self.save_dir='./checkpoint/pfnl'
        self.log_dir='./pfnl.txt'
    
    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        mf=64
        dk=3
        activate=tf.nn.leaky_relu
        num_block=20
        n,f1,w,h,c=x.shape
        ki=tf.contrib.layers.xavier_initializer()
        ds=1
        with tf.variable_scope('nlvsr',reuse=tf.AUTO_REUSE) as scope:
            conv0=Conv2D(mf, 5, strides=ds, padding='same', activation=activate, kernel_initializer=ki, name='conv0')
            conv1=[Conv2D(mf, dk, strides=ds, padding='same', activation=activate, kernel_initializer=ki, name='conv1_{}'.format(i)) for i in range(num_block)]
            conv10=[Conv2D(mf, 1, strides=ds, padding='same', activation=activate, kernel_initializer=ki, name='conv10_{}'.format(i)) for i in range(num_block)]
            conv2=[Conv2D(mf, dk, strides=ds, padding='same', activation=activate, kernel_initializer=ki, name='conv2_{}'.format(i)) for i in range(num_block)]
            convmerge1=Conv2D(48, 3, strides=ds, padding='same', activation=activate, kernel_initializer=ki, name='convmerge1')
            convmerge2=Conv2D(12, 3, strides=ds, padding='same', activation=None, kernel_initializer=ki, name='convmerge2')
            
            inp0=[x[:,i,:,:,:] for i in range(f1)]
            inp0=tf.concat(inp0,axis=-1)
            inp1=tf.space_to_depth(inp0,2)
            inp1=NonLocalBlock(inp1,int(c)*self.num_frames*4,sub_sample=1,nltype=1,scope='nlblock_{}'.format(0))
            inp1=tf.depth_to_space(inp1,2)
            inp0+=inp1
            inp0=tf.split(inp0, num_or_size_splits=self.num_frames, axis=-1)
            inp0=[conv0(f) for f in inp0]
            bic=tf.image.resize_images(x[:,self.num_frames//2,:,:,:],[w*self.scale,h*self.scale],method=2)

            for i in range(num_block):
                inp1=[conv1[i](f) for f in inp0]
                base=tf.concat(inp1,axis=-1)
                base=conv10[i](base)
                inp2=[tf.concat([base,f],-1) for f in inp1]
                inp2=[conv2[i](f) for f in inp2]
                inp0=[tf.add(inp0[j],inp2[j]) for j in range(f1)]

            merge=tf.concat(inp0,axis=-1)
            merge=convmerge1(merge)

            large1=tf.depth_to_space(merge,2)
            out1=convmerge2(large1)
            out=tf.depth_to_space(out1,2)
                
        return tf.stack([out+bic], axis=1,name='out')#out

    def build(self):
        """
        Builds the graph.

        Args:
            self: (todo): write your description
        """
        in_h,in_w=self.eval_in_size
        H = tf.placeholder(tf.float32, shape=[None, 1, None, None, 3], name='H_truth')
        L_train = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_frames, self.in_size, self.in_size, 3], name='L_train')
        L_eval = tf.placeholder(tf.float32, shape=[self.eval_basz, self.num_frames, in_h, in_w, 3], name='L_eval')
        SR_train = self.forward(L_train)
        SR_eval = self.forward(L_eval)
        loss=tf.reduce_mean(tf.sqrt((SR_train-H)**2+1e-6))
        eval_mse=tf.reduce_mean((SR_eval-H) ** 2, axis=[2,3,4])
        self.loss, self.eval_mse= loss, eval_mse
        self.L, self.L_eval, self.H, self.SR =  L_train, L_eval, H, SR_train
        
    def eval(self):
        """
        Evaluate the model.

        Args:
            self: (todo): write your description
        """
        print('Evaluating ...')
        if not hasattr(self, 'sess'):
            sess = tf.Session()
            self.load(sess, self.save_dir)
        else:
            sess = self.sess
            
        border=8
        in_h,in_w=self.eval_in_size
        out_h = in_h*self.scale #512
        out_w = in_w*self.scale #960
        bd=border//self.scale
        
        eval_gt = tf.placeholder(tf.float32, [None, self.num_frames, out_h, out_w, 3])
        eval_inp=DownSample(eval_gt, BLUR, scale=self.scale)
        
        filenames=open(self.eval_dir, 'rt').read().splitlines()#sorted(glob.glob(join(self.eval_dir,'*')))
        gt_list=[sorted(glob.glob(join(f,'truth','*.png'))) for f in filenames]
        
        center=15
        batch_gt = []
        batch_cnt=0
        mse_acc=None
        for gtlist in gt_list:
            max_frame=len(gtlist)
            for idx0 in range(center, max_frame, 32):
                index=np.array([i for i in range(idx0-self.num_frames//2,idx0+self.num_frames//2+1)])
                index=np.clip(index,0,max_frame-1).tolist()
                gt=[cv2_imread(gtlist[i]) for i in index]
                gt = [i[border:out_h+border, border:out_w+border, :].astype(np.float32) / 255.0 for i in gt]
                batch_gt.append(np.stack(gt, axis=0))
                
                if len(batch_gt) == self.eval_basz:
                    batch_gt = np.stack(batch_gt, 0)
                    batch_lr=sess.run(eval_inp,feed_dict={eval_gt:batch_gt})
                    mse_val=sess.run(self.eval_mse,feed_dict={self.L_eval:batch_lr, self.H:batch_gt[:,self.num_frames//2:self.num_frames//2+1]})
                    if mse_acc is None:
                        mse_acc = mse_val
                    else:
                        mse_acc = np.concatenate([mse_acc, mse_val], axis=0)
                    batch_gt = []
                    print('\tEval batch {} - {} ...'.format(batch_cnt, batch_cnt + self.eval_basz))
                    batch_cnt+=self.eval_basz
                    
        psnr_acc = 10 * np.log10(1.0 / mse_acc)
        mse_avg = np.mean(mse_acc, axis=0)
        psnr_avg = np.mean(psnr_acc, axis=0)
        for i in range(mse_avg.shape[0]):
            tf.summary.scalar('val_mse{}'.format(i), tf.convert_to_tensor(mse_avg[i], dtype=tf.float32))
        print('Eval PSNR: {}, MSE: {}'.format(psnr_avg, mse_avg))
        # write to log file
        with open(self.log_dir, 'a+') as f:
            mse_avg=(mse_avg*1e6).astype(np.int64)/(1e6)
            psnr_avg=(psnr_avg*1e6).astype(np.int64)/(1e6)
            f.write('{'+'"Iter": {} , "PSNR": {}, "MSE": {}'.format(sess.run(self.global_step), psnr_avg.tolist(), mse_avg.tolist())+'}\n')
    
    def train(self):
        """
        Train the model.

        Args:
            self: (todo): write your description
        """
        LR, HR= self.single_input_producer()
        global_step=tf.Variable(initial_value=0, trainable=False)
        self.global_step=global_step
        self.build()
        lr= tf.train.polynomial_decay(self.learning_rate, global_step, self.decay_step, end_learning_rate=self.end_lr, power=1.)
        
        vars_all=tf.trainable_variables()
        print('Params num of all:',get_num_params(vars_all))
        training_op = tf.train.AdamOptimizer(lr).minimize(self.loss, var_list=vars_all, global_step=global_step)
        
        
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config) 
        #sess=tf.Session()
        self.sess=sess
        sess.run(tf.global_variables_initializer())
        
        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        if self.reload:
            self.load(sess, self.save_dir)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        cost_time=0
        start_time=time.time()
        gs=sess.run(global_step)
        for step in range(sess.run(global_step), self.max_step):
            if step>gs and step%20==0:
                print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),'Step:{}, loss:{}'.format(step,loss_v))
                
            if step % 500 == 0:
                if step>gs:
                    self.save(sess, self.save_dir, step)
                cost_time=time.time()-start_time
                print('cost {}s.'.format(cost_time))
                self.eval()
                cost_time=time.time()-start_time
                start_time=time.time()
                print('cost {}s.'.format(cost_time))

            lr1,hr=sess.run([LR,HR])
            _,loss_v=sess.run([training_op,self.loss],feed_dict={self.L:lr1, self.H:hr})

            if step>500 and loss_v>10:
                print('Model collapsed with loss={}'.format(loss_v))
                break
                
            

    def test_video_truth(self, path, name='result', reuse=False, part=50):
        """
        Test for video.

        Args:
            self: (todo): write your description
            path: (str): write your description
            name: (str): write your description
            reuse: (todo): write your description
            part: (todo): write your description
        """
        save_path=join(path,name)
        automkdir(save_path)
        inp_path=join(path,'truth')
        imgs=sorted(glob.glob(join(inp_path,'*.png')))
        max_frame=len(imgs)
        imgs=np.array([cv2_imread(i) for i in imgs])/255.

        if part>max_frame:
            part=max_frame
        if max_frame%part ==0 :
            num_once=max_frame//part
        else:
            num_once=max_frame//part+1
        
        h,w,c=imgs[0].shape

        L_test = tf.placeholder(tf.float32, shape=[num_once, self.num_frames, h//self.scale, w//self.scale, 3], name='L_test')
        SR_test=self.forward(L_test)
        if not reuse:
            self.img_hr=tf.placeholder(tf.float32, shape=[None, None, None, 3], name='H_truth')
            self.img_lr=DownSample_4D(self.img_hr, BLUR, scale=self.scale)
            config = tf.ConfigProto() 
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config) 
            #sess=tf.Session()
            self.sess=sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)
            self.load(sess, self.save_dir)
        
        lrs=self.sess.run(self.img_lr,feed_dict={self.img_hr:imgs})

        lr_list=[]
        max_frame=lrs.shape[0]
        for i in range(max_frame):
            index=np.array([i for i in range(i-self.num_frames//2,i+self.num_frames//2+1)])
            index=np.clip(index,0,max_frame-1).tolist()
            lr_list.append(np.array([lrs[j] for j in index]))
        lr_list=np.array(lr_list)
        
        print('Save at {}'.format(save_path))
        print('{} Inputs With Shape {}'.format(lrs.shape[0],lrs.shape[1:]))
        h,w,c=lrs.shape[1:]
        
        
        all_time=[]
        for i in trange(part):
            st_time=time.time()
            sr=self.sess.run(SR_test,feed_dict={L_test : lr_list[i*num_once:(i+1)*num_once]})
            all_time.append(time.time()-st_time)
            for j in range(sr.shape[0]):
                img=sr[j][0]*255.
                img=np.clip(img,0,255)
                img=np.round(img,0).astype(np.uint8)
                cv2_imsave(join(save_path, '{:0>4}.png'.format(i*num_once+j)),img)
        all_time=np.array(all_time)
        if max_frame>0:
            all_time=np.array(all_time)
            print('spent {} s in total and {} s in average'.format(np.sum(all_time),np.mean(all_time[1:])))

    def test_video_lr(self, path, name='result', reuse=False, part=50):
        """
        Test for video learning.

        Args:
            self: (todo): write your description
            path: (str): write your description
            name: (str): write your description
            reuse: (todo): write your description
            part: (todo): write your description
        """
        save_path=join(path,name)
        automkdir(save_path)
        inp_path=join(path,'blur{}'.format(self.scale))
        imgs=sorted(glob.glob(join(inp_path,'*.png')))
        max_frame=len(imgs)
        lrs=np.array([cv2_imread(i) for i in imgs])/255.

        if part>max_frame:
            part=max_frame
        if max_frame%part ==0 :
            num_once=max_frame//part
        else:
            num_once=max_frame//part+1
        
        h,w,c=lrs[0].shape

        L_test = tf.placeholder(tf.float32, shape=[num_once, self.num_frames, h, w, 3], name='L_test')
        SR_test=self.forward(L_test)
        if not reuse:
            config = tf.ConfigProto() 
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config) 
            #sess=tf.Session()
            self.sess=sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)
            self.load(sess, self.save_dir)
        

        lr_list=[]
        max_frame=lrs.shape[0]
        for i in range(max_frame):
            index=np.array([i for i in range(i-self.num_frames//2,i+self.num_frames//2+1)])
            index=np.clip(index,0,max_frame-1).tolist()
            lr_list.append(np.array([lrs[j] for j in index]))
        lr_list=np.array(lr_list)
        
        print('Save at {}'.format(save_path))
        print('{} Inputs With Shape {}'.format(lrs.shape[0],lrs.shape[1:]))
        h,w,c=lrs.shape[1:]
        
        all_time=[]
        for i in trange(part):
            st_time=time.time()
            sr=self.sess.run(SR_test,feed_dict={L_test : lr_list[i*num_once:(i+1)*num_once]})
            all_time.append(time.time()-st_time)
            for j in range(sr.shape[0]):
                img=sr[j][0]*255.
                img=np.clip(img,0,255)
                img=np.round(img,0).astype(np.uint8)
                cv2_imsave(join(save_path, '{:0>4}.png'.format(i*num_once+j)),img)

        all_time=np.array(all_time)
        if max_frame>0:
            all_time=np.array(all_time)
            print('spent {} s in total and {} s in average'.format(np.sum(all_time),np.mean(all_time[1:])))

    def testvideos(self, path='/dev/f/data/video/test2/udm10', start=0, name='pfnl'):
        """
        Test if the video isochastic files.

        Args:
            self: (todo): write your description
            path: (str): write your description
            start: (todo): write your description
            name: (str): write your description
        """
        kind=sorted(glob.glob(join(path,'*')))
        kind=[k for k in kind if os.path.isdir(k)]
        reuse=False
        for k in kind:
            idx=kind.index(k)
            if idx>=start:
                if idx>start:
                    reuse=True
                datapath=join(path,k)
                self.test_video_truth(datapath, name=name, reuse=reuse, part=1000)
            
    
if __name__=='__main__':
    model=PFNL()
    model.train()
    #model.testvideos()
