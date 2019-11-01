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
from model.base_model import VSR

'''This work tries to rebuild FRVSR (Frame-Recurrent Video Super-Resolution).
The code is mainly based on https://github.com/psychopa4/MMCNN and https://github.com/jiangsutx/SPMC_VideoSR.
'''
        
class FRVSR(VSR):
    def __init__(self):
        self.num_frames=10
        self.scale=4
        self.in_size=32
        self.gt_size=self.in_size*self.scale
        self.eval_in_size=[128,240]
        self.batch_size=16
        self.eval_basz=4
        self.learning_rate=1e-4
        self.end_lr=1e-4
        self.reload=True
        self.max_step=int(4e5+1)
        self.decay_step=1.2e5
        self.train_dir='./data/filelist_train.txt'
        self.eval_dir='./data/filelist_val.txt'
        self.save_dir='./checkpoint/frvsr'
        self.log_dir='./frvsr.txt'
            
        
    def forward(self, x, xp=None, est=None):
        mf=128
        dk=3
        activate=tf.nn.relu
        num_block=10
        n,w,h,c=x.shape
        if xp is not None:
            uv=self.flow(x,xp)
            est=self.upscale_warp(uv,est)
        with tf.variable_scope('frvsr',reuse=tf.AUTO_REUSE) as scope:
            inp=x
            if xp is None:
                conv0=tf.layers.conv2d(inp, mf, dk, padding='same', activation=activate,name='conv0_0')
            else:
                inp=tf.concat([inp,est],axis=-1)
                conv0=tf.layers.conv2d(inp, mf, dk, padding='same', activation=activate,name='conv0_1')
            for j in range(num_block):
                conv1=tf.layers.conv2d(conv0, mf, dk, padding='same', activation=activate,name='conv1_{}'.format(j))
                conv2=tf.layers.conv2d(conv1, mf, dk, padding='same', activation=None, name='conv2_{}'.format(j))
                conv0+=conv2
                
            large1=tf.layers.conv2d_transpose(conv0, mf, dk, strides=2, padding='same', activation=activate, name='large1')
            large2=tf.layers.conv2d_transpose(large1, mf, dk, strides=2, padding='same', activation=activate, name='large2')
            out=tf.layers.conv2d(large2, 3, dk, padding='same', activation=None, name='out')
                
        return out
    
    def flow(self, i_t,i_pt):
        mf=128
        dk=3
        activate=tf.nn.leaky_relu
        
        num_block=10
        n,h,w,c=i_t.get_shape().as_list()
        with tf.variable_scope('flow',reuse=tf.AUTO_REUSE) as scope:
            i_all=tf.concat([i_t,i_pt],-1)
            x0=i_all
            for p in range(3):
                for q in range(2):
                    x0=tf.layers.conv2d(x0, 32*(2**p), dk, padding='same', activation=activate,name='conv0_{}_{}'.format(p,q))
                x0=tf.layers.max_pooling2d(x0,pool_size=2,strides=2)

            n1,h1,w1,c1=x0.get_shape().as_list()
            for p in range(3):
                for q in range(2):
                    x0=tf.layers.conv2d(x0, 256*(0.5**p), dk, padding='same', activation=activate,name='conv1_{}_{}'.format(p,q))
                x0=tf.image.resize_images(x0,[h1*(2**(p+1)),w1*(2**(p+1))],method=0)
            
            n2,h2,w2,c2=x0.get_shape().as_list()
            if not (h==h2 and w==w2):
                x0=tf.image.resize_images(x0,[h,w],method=0)

            x0=tf.layers.conv2d(x0, 32, dk, padding='same', activation=activate,name='conv2')
            uv=tf.layers.conv2d(x0, 2, dk, padding='same', activation=tf.nn.tanh,name='conv3')

        return uv

    def upscale_warp(self, uv, est):
        n,h,w,c=est.get_shape().as_list()
        upuv=tf.image.resize_images(uv,[h,w],method=0)
        warp_est=imwarp_forward(upuv,est,[h,w])
        warp_est=tf.space_to_depth(warp_est, self.scale, name='est')

        return warp_est

                    
    def build(self):
        in_h,in_w=self.eval_in_size
        H = tf.placeholder(tf.float32, shape=[None, self.num_frames, None, None, 3], name='H_truth')
        L = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_frames, self.in_size, self.in_size, 3], name='L_input')
        L_eval = tf.placeholder(tf.float32, shape=[self.eval_basz, self.num_frames, in_h, in_w, 3], name='L_eval')

        SR_train=[]
        sr=None
        xp=None
        warps_train=[]
        for i in range(self.num_frames):
            if i>0:
                xp=L[:,i-1]
                uv=self.flow(L[:,i],xp)
                warps_train.append(imwarp_forward(uv, xp, [self.in_size, self.in_size]))
            sr= self.forward(L[:,i], xp, sr)
            SR_train.append(sr)
        warps_train=tf.stack(warps_train,axis=1)
        SR_train=tf.stack(SR_train,axis=1)

        SR_eval=[]
        sr=None
        xp=None
        warps_eval=[]
        for i in range(self.num_frames):
            if i>0:
                xp=L_eval[:,i-1]
                uv=self.flow(L_eval[:,i],xp)
                warps_eval.append(imwarp_forward(uv, xp, [in_h, in_w]))
            sr= self.forward(L_eval[:,i], xp, sr)
            SR_eval.append(sr)
        warps_eval=tf.stack(warps_eval,axis=1)
        SR_eval=tf.stack(SR_eval,axis=1)


        sr_loss=tf.reduce_mean((SR_train-H)**2)
        eval_mse=tf.reduce_mean((SR_eval-H) ** 2, axis=[2,3,4])#[:,self.num_frames//2:self.num_frames//2+1]
        flow_loss=tf.reduce_mean((warps_train-L[:,1:])**2)
        flow_mse=tf.reduce_mean((warps_train-L[:,1:])**2, axis=[2,3,4])
        self.sr_loss, self.eval_mse, self.flow_loss, self.flow_mse= sr_loss, eval_mse, flow_loss, flow_mse
        self.all_loss=self.sr_loss+self.flow_loss
        self.L, self.L_eval, self.H, self.SR =  L, L_eval, H, SR_train
        
    def eval(self):
        print('Evaluating ...')
        if not hasattr(self, 'sess'):
            global_step=tf.Variable(initial_value=0, trainable=False)
            self.global_step=global_step
            self.build()
            sess = tf.Session()
            self.load(sess, self.save_dir)
        else:
            sess = self.sess
            
        border=8
        in_h,in_w=self.eval_in_size
        out_h = in_h*self.scale #512
        out_w = in_w*self.scale #960
        bd=border//self.scale
        
        filenames=open(self.eval_dir, 'rt').read().splitlines()#sorted(glob.glob(join(self.train_dir,'*')))
        hr_list=[sorted(glob.glob(join(f,'truth','*.png'))) for f in filenames]
        lr_list=[sorted(glob.glob(join(f,'blur{}'.format(self.scale),'*.png'))) for f in filenames]
        
        center=15
        batch_hr = []
        batch_lr = []
        batch_cnt=0
        mse_acc=None
        for lrlist,hrlist in zip(lr_list,hr_list):
            max_frame=len(lrlist)
            for idx0 in range(center, max_frame, 32):
                index=np.array([i for i in range(idx0-self.num_frames//2,idx0+self.num_frames//2)])
                index=np.clip(index,0,max_frame-1).tolist()
                gt=[cv2_imread(hrlist[i]) for i in index]
                inp=[cv2_imread(lrlist[i]) for i in index]
                inp=[i[bd:in_h+bd, bd:in_w+bd].astype(np.float32) / 255.0 for i in inp]
                gt = [i[border:out_h+border, border:out_w+border, :].astype(np.float32) / 255.0 for i in gt]
                batch_hr.append(np.stack(gt, axis=0))
                batch_lr.append(np.stack(inp, axis=0))
                
                if len(batch_hr) == self.eval_basz:
                    batch_hr = np.stack(batch_hr, 0)
                    batch_lr = np.stack(batch_lr, 0)
                    mse_val=sess.run(self.eval_mse,feed_dict={self.L_eval:batch_lr, self.H:batch_hr})
                    if mse_acc is None:
                        mse_acc = mse_val
                    else:
                        mse_acc = np.concatenate([mse_acc, mse_val], axis=0)
                    batch_hr = []
                    batch_lr=[]
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
        LR, HR= self.frvsr_input_producer() #input_producer(train_dir=self.train_dir, batch_size=self.batch_size, scale=self.scale, in_size=self.in_size, num_frames=self.num_frames)
        global_step=tf.Variable(initial_value=0, trainable=False)
        self.global_step=global_step
        self.build()
        lr= tf.train.polynomial_decay(self.learning_rate, global_step, self.decay_step, end_learning_rate=self.end_lr, power=1.)
        
        vars_all=tf.trainable_variables()
        vars_sr = [v for v in vars_all if 'frvsr' in v.name]
        vars_flow = [v for v in vars_all if 'flow' in v.name]
        #print ('Params:',np.sum([np.prod(v.get_shape().as_list()) for v in vars_all]))
        print('Params num of flow:',get_num_params(vars_flow))
        print('Params num of sr:',get_num_params(vars_sr))
        print('Params num of all:',get_num_params(vars_all))
        training_op = tf.train.AdamOptimizer(lr).minimize(self.all_loss, var_list=vars_all, global_step=global_step)
        
        
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
        for step in range(gs, self.max_step):
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
            _,loss_v=sess.run([training_op,self.all_loss],feed_dict={self.L:lr1, self.H:hr})
            
            if step>500 and loss_v>10:
                print('Model collapsed with loss={}'.format(loss_v))
                break
                
            
    def test_video(self, path, name='result', reuse=False):
        save_path=join(path,name)
        automkdir(save_path)
        
        inp_path=join(path,'blur{}'.format(self.scale))
        imgs=sorted(glob.glob(join(inp_path,'*.png')))
        imgs=np.array([cv2_imread(i)/255. for i in imgs])
        n,h,w,c=imgs.shape
        max_frame=n
        
        self.L = tf.placeholder(tf.float32, shape=[1, h, w, 3], name='L_input')
        self.LP = tf.placeholder(tf.float32, shape=[1, h, w, 3], name='Previous_L_input')
        self.est=tf.placeholder(tf.float32, shape=[1, h*self.scale, w*self.scale, 3], name='est')
        self.sr0 = self.forward(self.L)
        self.sr1 = self.forward(self.L, self.LP, self.est)
        if not reuse:
            config = tf.ConfigProto() 
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            #sess=tf.Session()
            self.sess=sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)
            self.load(sess, self.save_dir)
        
        print('Save at {}'.format(save_path))
        print('{} Inputs With Shape {}'.format(imgs.shape[0],imgs.shape[1:]))

        all_time=[]
        for i in trange(max_frame):
            st_time=time.time()
            if i==0:
                SR=self.sess.run(self.sr0,feed_dict={self.L : imgs[i:i+1]})
            else:
                SR=self.sess.run(self.sr1,feed_dict={self.L : imgs[i:i+1], self.LP : imgs[i-1:i], self.est : SR})
            all_time.append(time.time()-st_time)
            img=SR[0]*255.
            img=np.clip(img,0,255).astype(np.uint8)
            cv2_imsave(join(save_path, '{:0>4}.png'.format(i)),img)
        if max_frame>0:
            all_time=np.array(all_time)
            print('spent {} s in total and {} s in average'.format(np.sum(all_time),np.mean(all_time[1:])))

    def testvideos(self, path='/dev/f/data/video/test2/udm10', start=0, name='frvsr'):
        kind=sorted(glob.glob(join(path,'*')))
        kind=[k for k in kind if os.path.isdir(k)]
        reuse=False
        for k in kind:
            idx=kind.index(k)
            if idx>=start:
                if idx>start:
                    reuse=True
                datapath=join(path,k)
                self.test_video(datapath, name=name, reuse=reuse)
    
        
if __name__=='__main__':
    model=FRVSR()
    model.train()
    #model.testvideos()