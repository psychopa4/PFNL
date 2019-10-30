import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from os.path import join,exists
import glob
import random
import numpy as np
from PIL import Image
import time
import os
from utils import Huber,LoadImage, DownSample, DownSample_4D, BLUR, AVG_PSNR, depth_to_space_3D, DynFilter3D, LoadParams, automkdir, get_num_params, cv2_imread,cv2_imsave
from model.nets import FR_16L, FR_28L, FR_52L
from model.base_model import VSR
from tqdm import trange,tqdm
        
'''This work tries to rebuild DUFVSR (Deep Video Super-Resolution Network Using Dynamic Upsampling Filters Without Explicit Motion Compensation).
The code is mainly based on https://github.com/psychopa4/MMCNN, https://github.com/jiangsutx/SPMC_VideoSR and https://github.com/yhjo09/VSR-DUF.
'''
        
class DUFVSR(VSR):
    def __init__(self):
        self.num_frames=7
        self.scale=4
        self.in_size=32
        self.gt_size=self.in_size*self.scale
        self.eval_in_size=[128,240]
        self.batch_size=11  #can be increased with larger GPU memory
        self.eval_basz=4
        self.learning_rate=1e-3
        self.end_lr=1e-4
        self.reload=True
        self.max_step=int(1.5e5+1)
        self.decay_step=1.2e5
        self.train_dir='./data/filelist_train.txt'
        self.eval_dir='./data/filelist_val.txt'
        self.save_dir='./checkpoint/duf_52'
        self.log_dir='./duf_52.txt'
            
    def forward(self, x, is_train):  
        # shape of x: [B,T_in,H,W,C]

        # Generate filters and residual
        # Fx: [B,1,H,W,1*5*5,R*R]
        # Rx: [B,1,H,W,3*R*R]
        with tf.variable_scope('G',reuse=tf.AUTO_REUSE) as scope:
            Fx, Rx = FR_52L(x, is_train) 

            x_c = []
            for c in range(3):
                t = DynFilter3D(x[:,self.num_frames//2:self.num_frames//2+1,:,:,c], Fx[:,0,:,:,:,:], [1,5,5]) # [B,H,W,R*R]
                t = tf.depth_to_space(t, self.scale) # [B,H*R,W*R,1]
                x_c += [t]
            x = tf.concat(x_c, axis=3)   # [B,H*R,W*R,3]
            x = tf.expand_dims(x, axis=1)

            Rx = depth_to_space_3D(Rx, self.scale)   # [B,1,H*R,W*R,3]
            x += Rx
            
            return x
                    
    def build(self):
        H = tf.placeholder(tf.float32, shape=[None, 1, None, None, 3], name='H_truth')
        L = tf.placeholder(tf.float32, shape=[None, self.num_frames, None, None, 3], name='L_input')
        is_train = tf.placeholder(tf.bool, shape=[]) # Phase ,scalar
        SR = self.forward(L,is_train)
        loss=Huber(SR,H,0.01)#tf.reduce_mean(tf.sqrt((SR-H)**2+1e-6))
        eval_mse=tf.reduce_mean((SR-H) ** 2, axis=[2,3,4])#[:,self.num_frames//2:self.num_frames//2+1]
        self.loss, self.eval_mse= loss, eval_mse
        self.L, self.H, self.SR, self.is_train =  L, H, SR, is_train
        
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
        
        filenames=open(self.eval_dir, 'rt').read().splitlines()
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
                index=np.array([i for i in range(idx0-self.num_frames//2,idx0+self.num_frames//2+1)])
                index=np.clip(index,0,max_frame-1).tolist()
                gt=[cv2_imread(hrlist[idx0])]
                inp=[cv2_imread(lrlist[i]) for i in index]
                inp=[i[bd:in_h+bd, bd:in_w+bd].astype(np.float32) / 255.0 for i in inp]
                gt = [i[border:out_h+border, border:out_w+border, :].astype(np.float32) / 255.0 for i in gt]
                batch_hr.append(np.stack(gt, axis=0))
                batch_lr.append(np.stack(inp, axis=0))
                
                if len(batch_hr) == self.eval_basz:
                    batch_hr = np.stack(batch_hr, 0)
                    batch_lr = np.stack(batch_lr, 0)
                    mse_val=sess.run(self.eval_mse,feed_dict={self.L:batch_lr, self.H:batch_hr, self.is_train:False})
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
        LR, HR= self.double_input_producer()
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
        
        self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)
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
            _,loss_v=sess.run([training_op,self.loss],feed_dict={self.L:lr1, self.H:hr, self.is_train:True})
            
            if step>500 and loss_v>10:
                print('Model collapsed with loss={}'.format(loss_v))
                break
                
            
    def test_video_truth(self, path, name='result', reuse=False, part=8):
        save_path=join(path,name)
        automkdir(save_path)
        
        imgs=sorted(glob.glob(join(path,'truth','*.png')))
        imgs=[cv2_imread(i)/255. for i in imgs]
        
        test_gt = tf.placeholder(tf.float32, [None, self.num_frames, None, None, 3])
        test_inp=DownSample(test_gt, BLUR, scale=self.scale)
        
        if not reuse:
            self.build()
            sess=tf.Session()
            self.sess=sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)
            self.load(sess, self.save_dir)
        
        gt_list=[]
        max_frame=len(imgs)
        for i in range(max_frame):
            index=np.array([i for i in range(i-self.num_frames//2,i+self.num_frames//2+1)])
            index=np.clip(index,0,max_frame-1).tolist()
            gt=np.array([imgs[j] for j in index])
            gt_list.append(gt)
        gt_list=np.array(gt_list)
        lr_list=self.sess.run(test_inp,feed_dict={test_gt:gt_list})
        print('Save at {}'.format(save_path))
        print('{} Inputs With Shape {}'.format(lr_list.shape[0],lr_list.shape[1:]))
        
        part=min(part,max_frame)
        if max_frame%part ==0 :
            num_once=max_frame//part
        else:
            num_once=max_frame//part+1
        
        all_time=0
        for i in trange(part):
            st_time=time.time()
            sr=self.sess.run(self.SR,feed_dict={self.L : lr_list[i*num_once:(i+1)*num_once], self.is_train : False})
            onece_time=time.time()-st_time
            if i>0:
                all_time+=onece_time
            for j in range(sr.shape[0]):
                img=sr[j][0]*255.
                img=np.clip(img,0,255).astype(np.uint8)
                imgname='{:0>4}.png'.format(i*num_once+j)
                cv2_imsave(join(save_path, imgname),img)
        print('spent {} s in total and {} s in average'.format(all_time,all_time/(max_frame-1)))

    def test_video_lr(self, path, name='result', reuse=False, part=8):
        save_path=join(path,name)
        automkdir(save_path)
        
        inp_path=join(path,'blur{}'.format(self.scale))
        imgs=sorted(glob.glob(join(inp_path,'*.png')))
        imgs=np.array([cv2_imread(i)/255. for i in imgs])
        
        lr_list=[]
        max_frame=imgs.shape[0]
        for i in range(max_frame):
            index=np.array([i for i in range(i-self.num_frames//2,i+self.num_frames//2+1)])
            index=np.clip(index,0,max_frame-1).tolist()
            lr_list.append(np.array([imgs[j] for j in index]))
        lr_list=np.array(lr_list)

        if not reuse:
            self.build()
            sess=tf.Session()
            self.sess=sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)
            self.load(sess, self.save_dir)

        print('Save at {}'.format(save_path))
        print('{} Inputs With Shape {}'.format(lr_list.shape[0],lr_list.shape[1:]))

        part=min(part,max_frame)
        if max_frame%part ==0 :
            num_once=max_frame//part
        else:
            num_once=max_frame//part+1
        
        all_time=0
        for i in trange(part):
            st_time=time.time()
            sr=self.sess.run(self.SR,feed_dict={self.L : lr_list[i*num_once:(i+1)*num_once], self.is_train : False})
            onece_time=time.time()-st_time
            if i>0:
                all_time+=onece_time
            for j in range(sr.shape[0]):
                img=sr[j][0]*255.
                img=np.clip(img,0,255).astype(np.uint8)
                imgname='{:0>4}.png'.format(i*num_once+j)
                cv2_imsave(join(save_path, imgname),img)
        print('spent {} s in total and {} s in average'.format(all_time,all_time/(max_frame-1)))

    def testvideos(self, path='/dev/f/data/video/test2/vid4', start=0, name='duf_52'):
        kind=sorted(glob.glob(join(path,'*')))
        kind=[k for k in kind if os.path.isdir(k)]
        reuse=False
        for k in kind:
            idx=kind.index(k)
            if idx>=start:
                if idx>start:
                    reuse=True
                datapath=join(path,k)
                self.test_video_lr(datapath, name=name, reuse=reuse, part=1000)
            
    
        
if __name__=='__main__':
    model=DUFVSR()
    #model.train()
    model.testvideos()