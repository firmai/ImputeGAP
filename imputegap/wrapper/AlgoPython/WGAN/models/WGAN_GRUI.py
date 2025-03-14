#-*- coding: utf-8 -*-
from __future__ import division
import os
import math
import time
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops

from .ops import *
from .utils import *

"""
D输入标准化， 不要m 填充0
G输入去掉m,只有delta
g 没有每次累加z
"""
class WGAN(object):
    model_name = "WGAN_no_mask"     # name for checkpoint

    def __init__(self, sess, args, datasets):
        self.sess = sess
        self.isbatch_normal=args.isBatch_normal # default True
        self.isNormal=args.isNormal # default True
        self.lr = args.lr   #default 1e-3
        self.epoch = args.epoch     #default 30
        self.batch_size = args.batch_size  #default 128
        self.n_inputs = args.n_inputs                 # MNIST data input (img shape: 28*28) Default 41
        self.n_steps = datasets.maxLength                # time steps
        self.n_hidden_units = args.n_hidden_units        # neurons in hidden layer default 64
        self.n_classes = args.n_classes                # MNIST classes (0-9 digits) default 2
        self.pretrain_epoch=args.pretrain_epoch # default 5
        self.impute_iter=args.impute_iter # default 400
        self.isSlicing=args.isSlicing # default True
        self.g_loss_lambda=args.g_loss_lambda #default 0.1

        self.datasets=datasets
        self.z_dim = args.z_dim         # dimension of noise-vector
        self.gen_length=args.gen_length

        self.shape = args.shape

        # WGAN_GP parameter
        self.lambd = 0.25       # The higher value, the more stable, but the slower convergence
        self.disc_iters = args.disc_iters     # The number of critic iterations for one-step of generator

        # train
        self.learning_rate = args.lr
        self.beta1 = args.beta1

        self.grud_cell_d = tf.keras.layers.GRU(self.n_hidden_units, return_sequences=True, return_state=True)
        self.grud_cell_g = tf.keras.layers.GRU(self.n_hidden_units, return_sequences=True, return_state=True)

        # test
        self.sample_num = 64  # number of generated images to be saved

        self.num_batches = len(datasets.x) // self.batch_size


    def pretrainG(self, X, M, Delta,  Mean, Lastvalues, X_lengths, Keep_prob, is_training=True, reuse=False):

        with tf.name_scope("g_enerator"):
            """
            the rnn cell's variable scope is defined by tensorflow,
            if we want to update rnn cell's weights, the variable scope must contains 'g_' or 'd_'
            
            """

            wr_h = tf.Variable(tf.random.normal([self.n_inputs, self.n_hidden_units]), name="g_wr_h")
            w_out = tf.Variable(tf.random.normal([self.n_hidden_units, self.n_inputs]), name="g_w_out")

            br_h = tf.Variable(tf.zeros([self.n_hidden_units]), name="g_br_h")
            b_out = tf.Variable(tf.zeros([self.n_inputs]), name="g_b_out")

            w_z = tf.Variable(tf.random.normal([self.z_dim, self.n_inputs]), name="g_w_z")
            b_z = tf.Variable(tf.zeros([self.n_inputs]), name="g_b_z")

            X = tf.reshape(tf.convert_to_tensor(X), [-1, self.n_inputs])
            Delta = tf.reshape(tf.convert_to_tensor(Delta), [-1, self.n_inputs])

            rth= tf.matmul(Delta, wr_h)+br_h
            rth=math_ops.exp(-tf.maximum(0.0,rth))

            X=tf.concat([X,rth],1)

            X_in = tf.reshape(X, [-1, self.n_steps, self.n_inputs + self.n_hidden_units])

            init_state = None  # No need for zero_state in Keras

            outputs, final_state = self.grud_cell_g(X_in, training=is_training)

            #outputs: batch_size*n_steps*n_hiddensize
            outputs=tf.reshape(outputs,[-1,self.n_hidden_units])
            out_predict=tf.matmul(tf.nn.dropout(outputs,Keep_prob), w_out) + b_out
            out_predict=tf.reshape(out_predict,[-1,self.n_steps,self.n_inputs])
            return out_predict


    def discriminator(self, X, M, DeltaPre, Lastvalues ,DeltaSub ,SubValues , Mean,  X_lengths,Keep_prob, is_training=True, reuse=False, isTdata=True):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("d_iscriminator", reuse=reuse):

            wr_h=tf.get_variable("d_wr_h",shape=[self.n_inputs,self.n_hidden_units],initializer=tf.random_normal_initializer())
            w_out= tf.get_variable("d_w_out",shape=[self.n_hidden_units, 1],initializer=tf.random_normal_initializer())
            br_h= tf.get_variable("d_br_h",shape=[self.n_hidden_units, ],initializer=tf.constant_initializer(0.001))
            b_out= tf.get_variable("d_b_out",shape=[1, ],initializer=tf.constant_initializer(0.001))


            M=tf.reshape(M,[-1,self.n_inputs])
            X = tf.reshape(X, [-1, self.n_inputs])
            DeltaPre=tf.reshape(DeltaPre,[-1,self.n_inputs])


            rth= tf.matmul(DeltaPre, wr_h)+br_h
            rth=math_ops.exp(-tf.maximum(0.0,rth))
            # add noise
            #X=X+np.random.standard_normal(size=(self.batch_size*self.n_steps, self.n_inputs))/100
            X=tf.concat([X,rth],1)

            X_in = tf.reshape(X, [self.batch_size, self.n_steps, self.n_inputs + self.n_hidden_units])

            init_state = self.grud_cell_d.zero_state(self.batch_size, dtype=tf.float32) # 初始化全零 state
            outputs, final_state = self.grud_cell_g(X_in, mask=None, training=is_training)


            # final_state:batch_size*n_hiddensize
            # 不能用最后一个，应该用第length个  之前用了最后一个，所以输出无论如何都是b_out
            out_logit=tf.matmul(tf.nn.dropout(final_state,Keep_prob), w_out) + b_out
            out =tf.nn.sigmoid(out_logit)    #选取最后一个 output
            return out,out_logit

    def generator(self, z, Keep_prob, is_training=True, reuse=False):
        # x,delta,n_steps
        # z :[self.batch_size, self.z_dim]
        # first feed noize in rnn, then feed the previous output into next input
        # or we can feed noize and previous output into next input in future version
        with tf.name_scope("g_enerator"):
            #gennerate

            wr_h = tf.Variable(tf.random.normal([self.n_inputs, self.n_hidden_units]), name="g_wr_h")
            w_out = tf.Variable(tf.random.normal([self.n_hidden_units, self.n_inputs]), name="g_w_out")

            br_h = tf.Variable(tf.zeros([self.n_hidden_units]), name="g_br_h")
            b_out = tf.Variable(tf.zeros([self.n_inputs]), name="g_b_out")

            w_z = tf.Variable(tf.random.normal([self.z_dim, self.n_inputs]), name="g_w_z")
            b_z = tf.Variable(tf.zeros([self.n_inputs]), name="g_b_z")

            #self.times=tf.reshape(self.times,[self.batch_size,self.n_steps,self.n_inputs])
            #change z's dimension
            # batch_size*z_dim-->batch_size*n_inputs
            x=tf.matmul(z,w_z)+b_z
            x=tf.reshape(x,[-1,self.n_inputs])
            delta_zero=tf.constant(0.0,shape=[self.batch_size,self.n_inputs])
            #delta_normal=tf.constant(48.0*60.0/self.gen_length,shape=[self.batch_size,self.n_inputs])
            #delta:[batch_size,1,n_inputs]


            # combine X_in
            rth= tf.matmul(delta_zero, wr_h)+br_h
            rth=math_ops.exp(-tf.maximum(0.0,rth))
            x=tf.concat([x,rth],1)

            X_in = tf.reshape(x, [-1, 1, self.n_inputs+self.n_hidden_units])

            init_state = self.grud_cell_g.zero_state(self.batch_size, dtype=tf.float32) # 初始化全零 state
            #z=tf.reshape(z,[self.batch_size,1,self.z_dim])
            seq_len=tf.constant(1,shape=[self.batch_size])

            outputs, final_state = tf.nn.dynamic_rnn(self.grud_cell_g, X_in, \
                                initial_state=init_state,\
                                sequence_length=seq_len,
                                time_major=False)
            init_state=final_state
            #outputs: batch_size*1*n_hidden
            outputs=tf.reshape(outputs,[-1,self.n_hidden_units])
            # full connect
            out_predict=tf.matmul(tf.nn.dropout(outputs,Keep_prob), w_out) + b_out
            out_predict=tf.reshape(out_predict,[-1,1,self.n_inputs])

            total_result=tf.multiply(out_predict,1.0)

            for i in range(1,self.n_steps):
                out_predict=tf.reshape(out_predict,[self.batch_size,self.n_inputs])
                #输出加上noise z
                #out_predict=out_predict+tf.matmul(z,w_z)+b_z
                #
                delta_normal=tf.reshape(self.imputed_deltapre[:,i:(i+1),:],[self.batch_size,self.n_inputs])
                rth= tf.matmul(delta_normal, wr_h)+br_h
                rth=math_ops.exp(-tf.maximum(0.0,rth))
                x=tf.concat([out_predict,rth],1)
                X_in = tf.reshape(x, [-1, 1, self.n_inputs+self.n_hidden_units])

                outputs, final_state = tf.nn.dynamic_rnn(self.grud_cell_g, X_in, \
                            initial_state=init_state,\
                            sequence_length=seq_len,
                            time_major=False)
                init_state=final_state
                outputs=tf.reshape(outputs,[-1,self.n_hidden_units])
                out_predict=tf.matmul(tf.nn.dropout(outputs,Keep_prob), w_out) + b_out
                out_predict=tf.reshape(out_predict,[-1,1,self.n_inputs])
                total_result=tf.concat([total_result,out_predict],1)

            #delta:[batch_size,,n_inputs]

            if self.isbatch_normal:
                with tf.variable_scope("g_bn", reuse=tf.AUTO_REUSE):
                    total_result=bn(total_result,is_training=is_training, scope="g_bn_imple")


            last_values=tf.multiply(total_result,1)
            sub_values=tf.multiply(total_result,1)

            return total_result,self.imputed_deltapre,self.imputed_deltasub,self.imputed_m,self.x_lengths,last_values,sub_values

    def impute(self):
        with tf.variable_scope("impute", reuse=tf.AUTO_REUSE):
            z_need_tune=tf.get_variable("z_needtune",shape=[self.batch_size,self.z_dim],initializer=tf.random_normal_initializer(mean=0,stddev=0.1) )
            return z_need_tune

    def build_model(self):

        self.keep_prob = tf.Variable(1.0, dtype=tf.float32)
        self.x = tf.Variable(tf.zeros([self.batch_size, self.n_steps, self.n_inputs]), dtype=tf.float32)
        self.y = tf.Variable(tf.zeros([self.batch_size, self.n_classes]), dtype=tf.float32)
        self.m = tf.Variable(tf.zeros([self.batch_size, self.n_steps, self.n_inputs]), dtype=tf.float32)
        self.mean = tf.Variable(tf.zeros([self.n_inputs]), dtype=tf.float32)
        self.deltaPre = tf.Variable(tf.zeros([self.batch_size, self.n_steps, self.n_inputs]), dtype=tf.float32)
        self.lastvalues = tf.Variable(tf.zeros([self.batch_size, self.n_steps, self.n_inputs]), dtype=tf.float32)
        self.deltaSub = tf.Variable(tf.zeros([self.batch_size, self.n_steps, self.n_inputs]), dtype=tf.float32)
        self.subvalues = tf.Variable(tf.zeros([self.batch_size, self.n_steps, self.n_inputs]), dtype=tf.float32)
        self.x_lengths = tf.Variable(tf.zeros([self.batch_size], dtype=tf.int32), dtype=tf.int32)
        self.imputed_deltapre = tf.Variable(tf.zeros([self.batch_size, self.n_steps, self.n_inputs]), dtype=tf.float32)
        self.imputed_deltasub = tf.Variable(tf.zeros([self.batch_size, self.n_steps, self.n_inputs]), dtype=tf.float32)
        self.imputed_m = tf.Variable(tf.zeros([self.batch_size, self.n_steps, self.n_inputs]), dtype=tf.float32)
        self.z = tf.Variable(tf.zeros([self.batch_size, self.z_dim]), dtype=tf.float32, name='z')

        """ Loss Function """
        Pre_out=self.pretrainG(self.x, self.m, self.deltaPre,  self.mean,\
                                                      self.lastvalues, self.x_lengths,self.keep_prob, \
                                                      is_training=True, reuse=False)

        self.pretrain_loss=tf.reduce_sum(tf.square(tf.multiply(Pre_out,self.m)-self.x)) / tf.cast(tf.reduce_sum(self.x_lengths),tf.float32)

        #discriminator( X, M, DeltaPre, Lastvalues ,DeltaSub ,SubValues , Mean,  X_lengths,Keep_prob, is_training=True, reuse=False, isTdata=True):

        D_real, D_real_logits = self.discriminator(self.x, self.m, self.deltaPre,self.lastvalues,\
                                                   self.deltaSub,self.subvalues,  self.mean,\
                                                       self.x_lengths,self.keep_prob, \
                                                      is_training=True, reuse=False,isTdata=True)

        #G return total_result,self.imputed_deltapre,self.imputed_deltasub,self.imputed_m,self.x_lengths,last_values,sub_values
        g_x,g_deltapre,g_deltasub,g_m,G_x_lengths,g_last_values,g_sub_values = self.generator(self.z,self.keep_prob, is_training=True, reuse=True)

        D_fake, D_fake_logits = self.discriminator(g_x,g_m,g_deltapre,g_last_values,\
                                                   g_deltasub,g_sub_values,self.mean,\
                                                      G_x_lengths,self.keep_prob,
                                                      is_training=True, reuse=True ,isTdata=False)

        """
        impute loss
        """
        self.z_need_tune=self.impute()

        impute_out,impute_deltapre,impute_deltasub,impute_m,impute_x_lengths,impute_last_values,impute_sub_values=self.generator(self.z_need_tune,self.keep_prob, is_training=False, reuse=True)


        impute_fake, impute_fake_logits = self.discriminator(impute_out,impute_m,impute_deltapre,impute_last_values,\
                                                             impute_deltasub,impute_sub_values,self.mean,\
                                                      impute_x_lengths,self.keep_prob,
                                                      is_training=False, reuse=True ,isTdata=False)

        # loss for imputation

        self.impute_loss=tf.reduce_mean(tf.square(tf.multiply(impute_out,self.m)-self.x))-self.g_loss_lambda*tf.reduce_mean(impute_fake_logits)
        #self.impute_loss=tf.reduce_mean(tf.square(tf.multiply(impute_out,self.m)-self.x))

        self.impute_out=impute_out

        #the imputed results
        self.imputed=tf.multiply((1-self.m),self.impute_out)+self.x
        # get loss for discriminator
        d_loss_real = - tf.reduce_mean(D_real_logits)
        d_loss_fake = tf.reduce_mean(D_fake_logits)


        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = - d_loss_fake


        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        z_vars = [self.z_need_tune]
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # this code have used batch normalization, so the upside line should be executed
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                        .minimize(self.d_loss, var_list=d_vars)
            #self.d_optim=self.optim(self.learning_rate, self.beta1,self.d_loss,d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*self.disc_iters, beta1=self.beta1) \
                        .minimize(self.g_loss, var_list=g_vars)
            #self.g_optim=self.optim(self.learning_rate, self.beta1,self.g_loss,g_vars)
            self.g_pre_optim=tf.train.AdamOptimizer(self.learning_rate*2,beta1=self.beta1) \
                        .minimize(self.pretrain_loss,var_list=g_vars)
        self.impute_optim=tf.train.AdamOptimizer(self.learning_rate*7,beta1=self.beta1) \
                    .minimize(self.impute_loss,var_list=z_vars)




        #clip weight
        self.clip_all_vals = [p.assign(tf.clip_by_value(p, -0.99, 0.99)) for p in t_vars]
        self.clip_D = [p.assign(tf.clip_by_value(p, -0.99, 0.99)) for p in d_vars]
        self.clip_G = [p.assign(tf.clip_by_value(p, -0.99, 0.99)) for p in g_vars]


        """" Testing """
        # for test
        self.fake_x,self.fake_delta,_,_,_,_,_ = self.generator(self.z, self.keep_prob, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        g_pretrain_loss_sum=tf.summary.scalar("g_pretrain_loss", self.pretrain_loss)
        # final summary operations
        self.impute_sum=tf.summary.scalar("impute_loss", self.impute_loss)
        self.g_sum = g_loss_sum
        self.g_pretrain_sum=tf.summary.merge([g_pretrain_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum,d_loss_fake_sum, d_loss_sum])

    def optim(self,learning_rate,beta,loss,var):
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta)
        grads = optimizer.compute_gradients(loss,var_list=var)
        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
        train_op = optimizer.apply_gradients(grads)
        return train_op

    def pretrain(self, start_epoch,counter,start_time):

        if start_epoch < self.pretrain_epoch:
            #todo
            for epoch in range(start_epoch, self.pretrain_epoch):
            # get batch data
                self.datasets.shuffle(self.batch_size,True)
                idx=0
                for data_x,data_y,data_mean,data_m,data_deltaPre,data_x_lengths,data_lastvalues,_,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub in self.datasets.nextBatch():

                    # pretrain
                    _, summary_str, p_loss = self.sess.recoveryBitGRAPH([self.g_pre_optim, self.g_pretrain_sum, self.pretrain_loss],
                                                                        feed_dict={self.x: data_x,
                                                              self.m: data_m,
                                                              self.deltaPre: data_deltaPre,
                                                              self.mean: data_mean,
                                                              self.x_lengths: data_x_lengths,
                                                              self.lastvalues: data_lastvalues,
                                                              self.deltaSub:deltaSub,
                                                              self.subvalues:subvalues,
                                                              self.imputed_m:imputed_m,
                                                              self.imputed_deltapre:imputed_deltapre,
                                                              self.imputed_deltasub:imputed_deltasub,
                                                              self.keep_prob: 0.5})
                    counter += 1

                    # display training status
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, pretrain_loss: %.8f" \
                          % (epoch, idx, self.num_batches, time.time() - start_time, p_loss), end='\r')
                    idx+=1


    def train(self):

        tf.global_variables_initializer().recoveryBitGRAPH()
        start_epoch = 0
        start_batch_id = 0
        counter = 1
        # loop for epoch
        start_time = time.time()
        print('Start pretrain')
        self.pretrain(start_epoch,counter,start_time)
        if start_epoch < self.pretrain_epoch:
            start_epoch=self.pretrain_epoch

        for epoch in range(start_epoch, self.epoch):

            # get batch data
            self.datasets.shuffle(self.batch_size,True)
            idx=0
            for data_x,data_y,data_mean,data_m,data_deltaPre,data_x_lengths,data_lastvalues,_,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub in self.datasets.nextBatch():

                batch_z = np.random.standard_normal(size=(self.batch_size, self.z_dim))
                #_ = self.sess.run(self.clip_D)
                _ = self.sess.recoveryBitGRAPH(self.clip_all_vals)
                _, summary_str, d_loss = self.sess.recoveryBitGRAPH([self.d_optim, self.d_sum, self.d_loss],
                                                                    feed_dict={self.z: batch_z,
                                                          self.x: data_x,
                                                          self.m: data_m,
                                                          self.deltaPre: data_deltaPre,
                                                          self.mean: data_mean,
                                                          self.x_lengths: data_x_lengths,
                                                          self.lastvalues: data_lastvalues,
                                                          self.deltaSub:deltaSub,
                                                          self.subvalues:subvalues,
                                                          self.imputed_m:imputed_m,
                                                          self.imputed_deltapre:imputed_deltapre,
                                                          self.imputed_deltasub:imputed_deltasub,
                                                          self.keep_prob: 0.5})

                # update G network
                if counter%self.disc_iters==0:
                    #batch_z = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                    _, summary_str, g_loss = self.sess.recoveryBitGRAPH([self.g_optim, self.g_sum, self.g_loss],
                                                                        feed_dict={self.z: batch_z,
                                                           self.keep_prob: 0.5,
                                                           self.deltaPre: data_deltaPre,
                                                           self.mean: data_mean,
                                                           self.x_lengths: data_x_lengths,
                                                           self.lastvalues: data_lastvalues,
                                                           self.deltaSub:deltaSub,
                                                           self.subvalues:subvalues,
                                                           self.imputed_m:imputed_m,
                                                           self.imputed_deltapre:imputed_deltapre,
                                                           self.imputed_deltasub:imputed_deltasub,
                                                           self.mean: data_mean})
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f,counter:%4d" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss,counter), end='\r')
                    #debug

                counter += 1

                # display training status
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, counter:%4d" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, counter), end='\r')

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0 :
                    fake_x,fake_delta = self.sess.recoveryBitGRAPH([self.fake_x, self.fake_delta],
                                                                   feed_dict={self.z: batch_z,
                                                       self.deltaPre: data_deltaPre,
                                                       self.mean: data_mean,
                                                       self.x_lengths: data_x_lengths,
                                                       self.lastvalues: data_lastvalues,
                                                       self.deltaSub:deltaSub,
                                                       self.subvalues:subvalues,
                                                       self.imputed_m:imputed_m,
                                                       self.imputed_deltapre:imputed_deltapre,
                                                       self.imputed_deltasub:imputed_deltasub,
                                                       self.mean: data_mean,
                                                       self.keep_prob: 0.5})
                idx+=1
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0


        # self.save(self.checkpoint_dir, counter)

    def imputation(self,dataset,isTrain):
        self.output_mat = []
        self.tmp = []
        self.row = 0
        self.col = 0
        self.datasets = dataset
        self.datasets.shuffle(self.batch_size,False)
        tf.variables_initializer([self.z_need_tune]).recoveryBitGRAPH()
        start_time = time.time()
        batchid=1
        impute_tune_time=1

        for data_x,data_y,data_mean,data_m,data_deltaPre,data_x_lengths,data_lastvalues,_,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub in self.datasets.nextBatch():
            counter=1
            tf.variables_initializer([self.z_need_tune]).recoveryBitGRAPH()
            for i in range(0,self.impute_iter):
                _, impute_out, summary_str, impute_loss, imputed = self.sess.recoveryBitGRAPH([self.impute_optim, self.impute_out, self.impute_sum, self.impute_loss, self.imputed], \
                                                                                              feed_dict={self.x: data_x,
                                                                  self.m: data_m,
                                                                  self.deltaPre: data_deltaPre,
                                                                  self.mean: data_mean,
                                                                  self.x_lengths: data_x_lengths,
                                                                  self.lastvalues: data_lastvalues,
                                                                  self.deltaSub:deltaSub,
                                                                  self.subvalues:subvalues,
                                                                  self.imputed_m:imputed_m,
                                                                  self.imputed_deltapre:imputed_deltapre,
                                                                  self.imputed_deltasub:imputed_deltasub,
                                                                  self.keep_prob: 1.0})
                impute_tune_time+=1
                counter+=1
                if counter%10==0:
                    print("Batchid: [%2d] [%4d/%4d] time: %4.4f, impute_loss: %.8f, shape: %d,%d" \
                          % (batchid, i, self.impute_iter, time.time() - start_time, impute_loss, imputed.shape[0], imputed.shape[1]), end='\r')
                if counter == self.impute_iter:
                    for imputed_ts in imputed:
                        self.tmp.append(imputed_ts)

            batchid+=1
            impute_tune_time+=1
        self.output_mat = np.array(self.tmp)
        return self.output_mat.squeeze()

