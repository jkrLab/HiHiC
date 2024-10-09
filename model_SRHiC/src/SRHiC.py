#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
import os

import random
import time, datetime

# params

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# epoch_size = 256

def res_block(_input, feature_size=32):
    output = slim.conv2d(_input, feature_size * 4, [1, 1])
    output = tf.nn.relu(output)
    output = slim.conv2d(output, feature_size, [1, 1])
    output = slim.conv2d(output, feature_size , [7, 7])
    output = output + _input
    return output

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def model(train_input_dir,
          valid_input_dir,
          saver_dir,
          iterations_size,
          LOSS_LOG_DIR,
          GPU_ID,
          BATCH_SIZE,
          feature_size=32,
          ):
    ################################################## Added by HiHiC ######
    ########################################################################
    start = time.time()

    train_epoch = [] 
    train_loss = []
    train_time = []
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    epoch_size = BATCH_SIZE
    ########################################################################
    ########################################################################

    input_x = tf.placeholder(tf.float32, [None, 40, 40, 1], name="input_x")
    input_y = tf.placeholder(tf.float32, [None, 28, 28, 1], name="input_y")

    output_x_2=slim.conv2d(input_x,feature_size,[7,7])
    output_x_2 = slim.conv2d(output_x_2, feature_size, [5, 5])
    output_x_2 = res_block(output_x_2)
    output_x_2 = tf.nn.relu(slim.conv2d(output_x_2, feature_size , [5, 5], padding="VALID"))
    output_x_2 = tf.nn.relu(slim.conv2d(output_x_2, feature_size , [3, 3], padding="VALID"))
    output_x_2 = tf.nn.relu(slim.conv2d(output_x_2, feature_size , [5, 5], padding="VALID"))
    output_x_2 = tf.nn.relu(slim.conv2d(output_x_2, feature_size , [3, 3], padding="VALID"))
    output_x_2 = res_block(output_x_2)
    output_x_2 = tf.nn.relu(slim.conv2d(output_x_2, feature_size, [5, 5]))
    output_x = tf.nn.relu(slim.conv2d(output_x_2, 1, [5, 5]))



    loss = tf.reduce_sum(tf.losses.mean_squared_error(input_y, output_x), name='loss')
    pearson = tf.contrib.metrics.streaming_pearson_correlation(output_x, input_y, name="pearson")[1]  # local variable

    tf.add_to_collection("output_x", output_x)
    tf.add_to_collection("loss", loss)


    # Scalar to keep track for loss
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("pearson", pearson)

    # Saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
    Saver = tf.train.Saver(tf.global_variables())
    if not os.path.exists(saver_dir):
        os.mkdir(saver_dir)

    merged = tf.summary.merge_all()
    step = tf.Variable(0, dtype=tf.int32, name="step")
    step_op = tf.assign(step, step + 1)


    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss)
    gpu_options = tf.GPUOptions(allow_growth=True)

    print("Begin training...")

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        train_writer = tf.summary.FileWriter(saver_dir + "/train", sess.graph)
        
        # input_list = os.listdir(train_input_dir)
        input_list = [ input for input in os.listdir(train_input_dir) if input.endswith('.npy')] ############################################################# Added by HiHiC ##
        mean_valid_loss = 1e6 #Initialize to a very large value
        # try:
        for epoch in range(iterations_size+1):
            for file in input_list:
                # x = np.load(train_input_dir + file).astype(np.float32)
                x = np.load(os.path.join(train_input_dir, file)).astype(np.float32) ######################################################################### Added by HiHiC ##
                x = np.reshape(x, [x.shape[0], 40, 68, 1])
                size_input = int(x.shape[0] / epoch_size) + 1
                np.random.shuffle(x)
                total_loss = 0  # Total loss per iteration
                for i in range(size_input):
                    if i * epoch_size + epoch_size <= x.shape[0]:
                        input = x[i * epoch_size:i * epoch_size + epoch_size, :, 0:40]
                        truth = x[i * epoch_size:i * epoch_size + epoch_size, 0:28, 40:68]
                        Loss, Pearson, Merged, Step, _ = sess.run(
                            [loss, pearson, merged, step_op, train_op],
                            feed_dict={input_x: input,
                                        input_y: truth,})

                        train_writer.add_summary(Merged, Step)
                        total_loss += Loss
                        if Step % 10 == 1:
                            print("in the %sth iteration  %sth step" % (epoch, Step), " the training loss is  ",
                                    Loss)
                            print("in the %sth iteration  %sth step" % (epoch, Step),
                                    " the training pearson is  ",
                                    Pearson)
                        # if Step % 50 == 1 and epoch >= 10:
                        #     Saver.save(sess, saver_dir + '/model/', global_step=step)
                print("the train file {0}  the train mean loss is {1}".format(file, total_loss / size_input))

            # if epoch > 20 and epoch % 5 == 2:
                # valid_file = os.listdir(valid_input_dir)
                # valid_file = valid_input_dir + valid_file[0]
                # x = np.load(valid_file).astype(np.float32)
                valid_input = [ input for input in os.listdir(valid_input_dir) if input.endswith('.npy')] ################################################### Added by HiHiC ##
                data_all = [np.load(os.path.join(valid_input_dir, fname), allow_pickle=True).astype(np.float32) for fname in valid_input] 
                x = np.concatenate(data_all, axis=0) ##########################################################################################################################
                x = np.reshape(x, [x.shape[0], 40, 68, 1])
                size_input = int(x.shape[0] / epoch_size) + 1
                np.random.shuffle(x)
                valid_total_loss = 0
                for i in range(size_input - 1):
                    input = x[i * epoch_size:i * epoch_size + epoch_size, :, 0:40]
                    truth = x[i * epoch_size:i * epoch_size + epoch_size, 0:28, 40:68]
                    Loss, Pearson, _ = sess.run(
                        [loss, pearson, train_op],
                        feed_dict={input_x: input,
                                    input_y: truth})
                    valid_total_loss += Loss
                    print("in the %sth iteration  %sth step" % (epoch, Step),
                            "the validing the loss is  ", Loss)
                    print("in the %sth iteration  %sth step" % (epoch, Step),
                            " the validing pearson is  ", Pearson)

                temp_mean_valid_loss = valid_total_loss / size_input
                print(temp_mean_valid_loss)
                # if temp_mean_valid_loss > mean_valid_loss:
                #     raise Exception("error is small!")
                # mean_valid_loss = temp_mean_valid_loss

            ################################################## Added by HiHiC ######
            ########################################################################            
            if epoch:
                sec = time.time()-start
                times = str(datetime.timedelta(seconds=sec))
                short = times.split(".")[0].replace(':','.')
                    
                train_epoch.append(epoch)
                train_time.append(short)        
                train_loss.append(f"{temp_mean_valid_loss:.10f}")
                
                ckpt_file = f"{str(epoch).zfill(5)}_{short}_{temp_mean_valid_loss:.10f}"
                Saver.save(sess, os.path.join(saver_dir, ckpt_file), global_step=step)
                np.save(os.path.join(LOSS_LOG_DIR, f'train_loss_SRHiC'), [train_epoch, train_time, train_loss])
            ########################################################################
            ########################################################################
                    
        # except Exception as e:
        #     print(e)
        # finally:
        #     Saver.save(sess, saver_dir + '/model/', global_step=step)
        #     print("training is over...")


if __name__ == '__main__':
    pass