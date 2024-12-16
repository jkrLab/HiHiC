#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os


################################################## Added by HiHiC ######
########################################################################

import argparse

parser = argparse.ArgumentParser(description='SRHiC prediction process')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')

required.add_argument('--root_dir', type=str, metavar='/HiHiC', required=True,
                      help='HiHiC directory')
required.add_argument('--model', type=str, default='SRHiC', metavar='SRHiC', required=True,
                      help='model name')
required.add_argument('--ckpt_file', type=str, metavar='[2]', required=True,
                      help='pretrained model (.meta)')
required.add_argument('--batch_size', type=int, default=64, metavar='[3]', required=True,
                      help='input batch size for training (default: 64)')
required.add_argument('--gpu_id', type=int, default=0, metavar='[4]', required=True, 
                      help='GPU ID for training (defalut: 0)')
required.add_argument('--input_data', type=str, metavar='[5]', required=True,
                      help='directory path of training model')
required.add_argument('--output_data_dir', type=str, default='./output_enhanced', metavar='[6]', required=True,
                      help='directory path for saving enhanced output (default: HiHiC/output_enhanced/)')
args = parser.parse_args()

if args.model == "HiCANR1":
    model = "HiCARN_1"
else:
    model = "HiCARN_2"

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.makedirs(args.output_data_dir, exist_ok=True) #######################
########################################################################


epoch_size=args.batch_size
 
# def predict(test_input_dir,
#             checkpoint_dir,
#             predict_save_dir,):
def predict(test_file,
            meta_file,
            predict_save_dir,):

    # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    # if ckpt and ckpt.model_checkpoint_path:
    #     graph_path = ckpt.model_checkpoint_path + '.meta'
        
    with tf.Session() as sess:
        
        # load model
        tf.local_variables_initializer().run()
        # saver = tf.train.import_meta_graph(graph_path)
        # saver.restore(sess, ckpt.model_checkpoint_path)
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, meta_file.split('.meta')[0])
        graph = tf.get_default_graph()
        output_x = tf.get_collection('output_x')[0]
        input_x = graph.get_operation_by_name('input_x').outputs[0]
        print("restoration is done...")
        # test_files=os.listdir(test_input_dir)
        print(test_file)
        # for test_file in test_files:
            # test=os.path.join(test_input_dir,test_file)
            # x = np.load(test).astype(np.float32)[:,:,:40]
        test = np.load(test_file)
        x = test['data'].astype(np.float32)[:,:,:40]
        x = np.reshape(x, [x.shape[0], 40, 40, 1])
        size = int(x.shape[0] / epoch_size) + 1
        Out = np.zeros([1, 28, 28, 1])
        for z in range(size):
            out = sess.run(
                [output_x],
                feed_dict={input_x: x[z * epoch_size:z * epoch_size + epoch_size]}
            )

            out_temp=np.array(out).reshape([-1,28,28,1])
            Out = np.concatenate((Out, out_temp), axis=0)
        # name ="enhanced_{0}".format(test_file)
        Out = Out[1:]
        # np.save(predict_save_dir + '/predict/' + name, Out)
        # np.savez_compressed(os.path.join(predict_save_dir, f"SRHiC_predict_{low_res}.npz"), data=Out)
        prefix = os.path.splitext(os.path.basename(test_file))[0]      
        th_model = meta_file.split('/')[-1].split('_')[0]
        file = os.path.join(predict_save_dir, f'{prefix}_{args.model}_{th_model}ep.npz')
        np.savez_compressed(file, data=Out, inds=test['inds'])


        # else:
        #     print("---no checkpoint found---")



# predict(test_input_dir, checkpoint_dir, predict_save_dir)
predict(args.input_data, args.ckpt_file, args.output_data_dir)